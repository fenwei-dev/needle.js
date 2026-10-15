import {
  type AssistantMessage,
  type AssistantMessageEventStream,
  type Context,
  calculateCost,
  createAssistantMessageEventStream,
  type Model,
  type ProviderStreams,
  type SimpleStreamOptions,
  type StreamOptions,
  type ToolCall,
} from "@earendil-works/pi-ai";
import {
  Needle,
  type NeedleModel,
  type NeedleOptions,
  type NeedleResponse,
  type NeedleTool,
} from "needle.js";
import { convertContext } from "./context.js";
import { type NeedlePiModelOptions, NeedlePiModelPool } from "./model-pool.js";

export const NEEDLE_PI_API = "needle-local" as const;
export type NeedlePiApiId = typeof NEEDLE_PI_API;

export interface NeedlePiAgentLike {
  complete(
    text: string,
    options?: { maxNewTokens?: number; reasoningTokens?: number; signal?: AbortSignal },
  ): Promise<NeedleResponse>;
}

export interface NeedlePiApiOptions extends NeedlePiModelOptions {
  readonly agentOptions?: Omit<
    NeedleOptions,
    "model" | "weights" | "backend" | "backendOptions" | "tools" | "system"
  >;
  readonly _internal?: {
    readonly createAgent?: (model: NeedleModel, options: NeedleOptions) => NeedlePiAgentLike;
    readonly now?: () => number;
  };
}

export interface NeedlePiApi extends ProviderStreams {
  dispose(): Promise<void>;
}

export function createNeedlePiApi(options: NeedlePiApiOptions = {}): NeedlePiApi {
  const pool = new NeedlePiModelPool(options);
  const run = (
    model: Model<NeedlePiApiId>,
    context: Context,
    streamOptions?: StreamOptions | SimpleStreamOptions,
  ) => streamNeedle(pool, options, model, context, streamOptions);

  return {
    stream: (model, context, streamOptions) =>
      run(model as Model<NeedlePiApiId>, context, streamOptions),
    streamSimple: (model, context, streamOptions) =>
      run(model as Model<NeedlePiApiId>, context, streamOptions),
    dispose: () => pool.dispose(),
  };
}

function streamNeedle(
  pool: NeedlePiModelPool,
  settings: NeedlePiApiOptions,
  model: Model<NeedlePiApiId>,
  context: Context,
  options?: StreamOptions | SimpleStreamOptions,
): AssistantMessageEventStream {
  const stream = createAssistantMessageEventStream();
  const output = createOutput(model, settings._internal?.now?.() ?? Date.now());
  stream.push({ type: "start", partial: output });

  void (async () => {
    try {
      if (options?.signal?.aborted) throw new Error("Request was aborted");
      await options?.onPayload?.({ local: true, context }, model);
      const needleModel = await pool.get();
      const converted = convertContext(context);

      if (!context.tools || context.tools.length === 0) {
        const generated = await needleModel.generate(converted.rawPrompt, {
          maxNewTokens: options?.maxTokens ?? model.maxTokens,
          temperature: options?.temperature ?? 0,
          ...(options?.signal === undefined ? {} : { signal: options.signal }),
        });
        output.usage.input = generated.promptTokens;
        output.usage.output = generated.generatedTokens;
        output.usage.totalTokens = generated.promptTokens + generated.generatedTokens;
        calculateCost(model, output.usage);
        pushText(stream, output, generated.text);
        output.stopReason = generated.finishReason === "length" ? "length" : "stop";
        output.rawStopReason = generated.finishReason;
      } else {
        const response = await completeWithTools(
          needleModel,
          settings,
          converted,
          context,
          options,
        );
        applyResponseUsage(model, output, response);
        if (response.reasoning) pushThinking(stream, output, response.reasoning);
        for (const call of response.functionCalls) {
          pushToolCall(stream, output, {
            type: "toolCall",
            id: call.id,
            name: call.name,
            arguments: call.arguments,
          });
        }
        if (response.functionCalls.length > 0) {
          output.stopReason = "toolUse";
        } else {
          pushText(stream, output, converted.latestToolResult ?? response.rawCall);
          output.stopReason = "stop";
        }
        output.rawStopReason = response.type;
      }

      stream.push({ type: "done", reason: output.stopReason, message: output });
      stream.end();
    } catch (error) {
      output.stopReason = options?.signal?.aborted ? "aborted" : "error";
      output.errorMessage = error instanceof Error ? error.message : String(error);
      stream.push({ type: "error", reason: output.stopReason, error: output });
      stream.end();
    }
  })();

  return stream;
}

async function completeWithTools(
  model: NeedleModel,
  settings: NeedlePiApiOptions,
  converted: ReturnType<typeof convertContext>,
  context: Context,
  options?: StreamOptions | SimpleStreamOptions,
): Promise<NeedleResponse> {
  const tools = context.tools?.map((tool) => ({
    name: tool.name,
    description: tool.description,
    parameters: {
      ...(tool.parameters as object),
      type: "object",
    },
  })) as NeedleTool[];
  const createAgent =
    settings._internal?.createAgent ??
    ((currentModel: NeedleModel, agentOptions: NeedleOptions) =>
      new Needle(currentModel, agentOptions));
  const reasoningTokens =
    "reasoning" in (options ?? {})
      ? tokensForReasoning((options as SimpleStreamOptions).reasoning)
      : undefined;
  const agent = createAgent(model, {
    ...settings.agentOptions,
    model,
    tools,
    system: converted.system,
  });
  let response: NeedleResponse | undefined;
  for (const event of converted.events) {
    response = await agent.complete(event.text, {
      maxNewTokens: options?.maxTokens ?? 256,
      ...(reasoningTokens === undefined ? {} : { reasoningTokens }),
      ...(options?.signal === undefined ? {} : { signal: options.signal }),
    });
  }
  if (!response) throw new Error("Needle requires a user message or tool result");
  return response;
}

function createOutput(model: Model<NeedlePiApiId>, timestamp: number): AssistantMessage {
  return {
    role: "assistant",
    content: [],
    api: model.api,
    provider: model.provider,
    model: model.id,
    usage: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
    stopReason: "pending",
    timestamp,
  };
}

function applyResponseUsage(
  model: Model<NeedlePiApiId>,
  output: AssistantMessage,
  response: NeedleResponse,
): void {
  output.usage.input = response.metrics.promptTokens;
  output.usage.output = response.metrics.reasoningTokens + response.metrics.callTokens;
  output.usage.reasoning = response.metrics.reasoningTokens;
  output.usage.totalTokens = output.usage.input + output.usage.output;
  calculateCost(model, output.usage);
}

function pushText(
  stream: AssistantMessageEventStream,
  output: AssistantMessage,
  text: string,
): void {
  const block = { type: "text" as const, text };
  output.content.push(block);
  const contentIndex = output.content.length - 1;
  stream.push({ type: "text_start", contentIndex, partial: output });
  stream.push({ type: "text_delta", contentIndex, delta: text, partial: output });
  stream.push({ type: "text_end", contentIndex, content: text, partial: output });
}

function pushThinking(
  stream: AssistantMessageEventStream,
  output: AssistantMessage,
  thinking: string,
): void {
  const block = { type: "thinking" as const, thinking };
  output.content.push(block);
  const contentIndex = output.content.length - 1;
  stream.push({ type: "thinking_start", contentIndex, partial: output });
  stream.push({ type: "thinking_delta", contentIndex, delta: thinking, partial: output });
  stream.push({ type: "thinking_end", contentIndex, content: thinking, partial: output });
}

function pushToolCall(
  stream: AssistantMessageEventStream,
  output: AssistantMessage,
  toolCall: ToolCall,
): void {
  output.content.push(toolCall);
  const contentIndex = output.content.length - 1;
  const delta = JSON.stringify(toolCall.arguments);
  stream.push({ type: "toolcall_start", contentIndex, partial: output });
  stream.push({ type: "toolcall_delta", contentIndex, delta, partial: output });
  stream.push({ type: "toolcall_end", contentIndex, toolCall, partial: output });
}

function tokensForReasoning(level: SimpleStreamOptions["reasoning"]): number | undefined {
  if (!level) return undefined;
  return {
    minimal: 32,
    low: 64,
    medium: 128,
    high: 256,
    xhigh: 384,
    max: 512,
  }[level];
}
