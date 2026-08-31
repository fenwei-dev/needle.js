import {
  InvalidPromptError,
  type LanguageModelV4,
  type LanguageModelV4CallOptions,
  type LanguageModelV4Content,
  type LanguageModelV4GenerateResult,
  type LanguageModelV4StreamPart,
  type LanguageModelV4StreamResult,
  type SharedV4ProviderMetadata,
  type SharedV4Warning,
} from "@ai-sdk/provider";
import {
  Needle,
  type NeedleFunctionCall,
  type NeedleModel,
  type NeedleOptions,
  type NeedleResponse,
  type NeedleTool,
} from "needle.js";
import type { NeedleModelPool } from "./model-pool.js";
import { type ConvertedNeedlePrompt, convertPrompt } from "./prompt.js";

export type NeedleModelId = "needle-2" | "needle2" | (string & {});

export interface NeedleAgentLike {
  complete(
    text: string,
    options?: { maxNewTokens?: number; reasoningTokens?: number; signal?: AbortSignal },
  ): Promise<NeedleResponse>;
}

export interface NeedleLanguageModelConfig {
  readonly provider: string;
  readonly pool: NeedleModelPool;
  readonly agentOptions?: Omit<
    NeedleOptions,
    "model" | "weights" | "backend" | "backendOptions" | "tools" | "system"
  >;
  readonly createAgent?: (model: NeedleModel, options: NeedleOptions) => NeedleAgentLike;
  readonly now?: () => Date;
  readonly generateId?: () => string;
}

export class NeedleLanguageModel implements LanguageModelV4 {
  readonly specificationVersion = "v4";
  readonly supportedUrls = {};

  constructor(
    readonly modelId: NeedleModelId,
    private readonly config: NeedleLanguageModelConfig,
  ) {}

  get provider(): string {
    return this.config.provider;
  }

  async doGenerate(options: LanguageModelV4CallOptions): Promise<LanguageModelV4GenerateResult> {
    const converted = convertPrompt(options.prompt);
    const functionTools = (options.tools ?? []).filter((tool) => tool.type === "function");
    const providerToolCount = (options.tools?.length ?? 0) - functionTools.length;
    const jsonSchema =
      options.responseFormat?.type === "json" ? options.responseFormat.schema : undefined;
    const forceRaw = options.toolChoice?.type === "none";
    const mode = jsonSchema ? "json" : functionTools.length > 0 && !forceRaw ? "tools" : "raw";
    const warnings = collectWarnings(options, mode, providerToolCount);
    const model = await this.config.pool.get();

    if (mode === "raw") return this.#generateRaw(model, converted, options, warnings);

    const tools = jsonSchema
      ? [jsonResponseTool(options)]
      : prepareTools(functionTools, options.toolChoice);
    const response = await this.#generateTools(model, converted, tools, options);
    const content = jsonSchema
      ? jsonResponseContent(response)
      : toolResponseContent(response, converted.latestToolResult);
    const usage = responseUsage(response);
    const timestamp = this.#now();
    const responseId = this.#generateId();

    return {
      content,
      finishReason: {
        unified: response.functionCalls.length > 0 && !jsonSchema ? "tool-calls" : "stop",
        raw: response.type,
      },
      usage,
      warnings,
      providerMetadata: responseMetadata(response),
      request: { body: { mode, local: true } },
      response: { id: responseId, timestamp, modelId: this.modelId },
    };
  }

  async doStream(options: LanguageModelV4CallOptions): Promise<LanguageModelV4StreamResult> {
    const result = await this.doGenerate(options);
    const responseId = result.response?.id ?? this.#generateId();
    const timestamp = result.response?.timestamp ?? this.#now();
    const modelId = result.response?.modelId ?? this.modelId;
    let partCounter = 0;

    const stream = new ReadableStream<LanguageModelV4StreamPart>({
      start: (controller) => {
        controller.enqueue({ type: "stream-start", warnings: result.warnings });
        controller.enqueue({ type: "response-metadata", id: responseId, timestamp, modelId });
        for (const part of result.content) {
          if (part.type === "text") {
            const id = `${responseId}-text-${++partCounter}`;
            const metadata = part.providerMetadata
              ? { providerMetadata: part.providerMetadata }
              : {};
            controller.enqueue({ type: "text-start", id, ...metadata });
            controller.enqueue({ type: "text-delta", id, delta: part.text, ...metadata });
            controller.enqueue({ type: "text-end", id, ...metadata });
          } else if (part.type === "reasoning") {
            const id = `${responseId}-reasoning-${++partCounter}`;
            const metadata = part.providerMetadata
              ? { providerMetadata: part.providerMetadata }
              : {};
            controller.enqueue({ type: "reasoning-start", id, ...metadata });
            controller.enqueue({ type: "reasoning-delta", id, delta: part.text, ...metadata });
            controller.enqueue({ type: "reasoning-end", id, ...metadata });
          } else if (part.type === "tool-call") {
            controller.enqueue(part);
          }
        }
        if (options.includeRawChunks) controller.enqueue({ type: "raw", rawValue: result });
        controller.enqueue({
          type: "finish",
          usage: result.usage,
          finishReason: result.finishReason,
          ...(result.providerMetadata === undefined
            ? {}
            : { providerMetadata: result.providerMetadata }),
        });
        controller.close();
      },
    });

    return { stream };
  }

  async #generateRaw(
    model: NeedleModel,
    converted: ConvertedNeedlePrompt,
    options: LanguageModelV4CallOptions,
    warnings: SharedV4Warning[],
  ): Promise<LanguageModelV4GenerateResult> {
    const generated = await model.generate(converted.rawPrompt, {
      maxNewTokens: options.maxOutputTokens ?? 256,
      temperature: options.temperature ?? 0,
      topK: options.topK ?? 0,
      ...(options.seed === undefined ? {} : { seed: options.seed }),
      ...(options.stopSequences === undefined ? {} : { stop: options.stopSequences }),
      ...(options.abortSignal === undefined ? {} : { signal: options.abortSignal }),
    });
    return {
      content: [{ type: "text", text: generated.text }],
      finishReason: {
        unified: generated.finishReason === "length" ? "length" : "stop",
        raw: generated.finishReason,
      },
      usage: {
        inputTokens: {
          total: generated.promptTokens,
          noCache: generated.promptTokens,
          cacheRead: 0,
          cacheWrite: 0,
        },
        outputTokens: {
          total: generated.generatedTokens,
          text: generated.generatedTokens,
          reasoning: 0,
        },
      },
      warnings,
      providerMetadata: {
        needle: {
          backend: model.backend.kind,
          tokensPerSecond: generated.tokensPerSecond,
          elapsedMs: generated.elapsedMs,
        },
      },
      request: { body: { mode: "raw", local: true } },
      response: { id: this.#generateId(), timestamp: this.#now(), modelId: this.modelId },
    };
  }

  async #generateTools(
    model: NeedleModel,
    converted: ConvertedNeedlePrompt,
    tools: NeedleTool[],
    options: LanguageModelV4CallOptions,
  ): Promise<NeedleResponse> {
    const createAgent =
      this.config.createAgent ??
      ((currentModel, agentOptions) => new Needle(currentModel, agentOptions));
    const providerOptions = readNeedleProviderOptions(options.providerOptions?.needle);
    const agent = createAgent(model, {
      ...this.config.agentOptions,
      ...providerOptions,
      model,
      tools,
      system: converted.system,
    });
    let response: NeedleResponse | undefined;
    for (const event of converted.events) {
      response = await agent.complete(event.text, {
        maxNewTokens: options.maxOutputTokens ?? 256,
        ...(providerOptions.reasoningTokens === undefined
          ? {}
          : { reasoningTokens: providerOptions.reasoningTokens }),
        ...(options.abortSignal === undefined ? {} : { signal: options.abortSignal }),
      });
    }
    if (!response) {
      throw new InvalidPromptError({
        prompt: options.prompt,
        message: "Needle requires a user message or tool result",
      });
    }
    return response;
  }

  #now(): Date {
    return this.config.now?.() ?? new Date();
  }

  #generateId(): string {
    return this.config.generateId?.() ?? createId();
  }
}

function prepareTools(
  tools: Array<
    Extract<NonNullable<LanguageModelV4CallOptions["tools"]>[number], { type: "function" }>
  >,
  choice: LanguageModelV4CallOptions["toolChoice"],
): NeedleTool[] {
  const selected =
    choice?.type === "tool" ? tools.filter((tool) => tool.name === choice.toolName) : tools;
  return selected.map((tool) => ({
    name: tool.name,
    ...(tool.description === undefined ? {} : { description: tool.description }),
    parameters: {
      ...(tool.inputSchema as object),
      type: "object",
    },
  })) as NeedleTool[];
}

function jsonResponseTool(options: LanguageModelV4CallOptions): NeedleTool {
  const format = options.responseFormat?.type === "json" ? options.responseFormat : undefined;
  const rawName = format?.name ?? "json_response";
  const name = rawName.replaceAll(/[^a-zA-Z0-9_-]/g, "_").slice(0, 128) || "json_response";
  return {
    name,
    description: format?.description ?? "Return the requested structured JSON response",
    parameters: {
      ...(format?.schema as object),
      type: "object",
    },
  } as NeedleTool;
}

function toolResponseContent(
  response: NeedleResponse,
  latestToolResult?: string,
): LanguageModelV4Content[] {
  const content: LanguageModelV4Content[] = [];
  if (response.reasoning) content.push({ type: "reasoning", text: response.reasoning });
  for (const call of response.functionCalls) content.push(toolCallContent(call));
  if (response.functionCalls.length === 0) {
    content.push({ type: "text", text: latestToolResult ?? response.rawCall });
  }
  return content;
}

function jsonResponseContent(response: NeedleResponse): LanguageModelV4Content[] {
  const content: LanguageModelV4Content[] = [];
  if (response.reasoning) content.push({ type: "reasoning", text: response.reasoning });
  content.push({
    type: "text",
    text: JSON.stringify(response.functionCalls[0]?.arguments ?? {}),
  });
  return content;
}

function toolCallContent(call: NeedleFunctionCall): LanguageModelV4Content {
  return {
    type: "tool-call",
    toolCallId: call.id,
    toolName: call.name,
    input: JSON.stringify(call.arguments),
  };
}

function responseUsage(response: NeedleResponse): LanguageModelV4GenerateResult["usage"] {
  const outputTokens = response.metrics.reasoningTokens + response.metrics.callTokens;
  return {
    inputTokens: {
      total: response.metrics.promptTokens,
      noCache: response.metrics.promptTokens,
      cacheRead: 0,
      cacheWrite: 0,
    },
    outputTokens: {
      total: outputTokens,
      text: 0,
      reasoning: response.metrics.reasoningTokens,
    },
  };
}

function responseMetadata(response: NeedleResponse): SharedV4ProviderMetadata {
  return {
    needle: {
      confidence: response.confidence,
      selectedTools: [...response.selectedTools],
      prefillTokensPerSecond: response.metrics.prefillTokensPerSecond,
      decodeTokensPerSecond: response.metrics.decodeTokensPerSecond,
      rawCall: response.rawCall,
    },
  };
}

function collectWarnings(
  options: LanguageModelV4CallOptions,
  mode: "raw" | "tools" | "json",
  providerToolCount: number,
): SharedV4Warning[] {
  const warnings: SharedV4Warning[] = [];
  const unsupported = (enabled: boolean, feature: string, details?: string) => {
    if (enabled) warnings.push({ type: "unsupported", feature, ...(details ? { details } : {}) });
  };
  unsupported(options.topP !== undefined, "topP");
  unsupported(options.presencePenalty !== undefined, "presencePenalty");
  unsupported(options.frequencyPenalty !== undefined, "frequencyPenalty");
  unsupported(providerToolCount > 0, "provider-defined tools");
  unsupported(
    mode !== "raw" && options.temperature !== undefined && options.temperature !== 0,
    "temperature for constrained tool calls",
  );
  unsupported(mode !== "raw" && options.topK !== undefined, "topK for constrained tool calls");
  unsupported(mode !== "raw" && options.seed !== undefined, "seed for constrained tool calls");
  unsupported(
    mode !== "raw" && options.stopSequences !== undefined,
    "stopSequences for constrained tool calls",
  );
  unsupported(
    options.reasoning !== undefined && options.reasoning !== "provider-default",
    "reasoning effort",
  );
  if (options.toolChoice?.type === "required") {
    warnings.push({
      type: "compatibility",
      feature: "required tool choice",
      details: "Needle can still decline when no declared tool applies.",
    });
  }
  return warnings;
}

function readNeedleProviderOptions(
  value: unknown,
): Pick<
  NeedleOptions,
  | "maxCallsPerTurn"
  | "reasoningTokens"
  | "prefixSinkTokens"
  | "toolTokenBudget"
  | "maximumRetrievedTools"
> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  const record = value as Record<string, unknown>;
  return {
    ...numberOption(record, "maxCallsPerTurn"),
    ...numberOption(record, "reasoningTokens"),
    ...numberOption(record, "prefixSinkTokens"),
    ...numberOption(record, "toolTokenBudget"),
    ...numberOption(record, "maximumRetrievedTools"),
  };
}

function numberOption(record: Record<string, unknown>, key: string): Record<string, number> {
  const value = record[key];
  return typeof value === "number" && Number.isFinite(value) ? { [key]: value } : {};
}

let nextId = 0;
function createId(): string {
  nextId += 1;
  return `needle-${Date.now().toString(36)}-${nextId.toString(36)}`;
}
