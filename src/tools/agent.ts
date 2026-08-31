import { NeedleError, invariant } from "../errors.js";
import {
  NeedleModel,
  type LoadModelOptions,
} from "../model/model.js";
import { argmax, logSoftmaxAt } from "../model/math.js";
import {
  BOS_TOKEN_ID,
  EOS_TOKEN_ID,
  TokenPieceType,
  CHAT_MARKERS,
} from "../model/tokenizer.js";
import { ToolCallGrammar, type GrammarToolCall } from "./grammar.js";
import { retrieveTools } from "./retrieval.js";
import {
  normalizeTools,
  serializeTools,
  type JsonSchema,
  type JsonValue,
  type NeedleTool,
  type ToolInput,
} from "./schema.js";

export interface NeedleFunctionCall {
  readonly id: string;
  readonly name: string;
  readonly arguments: Record<string, JsonValue>;
}

export interface CompletionMetrics {
  readonly promptTokens: number;
  readonly reasoningTokens: number;
  readonly callTokens: number;
  readonly prefillMs: number;
  readonly decodeMs: number;
  readonly prefillTokensPerSecond: number;
  readonly decodeTokensPerSecond: number;
}

export interface NeedleResponse {
  readonly type: "call" | "respond";
  readonly success: boolean;
  readonly error: string | null;
  readonly errorCode: string | null;
  readonly functionCalls: readonly NeedleFunctionCall[];
  /** Snake-case compatibility with the official Needle response envelope. */
  readonly function_calls: readonly NeedleFunctionCall[];
  readonly reasoning: string;
  readonly confidence: number | null;
  readonly metrics: CompletionMetrics;
  readonly rawCall: string;
  readonly selectedTools: readonly string[];
  readonly results?: readonly unknown[];
}

export interface NeedleOptions extends LoadModelOptions {
  readonly model?: NeedleModel;
  readonly tools?: readonly ToolInput[];
  readonly system?: string;
  readonly maxCallsPerTurn?: number;
  readonly reasoningTokens?: number;
  readonly prefixSinkTokens?: number;
  readonly toolTokenBudget?: number;
  readonly maximumRetrievedTools?: number;
}

export interface CompleteOptions {
  readonly maxNewTokens?: number;
  readonly reasoningTokens?: number;
  readonly signal?: AbortSignal;
}

export interface RunOptions extends CompleteOptions {
  readonly maxSteps?: number;
  readonly confidenceThreshold?: number;
}

export interface ExtractOptions extends Omit<NeedleOptions, "tools">, CompleteOptions {}

/** High-level, stateful tool-calling interface. */
export class Needle {
  readonly model: NeedleModel;
  readonly tools: readonly NeedleTool[];
  readonly system: string;
  readonly options: NeedleOptions;

  #transcript = "";
  #activeTools: NeedleTool[] = [];
  #awaitingToolResult = false;
  #callCounter = 0;
  #disposed = false;

  constructor(model: NeedleModel, options: Omit<NeedleOptions, "model" | "weights" | "backend" | "backendOptions"> = {}) {
    this.model = model;
    this.tools = normalizeTools(options.tools ?? []);
    this.system = options.system ?? "";
    this.options = options;
  }

  static async create(options: NeedleOptions = {}): Promise<Needle> {
    const model = options.model ?? await NeedleModel.load(options);
    return new Needle(model, options);
  }

  async complete(text: string, options: CompleteOptions = {}): Promise<NeedleResponse> {
    invariant(!this.#disposed, "BACKEND_UNAVAILABLE", "Needle agent has been disposed");
    invariant(this.tools.length > 0, "INVALID_TOOL_SCHEMA", "Needle tool calling requires at least one tool");
    const selected = this.#selectTools(text);
    const prompt = this.#renderPrompt(text, selected);
    const tokenizer = this.model.tokenizer;
    const promptIds = [BOS_TOKEN_ID, ...tokenizer.encode(prompt)];
    const maximumNewTokens = Math.max(1, options.maxNewTokens ?? 256);
    const maximumLength = Math.min(
      this.model.weights.geometry.maximumSequenceLength,
      promptIds.length + maximumNewTokens,
    );
    if (promptIds.length >= maximumLength) {
      throw new NeedleError("CONTEXT_OVERFLOW", `Conversation has ${promptIds.length} tokens and leaves no decode room in the ${this.model.weights.geometry.maximumSequenceLength}-token context`);
    }

    const runtime = this.model.createRuntime({ collectConfidence: true });
    const sinkLength = Math.min(
      this.options.prefixSinkTokens ?? 160,
      promptIds.length,
    );
    runtime.reset({ maximumLength, sinkLength });
    const prefillStarted = performance.now();
    let logits = await runtime.prefill(promptIds, options.signal);
    const prefillMs = performance.now() - prefillStarted;
    const decodeStarted = performance.now();
    const reasoningIds: number[] = [];
    const callIds: number[] = [];
    const callLogProbabilities: number[] = [];
    const toolCallStartId = tokenizer.idForPiece(CHAT_MARKERS.toolCallStart);
    const imEndId = tokenizer.idForPiece(CHAT_MARKERS.imEnd);
    const reasoningLimit = Math.min(
      options.reasoningTokens ?? this.options.reasoningTokens ?? 256,
      maximumNewTokens,
    );
    let openedCall = false;

    for (let index = 0; index < reasoningLimit && runtime.position < maximumLength; index++) {
      if (options.signal?.aborted) throw new NeedleError("GENERATION_ABORTED", "Needle completion was aborted", { cause: options.signal.reason });
      const token = argmax(logits);
      if (token === toolCallStartId) {
        const next = await runtime.step(token, { signal: options.signal });
        invariant(next !== null, "INVALID_CACT", "Tool-call marker step returned no logits");
        logits = next;
        openedCall = true;
        break;
      }
      if (token === EOS_TOKEN_ID || token === imEndId) break;
      reasoningIds.push(token);
      const next = await runtime.step(token, { signal: options.signal });
      invariant(next !== null, "INVALID_CACT", "Reasoning step returned no logits");
      logits = next;
    }

    let rawCall = "[]";
    let grammarCalls: readonly GrammarToolCall[] = [];
    if (openedCall) {
      let grammar = new ToolCallGrammar(selected, this.options.maxCallsPerTurn ?? 4);
      const chunks: Uint8Array[] = [];
      let outputBytes = 0;
      const remaining = maximumNewTokens - reasoningIds.length;
      for (let step = 0; step < remaining && runtime.position < maximumLength; step++) {
        let bestToken = -1;
        let bestLogit = Number.NEGATIVE_INFINITY;
        let bestGrammar: ToolCallGrammar | undefined;
        let bestBytes: Uint8Array | undefined;
        for (let token = 0; token < tokenizer.vocabularySize; token++) {
          const type = tokenizer.types[token];
          if (type !== TokenPieceType.Normal && type !== TokenPieceType.Byte) continue;
          const bytes = tokenizer.pieceBytes(token);
          if (bytes.byteLength === 0 || outputBytes + bytes.byteLength > 65_536) continue;
          const candidate = grammar.clone();
          if (!candidate.feed(bytes)) continue;
          const logit = logits[token] ?? Number.NEGATIVE_INFINITY;
          if (logit > bestLogit) {
            bestToken = token;
            bestLogit = logit;
            bestGrammar = candidate;
            bestBytes = bytes;
          }
        }
        if (bestToken < 0 || !bestGrammar || !bestBytes) {
          throw new NeedleError("GRAMMAR_DEAD_END", "No token can continue the schema-constrained tool call");
        }
        chunks.push(bestBytes);
        outputBytes += bestBytes.byteLength;
        callIds.push(bestToken);
        callLogProbabilities.push(logSoftmaxAt(logits, bestToken));
        grammar = bestGrammar;
        if (grammar.complete) {
          grammarCalls = grammar.calls;
          if (runtime.position < maximumLength) await runtime.step(bestToken, { wantLogits: false, signal: options.signal });
          break;
        }
        const next = await runtime.step(bestToken, { signal: options.signal });
        invariant(next !== null, "INVALID_CACT", "Constrained decode step returned no logits");
        logits = next;
        grammarCalls = grammar.calls;
      }
      const joined = new Uint8Array(outputBytes);
      let offset = 0;
      for (const chunk of chunks) {
        joined.set(chunk, offset);
        offset += chunk.byteLength;
      }
      rawCall = new TextDecoder("utf-8", { fatal: false }).decode(joined);
      if (!grammarCalls.length && rawCall) {
        try {
          const parsed = JSON.parse(rawCall) as GrammarToolCall[];
          if (Array.isArray(parsed)) grammarCalls = parsed;
        } catch {
          throw new NeedleError("GRAMMAR_DEAD_END", `Constrained output did not finish as JSON: ${rawCall}`);
        }
      }
    }

    const calls = grammarCalls.map((call) => ({
      id: `call_${++this.#callCounter}`,
      name: call.name,
      arguments: call.arguments,
    }));
    const rawReasoning = tokenizer.decode(reasoningIds);
    const reasoning = cleanReasoning(rawReasoning);
    const headConfidence = runtime.confidence();
    const decodeConfidence = callLogProbabilities.length
      ? Math.exp(Math.min(...callLogProbabilities))
      : undefined;
    const confidence = headConfidence === undefined
      ? decodeConfidence ?? null
      : decodeConfidence === undefined
        ? headConfidence
        : Math.min(headConfidence, decodeConfidence);
    const decodeMs = performance.now() - decodeStarted;
    const decodeTokens = reasoningIds.length + callIds.length;
    const metrics: CompletionMetrics = {
      promptTokens: promptIds.length,
      reasoningTokens: reasoningIds.length,
      callTokens: callIds.length,
      prefillMs,
      decodeMs,
      prefillTokensPerSecond: promptIds.length / Math.max(prefillMs / 1000, 1e-9),
      decodeTokensPerSecond: decodeTokens / Math.max(decodeMs / 1000, 1e-9),
    };

    this.#appendAssistant(rawReasoning, rawCall, calls.length > 0 && openedCall);
    this.#awaitingToolResult = calls.length > 0;
    const response: NeedleResponse = {
      type: calls.length > 0 ? "call" : "respond",
      success: true,
      error: null,
      errorCode: null,
      functionCalls: calls,
      function_calls: calls,
      reasoning,
      confidence,
      metrics,
      rawCall: openedCall ? rawCall : "[]",
      selectedTools: selected.map((tool) => tool.name),
    };
    return response;
  }

  async run(text: string, options: RunOptions = {}): Promise<NeedleResponse> {
    let response = await this.complete(text, options);
    const results: unknown[] = [];
    const maximumSteps = Math.max(1, options.maxSteps ?? 8);
    for (let step = 0; step < maximumSteps; step++) {
      if (response.type !== "call" || response.functionCalls.length === 0) break;
      if (options.confidenceThreshold !== undefined
        && response.confidence !== null
        && response.confidence < options.confidenceThreshold) break;
      const turnResults: unknown[] = [];
      for (const call of response.functionCalls) {
        const implementation = this.tools.find((tool) => tool.name === call.name)?.execute;
        if (!implementation) {
          const result = { error: `Unknown or unimplemented tool: ${call.name}` };
          turnResults.push(result);
          results.push(result);
          continue;
        }
        try {
          const result = await implementation(call.arguments, {
            ...(options.signal === undefined ? {} : { signal: options.signal }),
            callId: call.id,
            step,
          });
          turnResults.push(result);
          results.push(result);
        } catch (cause) {
          const result = { error: cause instanceof Error ? cause.message : String(cause) };
          turnResults.push(result);
          results.push(result);
        }
      }
      response = await this.complete(safeStringify(turnResults), options);
    }
    return { ...response, results };
  }

  reset(): void {
    this.#transcript = "";
    this.#activeTools = [];
    this.#awaitingToolResult = false;
  }

  async dispose(): Promise<void> {
    if (this.#disposed) return;
    this.#disposed = true;
    await this.model.dispose();
  }

  #selectTools(query: string): NeedleTool[] {
    if (this.#activeTools.length > 0) return this.#activeTools;
    this.#activeTools = retrieveTools(query, this.tools, {
      tokenizer: this.model.tokenizer,
      tokenBudget: this.options.toolTokenBudget ?? 180,
      maximumTools: this.options.maximumRetrievedTools ?? 5,
    });
    return this.#activeTools;
  }

  #renderPrompt(text: string, tools: readonly NeedleTool[]): string {
    if (this.#transcript.length === 0) {
      const system = this.system
        ? `${CHAT_MARKERS.imStart}system\n${this.system}${CHAT_MARKERS.imEnd}\n`
        : "";
      this.#transcript = `${system}${CHAT_MARKERS.imStart}user\n${CHAT_MARKERS.toolsStart}${serializeTools(tools)}${CHAT_MARKERS.toolsEnd}\n${text}${CHAT_MARKERS.imEnd}\n`;
    } else if (this.#awaitingToolResult) {
      this.#transcript += `${CHAT_MARKERS.imStart}user\n${CHAT_MARKERS.toolResultStart}${text}${CHAT_MARKERS.toolResultEnd}${CHAT_MARKERS.imEnd}\n`;
    } else {
      this.#transcript += `${CHAT_MARKERS.imStart}user\n${text}${CHAT_MARKERS.imEnd}\n`;
    }
    return `${this.#transcript}${CHAT_MARKERS.imStart}assistant\n`;
  }

  #appendAssistant(reasoning: string, rawCall: string, called: boolean): void {
    this.#transcript += `${CHAT_MARKERS.imStart}assistant\n${reasoning}`;
    if (called) this.#transcript += `${CHAT_MARKERS.toolCallStart}${rawCall}${CHAT_MARKERS.toolCallEnd}`;
    this.#transcript += `${CHAT_MARKERS.imEnd}\n`;
  }
}

function cleanReasoning(raw: string): string {
  const start = raw.indexOf(CHAT_MARKERS.thinkStart);
  const end = raw.lastIndexOf(CHAT_MARKERS.thinkEnd);
  if (start >= 0 && end > start) {
    return raw.slice(start + CHAT_MARKERS.thinkStart.length, end).trim();
  }
  return raw.replaceAll(CHAT_MARKERS.thinkStart, "").replaceAll(CHAT_MARKERS.thinkEnd, "").trim();
}

function safeStringify(value: unknown): string {
  const seen = new WeakSet<object>();
  return JSON.stringify(value, (_key, current: unknown) => {
    if (typeof current === "bigint") return current.toString();
    if (current && typeof current === "object") {
      if (seen.has(current)) return "[Circular]";
      seen.add(current);
    }
    return current;
  }) ?? "null";
}

export async function createNeedle(options: NeedleOptions = {}): Promise<Needle> {
  return Needle.create(options);
}

export async function extract<Result extends object = Record<string, JsonValue>>(
  text: string,
  schema: JsonSchema,
  options: ExtractOptions = {},
): Promise<Result | null> {
  const parameters: JsonSchema & { type: "object" } = {
    ...schema,
    type: "object",
  };
  const agent = await Needle.create({
    ...options,
    tools: [{
      name: schema.title ?? "extract",
      description: schema.description ?? "Extract the structured record from the supplied text",
      parameters,
    }],
  });
  try {
    const response = await agent.complete(text, options);
    return (response.functionCalls[0]?.arguments as Result | undefined) ?? null;
  } finally {
    await agent.dispose();
  }
}
