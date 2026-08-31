import type { InferenceBackend } from "../backends/backend.js";
import { CpuBackend } from "../backends/cpu.js";
import { invariant, NeedleError } from "../errors.js";
import type { LoadWeightsOptions, WeightSource } from "../weights/source.js";
import { loadWeights } from "../weights/source.js";
import { type CactWeights, parseCact } from "./cact.js";
import { argmax, logSoftmaxAt } from "./math.js";
import { NeedleRuntime, type RuntimeOptions } from "./runtime.js";
import { BOS_TOKEN_ID, EOS_TOKEN_ID, NeedleTokenizer } from "./tokenizer.js";

export type BackendSelection = "cpu" | "typegpu" | "vgpu" | "auto" | InferenceBackend;

export interface LoadModelOptions extends LoadWeightsOptions {
  readonly weights?: WeightSource;
  readonly backend?: BackendSelection;
  readonly backendOptions?: unknown;
  readonly runtime?: RuntimeOptions;
}

export interface GenerateOptions {
  readonly maxNewTokens?: number;
  readonly temperature?: number;
  readonly topK?: number;
  readonly seed?: number;
  readonly stop?: readonly string[];
  readonly signal?: AbortSignal;
  readonly sinkTokens?: number;
  readonly onToken?: (event: GeneratedToken) => void | Promise<void>;
}

export interface GeneratedToken {
  readonly id: number;
  readonly index: number;
  readonly piece: string;
  readonly text: string;
  readonly logProbability: number;
}

export interface GenerationResult {
  readonly text: string;
  readonly tokenIds: readonly number[];
  readonly tokenLogProbabilities: readonly number[];
  readonly finishReason: "eos" | "length" | "stop";
  readonly promptTokens: number;
  readonly generatedTokens: number;
  readonly elapsedMs: number;
  readonly tokensPerSecond: number;
}

/** Loaded Needle 2 weights, tokenizer, and one of the three compute backends. */
export class NeedleModel {
  readonly weights: CactWeights;
  readonly tokenizer: NeedleTokenizer;
  readonly backend: InferenceBackend;
  readonly runtimeOptions: RuntimeOptions;
  #disposed = false;

  private constructor(
    weights: CactWeights,
    tokenizer: NeedleTokenizer,
    backend: InferenceBackend,
    runtimeOptions: RuntimeOptions,
  ) {
    this.weights = weights;
    this.tokenizer = tokenizer;
    this.backend = backend;
    this.runtimeOptions = runtimeOptions;
  }

  static async load(options: LoadModelOptions = {}): Promise<NeedleModel> {
    const bytes = await loadWeights(options.weights, options);
    const weights = parseCact(bytes);
    const tokenizer = new NeedleTokenizer(weights.tokenizerBlob);
    invariant(
      tokenizer.vocabularySize === weights.geometry.vocabularySize,
      "INVALID_CACT",
      `Tokenizer has ${tokenizer.vocabularySize} pieces but model vocabulary is ${weights.geometry.vocabularySize}`,
    );
    const backend = await resolveBackend(weights, options.backend ?? "cpu", options.backendOptions);
    return new NeedleModel(weights, tokenizer, backend, options.runtime ?? {});
  }

  createRuntime(options: RuntimeOptions = {}): NeedleRuntime {
    invariant(!this.#disposed, "BACKEND_UNAVAILABLE", "Needle model has been disposed");
    return new NeedleRuntime(this.weights, this.backend, { ...this.runtimeOptions, ...options });
  }

  async generate(prompt: string, options: GenerateOptions = {}): Promise<GenerationResult> {
    invariant(!this.#disposed, "BACKEND_UNAVAILABLE", "Needle model has been disposed");
    const started = performance.now();
    const maximumNewTokens = Math.max(0, options.maxNewTokens ?? 96);
    const promptIds = [BOS_TOKEN_ID, ...this.tokenizer.encode(prompt)];
    const maximumLength = Math.min(
      this.weights.geometry.maximumSequenceLength,
      promptIds.length + maximumNewTokens,
    );
    if (promptIds.length >= maximumLength && maximumNewTokens > 0) {
      throw new NeedleError(
        "CONTEXT_OVERFLOW",
        `Prompt has ${promptIds.length} tokens and leaves no generation room in a ${maximumLength}-token context`,
      );
    }
    const runtime = this.createRuntime({ collectConfidence: false });
    runtime.reset({
      maximumLength: Math.max(promptIds.length, maximumLength),
      sinkLength: options.sinkTokens ?? 0,
    });
    let logits = await runtime.prefill(promptIds, options.signal);
    const random = xorshift32(options.seed ?? 0x6d2b79f5);
    const ids: number[] = [];
    const logProbabilities: number[] = [];
    let text = "";
    let finishReason: GenerationResult["finishReason"] = "length";

    for (let index = 0; index < maximumNewTokens && runtime.position < maximumLength; index++) {
      const token = sample(logits, options.temperature ?? 0, options.topK ?? 0, random);
      if (token === EOS_TOKEN_ID) {
        finishReason = "eos";
        break;
      }
      const logProbability = logSoftmaxAt(logits, token);
      ids.push(token);
      logProbabilities.push(logProbability);
      text = this.tokenizer.decode(ids);
      await options.onToken?.({
        id: token,
        index,
        piece: this.tokenizer.pieceText(token),
        text,
        logProbability,
      });
      if (options.stop?.some((stop) => text.endsWith(stop))) {
        finishReason = "stop";
        break;
      }
      if (index + 1 >= maximumNewTokens || runtime.position >= maximumLength) break;
      const next = await runtime.step(token, { signal: options.signal });
      invariant(next !== null, "INVALID_CACT", "Decode step did not return logits");
      logits = next;
    }

    const elapsedMs = performance.now() - started;
    return {
      text,
      tokenIds: ids,
      tokenLogProbabilities: logProbabilities,
      finishReason,
      promptTokens: promptIds.length,
      generatedTokens: ids.length,
      elapsedMs,
      tokensPerSecond: ids.length / Math.max(elapsedMs / 1000, 1e-9),
    };
  }

  async dispose(): Promise<void> {
    if (this.#disposed) return;
    this.#disposed = true;
    await this.backend.dispose();
  }
}

async function resolveBackend(
  weights: CactWeights,
  selection: BackendSelection,
  options: unknown,
): Promise<InferenceBackend> {
  if (typeof selection === "object") {
    invariant(
      selection.weights === weights,
      "BACKEND_UNAVAILABLE",
      "Custom backend was initialized for different CactWeights",
    );
    return selection;
  }
  if (selection === "cpu") return new CpuBackend(weights);
  if (selection === "typegpu") {
    try {
      const { createTypeGPUBackend } = await import("../backends/typegpu.js");
      return createTypeGPUBackend(weights, (options ?? {}) as never);
    } catch (cause) {
      if (cause instanceof NeedleError) throw cause;
      throw new NeedleError(
        "BACKEND_UNAVAILABLE",
        "The typegpu backend requires the optional `typegpu` peer dependency and WebGPU",
        { cause },
      );
    }
  }
  if (selection === "vgpu") {
    try {
      const { createVGPUBackend } = await import("../backends/vgpu.js");
      return createVGPUBackend(weights, (options ?? {}) as never);
    } catch (cause) {
      if (cause instanceof NeedleError) throw cause;
      throw new NeedleError(
        "BACKEND_UNAVAILABLE",
        "The vgpu backend requires the optional `vgpu` peer dependency and WebGPU",
        { cause },
      );
    }
  }

  // Auto intentionally falls through to the portable backend. Applications
  // can request a GPU backend explicitly to make a missing GPU actionable.
  if (typeof navigator !== "undefined" && navigator.gpu) {
    try {
      const { createTypeGPUBackend } = await import("../backends/typegpu.js");
      return await createTypeGPUBackend(weights, (options ?? {}) as never);
    } catch {
      try {
        const { createVGPUBackend } = await import("../backends/vgpu.js");
        return await createVGPUBackend(weights, (options ?? {}) as never);
      } catch {
        // Portable fallback below.
      }
    }
  }
  return new CpuBackend(weights);
}

function xorshift32(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 0x1_0000_0000;
  };
}

function sample(
  logits: Float32Array,
  temperature: number,
  topK: number,
  random: () => number,
): number {
  if (!(temperature > 0)) return argmax(logits);
  const count = topK > 0 ? Math.min(Math.floor(topK), logits.length) : logits.length;
  const indices = Array.from({ length: logits.length }, (_, index) => index);
  if (count < logits.length)
    indices.sort((left, right) => (logits[right] ?? 0) - (logits[left] ?? 0));
  indices.length = count;
  let maximum = Number.NEGATIVE_INFINITY;
  for (const index of indices) maximum = Math.max(maximum, (logits[index] ?? 0) / temperature);
  const probabilities = new Float64Array(count);
  let sum = 0;
  for (let index = 0; index < count; index++) {
    const probability = Math.exp((logits[indices[index] ?? 0] ?? 0) / temperature - maximum);
    probabilities[index] = probability;
    sum += probability;
  }
  let threshold = random() * sum;
  for (let index = 0; index < count; index++) {
    threshold -= probabilities[index] ?? 0;
    if (threshold <= 0) return indices[index] ?? 0;
  }
  return indices[count - 1] ?? 0;
}
