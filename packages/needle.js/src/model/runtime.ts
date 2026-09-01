import type {
  FusedAttentionSession,
  InferenceBackend,
  MatvecRequest,
  ResidentTokenSelection,
} from "../backends/backend.js";
import { denseMatvec } from "../backends/cq.js";
import { invariant, NeedleError } from "../errors.js";
import type { CactLayer, CactProbeHead, CactWeights } from "./cact.js";
import {
  applyRope,
  argmax,
  hadamardMlp,
  logSoftmaxAt,
  rmsNorm,
  rmsUnit,
  sigmoid,
  sinkhorn,
  softmaxInPlace,
} from "./math.js";

const ENGRAM_SEED = 0x9e37_79b9;
const ENGRAM_PRIME = 0x0100_0193;

export interface RuntimeOptions {
  /** Int8 matches the published engine. Float32 is useful for reference diffs. */
  readonly kvCache?: "int8" | "float32";
  /** Compute the optional post-hoc confidence head with online probe pooling. */
  readonly collectConfidence?: boolean;
  readonly onLayer?: (event: {
    readonly position: number;
    readonly layer: number;
    readonly layers: number;
  }) => void;
}

export interface RuntimeResetOptions {
  readonly maximumLength: number;
  /** Prefix tokens pinned alongside the sliding attention window. */
  readonly sinkLength?: number;
}

class OnlineProbePool {
  readonly #head: CactProbeHead;
  readonly #dimension: number;
  readonly #maximum: Float64Array;
  readonly #denominator: Float64Array;
  readonly #weighted: Float64Array;

  constructor(head: CactProbeHead, dimension: number) {
    this.#head = head;
    this.#dimension = dimension;
    this.#maximum = new Float64Array(head.probeCount);
    this.#maximum.fill(Number.NEGATIVE_INFINITY);
    this.#denominator = new Float64Array(head.probeCount);
    this.#weighted = new Float64Array(head.probeCount * dimension);
  }

  add(cell: Float32Array): void {
    const scale = 1 / Math.sqrt(this.#dimension);
    for (let probe = 0; probe < this.#head.probeCount; probe++) {
      let score = 0;
      const probeOffset = probe * this.#dimension;
      for (let index = 0; index < this.#dimension; index++) {
        score += (cell[index] ?? 0) * (this.#head.probes[probeOffset + index] ?? 0);
      }
      score *= scale;
      const previousMaximum = this.#maximum[probe] ?? Number.NEGATIVE_INFINITY;
      const weightedOffset = probe * this.#dimension;
      if (score <= previousMaximum) {
        const weight = Math.exp(score - previousMaximum);
        this.#denominator[probe] = (this.#denominator[probe] ?? 0) + weight;
        for (let index = 0; index < this.#dimension; index++) {
          this.#weighted[weightedOffset + index] =
            (this.#weighted[weightedOffset + index] ?? 0) + weight * (cell[index] ?? 0);
        }
      } else {
        const oldWeight = Number.isFinite(previousMaximum) ? Math.exp(previousMaximum - score) : 0;
        this.#denominator[probe] = (this.#denominator[probe] ?? 0) * oldWeight + 1;
        for (let index = 0; index < this.#dimension; index++) {
          this.#weighted[weightedOffset + index] =
            (this.#weighted[weightedOffset + index] ?? 0) * oldWeight + (cell[index] ?? 0);
        }
        this.#maximum[probe] = score;
      }
    }
  }

  output(): Float32Array {
    const pooled = new Float32Array(this.#head.probeCount * this.#dimension);
    for (let probe = 0; probe < this.#head.probeCount; probe++) {
      const denominator = Math.max(this.#denominator[probe] ?? 0, 1e-30);
      const offset = probe * this.#dimension;
      for (let index = 0; index < this.#dimension; index++) {
        pooled[offset + index] = (this.#weighted[offset + index] ?? 0) / denominator;
      }
    }
    return denseMatvec(this.#head.projection, this.#head.outputSize, pooled, this.#head.bias);
  }
}

/** Incremental, one-token Needle 2 decoder shared by all three backends. */
export class NeedleRuntime {
  readonly weights: CactWeights;
  readonly backend: InferenceBackend;
  readonly options: RuntimeOptions;

  #maximumLength = 0;
  #sinkLength = 0;
  #kvAllocation = 0;
  #position = 0;
  #history: number[] = [];
  #keyCacheInt8: Int8Array | undefined;
  #valueCacheInt8: Int8Array | undefined;
  #keyCacheFloat: Float32Array | undefined;
  #valueCacheFloat: Float32Array | undefined;
  #keyScale = new Float32Array(0);
  #valueScale = new Float32Array(0);
  #engramRing = new Float32Array(0);
  #engramValid = new Uint8Array(0);
  #engramDepth = 0;
  #engramPosition = 0;
  #confidencePool: OnlineProbePool | undefined;
  #fusedAttention: FusedAttentionSession | undefined;
  #fusedAttentionEnabled = false;
  #usesResidentConfidence = false;
  #selectionRequest: Uint32Array | null | undefined;
  #selectionResult: ResidentTokenSelection | undefined;

  constructor(weights: CactWeights, backend: InferenceBackend, options: RuntimeOptions = {}) {
    this.weights = weights;
    this.backend = backend;
    this.options = options;
    this.#fusedAttention = backend.createFusedAttentionSession?.();
  }

  get position(): number {
    return this.#position;
  }

  get maximumLength(): number {
    return this.#maximumLength;
  }

  reset(options: RuntimeResetOptions): void {
    const geometry = this.weights.geometry;
    invariant(
      Number.isInteger(options.maximumLength) &&
        options.maximumLength > 0 &&
        options.maximumLength <= geometry.maximumSequenceLength,
      "CONTEXT_OVERFLOW",
      `Runtime length ${options.maximumLength} is outside model limit ${geometry.maximumSequenceLength}`,
    );
    this.#maximumLength = options.maximumLength;
    this.#sinkLength = Math.min(Math.max(options.sinkLength ?? 0, 0), options.maximumLength);
    this.#position = 0;
    this.#history = [];
    this.#selectionRequest = undefined;
    this.#selectionResult = undefined;

    const window = geometry.kvWindow;
    this.#kvAllocation =
      window > 0
        ? Math.min(options.maximumLength, this.#sinkLength + window)
        : options.maximumLength;
    this.#fusedAttentionEnabled =
      this.#fusedAttention?.reset({
        maximumLength: options.maximumLength,
        sinkLength: this.#sinkLength,
        kvCache: this.options.kvCache ?? "int8",
        collectConfidence: this.options.collectConfidence !== false,
      }) ?? false;
    if (this.#fusedAttentionEnabled) {
      this.#keyCacheInt8 = undefined;
      this.#valueCacheInt8 = undefined;
      this.#keyCacheFloat = undefined;
      this.#valueCacheFloat = undefined;
      this.#keyScale = new Float32Array(0);
      this.#valueScale = new Float32Array(0);
    } else {
      const kvElements =
        geometry.numberOfLayers *
        geometry.numberOfKVHeads *
        this.#kvAllocation *
        geometry.headDimension;
      const scaleElements = geometry.numberOfLayers * geometry.numberOfKVHeads * this.#kvAllocation;
      if ((this.options.kvCache ?? "int8") === "float32") {
        this.#keyCacheFloat = new Float32Array(kvElements);
        this.#valueCacheFloat = new Float32Array(kvElements);
        this.#keyCacheInt8 = undefined;
        this.#valueCacheInt8 = undefined;
      } else {
        this.#keyCacheInt8 = new Int8Array(kvElements);
        this.#valueCacheInt8 = new Int8Array(kvElements);
        this.#keyCacheFloat = undefined;
        this.#valueCacheFloat = undefined;
      }
      this.#keyScale = new Float32Array(scaleElements);
      this.#valueScale = new Float32Array(scaleElements);
    }

    const maximumOrder = Math.max(1, ...geometry.engramOrders);
    this.#engramDepth = (geometry.engramConvolutionTaps - 1) * maximumOrder + 1;
    this.#engramRing = new Float32Array(
      geometry.engramLayers.length * this.#engramDepth * geometry.modelDimension,
    );
    this.#engramValid = new Uint8Array(geometry.engramLayers.length * this.#engramDepth);
    this.#engramPosition = 0;

    const confidence = this.weights.heads.get("confidence");
    this.#usesResidentConfidence = Boolean(
      this.options.collectConfidence !== false &&
        this.#fusedAttention?.residentLayersEnabled?.() &&
        this.#fusedAttention.residentConfidence,
    );
    this.#confidencePool =
      this.options.collectConfidence !== false && confidence && !this.#usesResidentConfidence
        ? new OnlineProbePool(confidence, geometry.modelDimension)
        : undefined;
  }

  async prefill(tokens: readonly number[], signal?: AbortSignal): Promise<Float32Array> {
    invariant(tokens.length > 0, "INVALID_CACT", "Cannot prefill an empty token sequence");
    invariant(
      this.#maximumLength >= tokens.length,
      "CONTEXT_OVERFLOW",
      `Prompt has ${tokens.length} tokens but runtime holds ${this.#maximumLength}`,
    );
    let logits: Float32Array | null = null;
    for (let index = 0; index < tokens.length; index++) {
      logits = await this.step(tokens[index] ?? 0, {
        wantLogits: index === tokens.length - 1,
        signal,
      });
    }
    invariant(logits !== null, "INVALID_CACT", "Prefill did not produce final logits");
    return logits;
  }

  async prefillSelected(
    tokens: readonly number[],
    allowedTokenIds?: Uint32Array,
    signal?: AbortSignal,
  ): Promise<ResidentTokenSelection> {
    invariant(tokens.length > 0, "INVALID_CACT", "Cannot prefill an empty token sequence");
    for (let index = 0; index + 1 < tokens.length; index++) {
      await this.step(tokens[index] ?? 0, { wantLogits: false, signal });
    }
    return this.stepSelected(tokens[tokens.length - 1] ?? 0, allowedTokenIds, signal);
  }

  async stepSelected(
    token: number,
    allowedTokenIds?: Uint32Array,
    signal?: AbortSignal,
  ): Promise<ResidentTokenSelection> {
    invariant(
      this.#selectionRequest === undefined,
      "BACKEND_UNAVAILABLE",
      "Selection is already active",
    );
    this.#selectionRequest = allowedTokenIds ?? null;
    this.#selectionResult = undefined;
    try {
      await this.step(token, { signal });
      const result = this.#selectionResult;
      invariant(result, "BACKEND_UNAVAILABLE", "Token selection did not produce a result");
      return result;
    } finally {
      this.#selectionRequest = undefined;
      this.#selectionResult = undefined;
    }
  }

  async step(
    token: number,
    options: { readonly wantLogits?: boolean; readonly signal?: AbortSignal | undefined } = {},
  ): Promise<Float32Array | null> {
    if (options.signal?.aborted)
      throw new NeedleError("GENERATION_ABORTED", "Needle generation was aborted", {
        cause: options.signal.reason,
      });
    const geometry = this.weights.geometry;
    invariant(this.#maximumLength > 0, "INVALID_CACT", "Call runtime.reset() before decoding");
    invariant(
      this.#position < this.#maximumLength,
      "CONTEXT_OVERFLOW",
      `Position ${this.#position} reached runtime limit ${this.#maximumLength}`,
    );
    invariant(
      Number.isInteger(token) && token >= 0 && token < geometry.vocabularySize,
      "INVALID_CACT",
      `Token ${token} is outside the vocabulary`,
    );

    const position = this.#position;
    this.#history.push(token);
    const dimension = geometry.modelDimension;
    const lanes = geometry.mhcLanes;
    const embedding = this.backend.row(this.weights.embedding, token);
    const scaledEmbedding = new Float32Array(dimension);
    const embeddingScale = Math.sqrt(dimension);
    for (let index = 0; index < dimension; index++)
      scaledEmbedding[index] = (embedding[index] ?? 0) * embeddingScale;
    this.#confidencePool?.add(scaledEmbedding);

    let x: Float32Array = new Float32Array(lanes * dimension);
    for (let lane = 0; lane < lanes; lane++) x.set(scaledEmbedding, lane * dimension);
    const [engramKeys, engramValues] = await this.#engramStep();
    const attentionWidth = geometry.numberOfHeads * geometry.headDimension;
    const residentLayers = (await this.#fusedAttention?.beginResidentToken?.(x)) ?? false;
    let residentLogits: Float32Array | null | undefined;
    const usingResidentLayers = Boolean(
      residentLayers &&
        this.#fusedAttention?.forwardResidentLayer &&
        this.#fusedAttention.finishResidentToken,
    );

    if (
      usingResidentLayers &&
      this.#fusedAttention?.forwardResidentLayer &&
      this.#fusedAttention.finishResidentToken
    ) {
      for (let layerIndex = 0; layerIndex < geometry.numberOfLayers; layerIndex++) {
        if (options.signal?.aborted)
          throw new NeedleError("GENERATION_ABORTED", "Needle generation was aborted", {
            cause: options.signal.reason,
          });
        this.options.onLayer?.({ position, layer: layerIndex, layers: geometry.numberOfLayers });
        const engramSite = geometry.engramLayers.indexOf(layerIndex);
        await this.#fusedAttention.forwardResidentLayer({
          layerIndex,
          position,
          ...(engramSite >= 0
            ? { engramKey: engramKeys[engramSite], engramValue: engramValues[engramSite] }
            : {}),
        });
      }
      if (
        this.#selectionRequest !== undefined &&
        this.#fusedAttention.finishResidentTokenForSelection &&
        this.#fusedAttention.selectResidentToken
      ) {
        await this.#fusedAttention.finishResidentTokenForSelection();
        residentLogits = null;
      } else {
        residentLogits = await this.#fusedAttention.finishResidentToken(
          options.wantLogits !== false,
        );
      }
    } else {
      for (let layerIndex = 0; layerIndex < geometry.numberOfLayers; layerIndex++) {
        if (options.signal?.aborted)
          throw new NeedleError("GENERATION_ABORTED", "Needle generation was aborted", {
            cause: options.signal.reason,
          });
        this.options.onLayer?.({ position, layer: layerIndex, layers: geometry.numberOfLayers });
        const layer = this.weights.layers[layerIndex];
        invariant(layer !== undefined, "INVALID_CACT", `Missing layer ${layerIndex}`);

        const normalizedLanes = rmsUnit(x);
        const [phiPre, phiPost, phiResidual] = await this.#matvecBatch([
          {
            matrix: this.weights.mhcPhiPre,
            input: normalizedLanes,
            range: { rowStart: layerIndex * lanes, rowCount: lanes },
          },
          {
            matrix: this.weights.mhcPhiPost,
            input: normalizedLanes,
            range: { rowStart: layerIndex * lanes, rowCount: lanes },
          },
          {
            matrix: this.weights.mhcPhiResidual,
            input: normalizedLanes,
            range: { rowStart: layerIndex * lanes * lanes, rowCount: lanes * lanes },
          },
        ] as const);

        const hPre = new Float32Array(lanes);
        const ownLane = layerIndex % lanes;
        for (let lane = 0; lane < lanes; lane++) {
          const offset = lane === ownLane ? 4 : -4;
          hPre[lane] = sigmoid(
            (this.weights.mhcAPre[layerIndex] ?? 0) * (phiPre[lane] ?? 0) +
              (this.weights.mhcBPre[layerIndex * lanes + lane] ?? 0) +
              offset,
          );
        }
        const updateInput = new Float32Array(dimension);
        for (let column = 0; column < dimension; column++) {
          let sum = 0;
          for (let lane = 0; lane < lanes; lane++)
            sum += (hPre[lane] ?? 0) * (x[lane * dimension + column] ?? 0);
          updateInput[column] = sum;
        }

        let blockInput = updateInput;
        const engramSite = geometry.engramLayers.indexOf(layerIndex);
        if (engramSite >= 0) {
          const key = engramKeys[engramSite];
          const value = engramValues[engramSite];
          if (key && value) {
            const normalizedInput = rmsUnit(updateInput);
            const normalizedKey = rmsUnit(key);
            let dot = 0;
            for (let index = 0; index < dimension; index++)
              dot += (normalizedInput[index] ?? 0) * (normalizedKey[index] ?? 0);
            const alpha = sigmoid(dot / Math.sqrt(dimension));
            blockInput = new Float32Array(dimension);
            for (let index = 0; index < dimension; index++)
              blockInput[index] = (updateInput[index] ?? 0) + alpha * (value[index] ?? 0);
          }
        }

        // Sandwich-normalized GQA attention.
        const attentionInput = rmsNorm(blockInput, layer.normInput);
        let delta: Float32Array | undefined;
        let fusedNextX: Float32Array | undefined;
        if (this.#fusedAttentionEnabled && this.#fusedAttention) {
          const fused = await this.#fusedAttention.forward({
            layer,
            layerIndex,
            position,
            input: attentionInput,
            blockInput,
            updateInput,
            x,
            phiPost,
            phiResidual,
          });
          if (fused.kind === "nextX") {
            fusedNextX = fused.values;
          } else {
            delta =
              fused.kind === "delta"
                ? fused.values
                : this.#attentionDelta(fused.values, blockInput, updateInput, layer);
          }
        } else {
          const [query, key, value, gate] = await this.#matvecBatch([
            { matrix: layer.queryProjection, input: attentionInput },
            { matrix: layer.keyProjection, input: attentionInput },
            { matrix: layer.valueProjection, input: attentionInput },
            { matrix: layer.gateProjection, input: attentionInput },
          ] as const);
          for (let head = 0; head < geometry.numberOfHeads; head++) {
            this.#rmsNormSlice(
              query,
              head * geometry.headDimension,
              geometry.headDimension,
              layer.queryNorm,
            );
          }
          for (let head = 0; head < geometry.numberOfKVHeads; head++) {
            this.#rmsNormSlice(
              key,
              head * geometry.headDimension,
              geometry.headDimension,
              layer.keyNorm,
            );
          }
          applyRope(
            query,
            geometry.numberOfHeads,
            geometry.headDimension,
            position,
            geometry.ropeTheta,
          );
          applyRope(
            key,
            geometry.numberOfKVHeads,
            geometry.headDimension,
            position,
            geometry.ropeTheta,
          );
          this.#storeKV(layerIndex, position, key, value);
          const attentionOutput = this.#attention(layerIndex, position, query);
          for (let index = 0; index < attentionWidth; index++)
            attentionOutput[index] = (attentionOutput[index] ?? 0) * sigmoid(gate[index] ?? 0);
          const projected = await this.backend.matvec(layer.outputProjection, attentionOutput);
          delta = this.#attentionDelta(projected, blockInput, updateInput, layer);
        }

        let nextX = fusedNextX;
        if (!nextX) {
          invariant(delta, "BACKEND_UNAVAILABLE", "Layer delta is missing");
          const hPost = new Float32Array(lanes);
          for (let lane = 0; lane < lanes; lane++) {
            const offset = lane === ownLane ? 0 : -4;
            hPost[lane] =
              2 *
              sigmoid(
                (this.weights.mhcAPost[layerIndex] ?? 0) * (phiPost[lane] ?? 0) +
                  (this.weights.mhcBPost[layerIndex * lanes + lane] ?? 0) +
                  offset,
              );
          }
          const routing = new Float32Array(lanes * lanes);
          for (let index = 0; index < routing.length; index++) {
            routing[index] =
              (this.weights.mhcAResidual[layerIndex] ?? 0) * (phiResidual[index] ?? 0) +
              (this.weights.mhcBResidual[layerIndex * lanes * lanes + index] ?? 0);
          }
          sinkhorn(routing, lanes);
          nextX = new Float32Array(x.length);
          for (let lane = 0; lane < lanes; lane++) {
            for (let column = 0; column < dimension; column++) {
              let sum = 0;
              for (let sourceLane = 0; sourceLane < lanes; sourceLane++) {
                sum +=
                  (routing[lane * lanes + sourceLane] ?? 0) *
                  (x[sourceLane * dimension + column] ?? 0);
              }
              nextX[lane * dimension + column] = sum + (hPost[lane] ?? 0) * (delta[column] ?? 0);
            }
          }
        }
        x = nextX;
        const hidden = new Float32Array(dimension);
        for (let column = 0; column < dimension; column++) {
          let sum = 0;
          for (let lane = 0; lane < lanes; lane++) sum += x[lane * dimension + column] ?? 0;
          hidden[column] = sum / lanes;
        }
        this.#confidencePool?.add(hidden);
      }
    }

    this.#position++;
    if (
      usingResidentLayers &&
      this.#selectionRequest !== undefined &&
      this.#fusedAttention?.selectResidentToken
    ) {
      this.#selectionResult = await this.#fusedAttention.selectResidentToken(
        this.#selectionRequest ?? undefined,
      );
      return null;
    }
    if (options.wantLogits === false) return null;
    if (usingResidentLayers) {
      invariant(residentLogits, "BACKEND_UNAVAILABLE", "Resident logits are missing");
      return residentLogits;
    }
    const mean = new Float32Array(dimension);
    for (let column = 0; column < dimension; column++) {
      let sum = 0;
      for (let lane = 0; lane < lanes; lane++) sum += x[lane * dimension + column] ?? 0;
      mean[column] = sum / lanes;
    }
    const final = rmsNorm(mean, this.weights.finalNorm);
    const logits = await this.backend.matvec(this.weights.embedding, final);
    if (this.#selectionRequest !== undefined) {
      this.#selectionResult = selectToken(logits, this.#selectionRequest);
      return null;
    }
    return logits;
  }

  #attentionDelta(
    projected: Float32Array,
    blockInput: Float32Array,
    updateInput: Float32Array,
    layer: CactLayer,
  ): Float32Array {
    const dimension = this.weights.geometry.modelDimension;
    const normalizedProjected = rmsNorm(projected, layer.postAttentionNorm);
    const attentionScale = sigmoid(layer.attentionGate);
    const afterAttention = new Float32Array(dimension);
    for (let index = 0; index < dimension; index++)
      afterAttention[index] =
        (blockInput[index] ?? 0) + attentionScale * (normalizedProjected[index] ?? 0);
    const preHadamard = rmsNorm(afterAttention, layer.preHadamardNorm);
    const mlp = hadamardMlp(
      preHadamard,
      layer.hadamardD1,
      layer.hadamardD2,
      layer.hadamardD3,
      this.weights.geometry.hadamardDimension,
    );
    const delta = new Float32Array(dimension);
    for (let index = 0; index < dimension; index++) {
      delta[index] = (afterAttention[index] ?? 0) + (mlp[index] ?? 0) - (updateInput[index] ?? 0);
    }
    return delta;
  }

  async #matvecBatch<const Requests extends readonly MatvecRequest[]>(
    requests: Requests,
  ): Promise<{ [Index in keyof Requests]: Float32Array }> {
    let outputs: readonly Float32Array[];
    if (this.backend.matvecBatch) {
      outputs = await this.backend.matvecBatch(requests);
    } else {
      const sequential: Float32Array[] = [];
      for (const request of requests) {
        sequential.push(await this.backend.matvec(request.matrix, request.input, request.range));
      }
      outputs = sequential;
    }
    invariant(
      outputs.length === requests.length,
      "BACKEND_UNAVAILABLE",
      `Backend returned ${outputs.length} results for ${requests.length} matvecs`,
    );
    return outputs as { [Index in keyof Requests]: Float32Array };
  }

  async resolveConfidence(): Promise<number | undefined> {
    if (this.#usesResidentConfidence && this.#fusedAttention?.residentConfidence) {
      return this.#fusedAttention.residentConfidence();
    }
    return this.confidence();
  }

  confidence(): number | undefined {
    const output = this.#confidencePool?.output();
    if (!output || output.length === 0) return undefined;
    return sigmoid(output[0] ?? 0);
  }

  #rmsNormSlice(values: Float32Array, offset: number, length: number, scale: Float32Array): void {
    let sumSquares = 0;
    for (let index = 0; index < length; index++) {
      const value = values[offset + index] ?? 0;
      sumSquares += value * value;
    }
    const inverse = 1 / Math.sqrt(sumSquares / length + 1e-6);
    for (let index = 0; index < length; index++) {
      values[offset + index] = (1 + (scale[index] ?? 0)) * (values[offset + index] ?? 0) * inverse;
    }
  }

  #kvSlot(position: number): number {
    const window = this.weights.geometry.kvWindow;
    if (window === 0) return position;
    if (position < this.#sinkLength) return position;
    return this.#sinkLength + ((position - this.#sinkLength) % window);
  }

  #cacheBase(layer: number, head: number, slot: number): number {
    const geometry = this.weights.geometry;
    return (
      ((layer * geometry.numberOfKVHeads + head) * this.#kvAllocation + slot) *
      geometry.headDimension
    );
  }

  #scaleIndex(layer: number, head: number, slot: number): number {
    const geometry = this.weights.geometry;
    return (layer * geometry.numberOfKVHeads + head) * this.#kvAllocation + slot;
  }

  #storeKV(layer: number, position: number, key: Float32Array, value: Float32Array): void {
    const geometry = this.weights.geometry;
    const slot = this.#kvSlot(position);
    for (let head = 0; head < geometry.numberOfKVHeads; head++) {
      const vectorOffset = head * geometry.headDimension;
      const cacheOffset = this.#cacheBase(layer, head, slot);
      const scaleOffset = this.#scaleIndex(layer, head, slot);
      if (this.#keyCacheFloat && this.#valueCacheFloat) {
        for (let index = 0; index < geometry.headDimension; index++) {
          this.#keyCacheFloat[cacheOffset + index] = key[vectorOffset + index] ?? 0;
          this.#valueCacheFloat[cacheOffset + index] = value[vectorOffset + index] ?? 0;
        }
        this.#keyScale[scaleOffset] = 1;
        this.#valueScale[scaleOffset] = 1;
      } else {
        const keyCache = this.#keyCacheInt8;
        const valueCache = this.#valueCacheInt8;
        invariant(keyCache && valueCache, "INVALID_CACT", "KV cache was not allocated");
        let keyMaximum = 1e-12;
        let valueMaximum = 1e-12;
        for (let index = 0; index < geometry.headDimension; index++) {
          keyMaximum = Math.max(keyMaximum, Math.abs(key[vectorOffset + index] ?? 0));
          valueMaximum = Math.max(valueMaximum, Math.abs(value[vectorOffset + index] ?? 0));
        }
        const keyScale = keyMaximum / 127;
        const valueScale = valueMaximum / 127;
        this.#keyScale[scaleOffset] = keyScale;
        this.#valueScale[scaleOffset] = valueScale;
        for (let index = 0; index < geometry.headDimension; index++) {
          keyCache[cacheOffset + index] = roundTiesToEven(
            (key[vectorOffset + index] ?? 0) / keyScale,
          );
          valueCache[cacheOffset + index] = roundTiesToEven(
            (value[vectorOffset + index] ?? 0) / valueScale,
          );
        }
      }
    }
  }

  #attention(layer: number, position: number, query: Float32Array): Float32Array {
    const geometry = this.weights.geometry;
    const repetitions = geometry.numberOfHeads / geometry.numberOfKVHeads;
    const output = new Float32Array(geometry.numberOfHeads * geometry.headDimension);
    const prefixCount = Math.min(this.#sinkLength, position + 1);
    const recentLow =
      geometry.kvWindow > 0 ? Math.max(prefixCount, position + 1 - geometry.kvWindow) : prefixCount;
    const logicalPositions: number[] = [];
    for (let index = 0; index < prefixCount; index++) logicalPositions.push(index);
    for (let index = recentLow; index <= position; index++) logicalPositions.push(index);
    const scores = new Float32Array(logicalPositions.length);
    const inverseRoot = 1 / Math.sqrt(geometry.headDimension);

    for (let kvHead = 0; kvHead < geometry.numberOfKVHeads; kvHead++) {
      for (let repetition = 0; repetition < repetitions; repetition++) {
        const queryHead = kvHead * repetitions + repetition;
        const queryOffset = queryHead * geometry.headDimension;
        for (let time = 0; time < logicalPositions.length; time++) {
          const logical = logicalPositions[time] ?? 0;
          const slot = this.#kvSlot(logical);
          const cacheOffset = this.#cacheBase(layer, kvHead, slot);
          const scaleOffset = this.#scaleIndex(layer, kvHead, slot);
          const keyScale = this.#keyScale[scaleOffset] ?? 1;
          let dot = 0;
          if (this.#keyCacheFloat) {
            for (let index = 0; index < geometry.headDimension; index++) {
              dot +=
                (query[queryOffset + index] ?? 0) * (this.#keyCacheFloat[cacheOffset + index] ?? 0);
            }
          } else {
            const cache = this.#keyCacheInt8;
            invariant(cache, "INVALID_CACT", "Int8 key cache is missing");
            for (let index = 0; index < geometry.headDimension; index++) {
              dot +=
                (query[queryOffset + index] ?? 0) * (cache[cacheOffset + index] ?? 0) * keyScale;
            }
          }
          scores[time] = dot * inverseRoot;
        }
        softmaxInPlace(scores, logicalPositions.length);
        const outputOffset = queryHead * geometry.headDimension;
        for (let time = 0; time < logicalPositions.length; time++) {
          const logical = logicalPositions[time] ?? 0;
          const slot = this.#kvSlot(logical);
          const cacheOffset = this.#cacheBase(layer, kvHead, slot);
          const scaleOffset = this.#scaleIndex(layer, kvHead, slot);
          const weight = scores[time] ?? 0;
          if (this.#valueCacheFloat) {
            for (let index = 0; index < geometry.headDimension; index++) {
              output[outputOffset + index] =
                (output[outputOffset + index] ?? 0) +
                weight * (this.#valueCacheFloat[cacheOffset + index] ?? 0);
            }
          } else {
            const cache = this.#valueCacheInt8;
            invariant(cache, "INVALID_CACT", "Int8 value cache is missing");
            const valueScale = this.#valueScale[scaleOffset] ?? 1;
            for (let index = 0; index < geometry.headDimension; index++) {
              output[outputOffset + index] =
                (output[outputOffset + index] ?? 0) +
                weight * (cache[cacheOffset + index] ?? 0) * valueScale;
            }
          }
        }
      }
    }
    return output;
  }

  async #engramStep(): Promise<[Float32Array[], Float32Array[]]> {
    const geometry = this.weights.geometry;
    if (geometry.engramLayers.length === 0) return [[], []];
    const headsPerOrder = geometry.numberOfEngramTables / geometry.engramOrders.length;
    invariant(
      Number.isInteger(headsPerOrder),
      "INVALID_CACT",
      "Engram table count is not divisible by order count",
    );
    const indices: number[] = [];
    const valid: boolean[] = [];
    for (let orderIndex = 0; orderIndex < geometry.engramOrders.length; orderIndex++) {
      const order = geometry.engramOrders[orderIndex] ?? 0;
      for (let head = 0; head < headsPerOrder; head++) {
        let hash = Math.imul(ENGRAM_SEED, orderIndex * headsPerOrder + head + 1) >>> 0;
        let isValid = true;
        for (let offset = 0; offset < order; offset++) {
          const historyIndex = this.#history.length - 1 - offset;
          const token = historyIndex >= 0 ? (this.#history[historyIndex] ?? 0) : 0;
          hash = Math.imul((hash ^ token) >>> 0, ENGRAM_PRIME) >>> 0;
          if (offset === order - 1) isValid = historyIndex >= 0;
        }
        hash = (hash ^ (hash >>> 15)) >>> 0;
        indices.push(hash % geometry.engramSlots);
        valid.push(isValid);
      }
    }

    const projectionRequests: MatvecRequest[] = [];
    for (let siteIndex = 0; siteIndex < this.weights.engrams.length; siteIndex++) {
      const site = this.weights.engrams[siteIndex];
      invariant(site, "INVALID_CACT", `Missing engram site ${siteIndex}`);
      const concatenated = new Float32Array(
        geometry.numberOfEngramTables * geometry.engramSubDimension,
      );
      for (let table = 0; table < geometry.numberOfEngramTables; table++) {
        if (valid[table]) {
          const row = table * geometry.engramSlots + (indices[table] ?? 0);
          concatenated.set(this.backend.row(site.tables, row), table * geometry.engramSubDimension);
        }
      }
      projectionRequests.push(
        { matrix: site.keyProjection, input: concatenated },
        { matrix: site.valueProjection, input: concatenated },
      );
    }

    const projected = await this.#matvecBatch(projectionRequests);
    const keys: Float32Array[] = [];
    const values: Float32Array[] = [];
    const maximumOrder = Math.max(...geometry.engramOrders);
    for (let siteIndex = 0; siteIndex < this.weights.engrams.length; siteIndex++) {
      const site = this.weights.engrams[siteIndex];
      const key = projected[siteIndex * 2];
      const valueNow = projected[siteIndex * 2 + 1];
      invariant(site && key && valueNow, "INVALID_CACT", `Missing engram site ${siteIndex}`);
      keys.push(key);

      const slot = this.#engramPosition % this.#engramDepth;
      const ringOffset = (siteIndex * this.#engramDepth + slot) * geometry.modelDimension;
      this.#engramRing.set(valueNow, ringOffset);
      this.#engramValid[siteIndex * this.#engramDepth + slot] = 1;
      const mixed = new Float32Array(geometry.modelDimension);
      for (let tap = 0; tap < geometry.engramConvolutionTaps; tap++) {
        const previousPosition = this.#engramPosition - tap * maximumOrder;
        if (previousPosition < 0) continue;
        const previousSlot = previousPosition % this.#engramDepth;
        if (!this.#engramValid[siteIndex * this.#engramDepth + previousSlot]) continue;
        const previousOffset =
          (siteIndex * this.#engramDepth + previousSlot) * geometry.modelDimension;
        const tapOffset = tap * geometry.modelDimension;
        for (let index = 0; index < geometry.modelDimension; index++) {
          mixed[index] =
            (mixed[index] ?? 0) +
            (site.taps[tapOffset + index] ?? 0) * (this.#engramRing[previousOffset + index] ?? 0);
        }
      }
      values.push(mixed);
    }
    this.#engramPosition++;
    return [keys, values];
  }
}

function selectToken(
  logits: Float32Array,
  allowedTokenIds: Uint32Array | null,
): ResidentTokenSelection {
  if (!allowedTokenIds) {
    const id = argmax(logits);
    return { id, logProbability: logSoftmaxAt(logits, id) };
  }
  invariant(allowedTokenIds.length > 0, "INVALID_CACT", "Allowed token set is empty");
  let id = allowedTokenIds[0] ?? 0;
  invariant(id < logits.length, "INVALID_CACT", `Token ${id} is outside logits`);
  let best = logits[id] ?? Number.NEGATIVE_INFINITY;
  for (let index = 1; index < allowedTokenIds.length; index++) {
    const candidate = allowedTokenIds[index] ?? 0;
    invariant(candidate < logits.length, "INVALID_CACT", `Token ${candidate} is outside logits`);
    const value = logits[candidate] ?? Number.NEGATIVE_INFINITY;
    if (value > best) {
      best = value;
      id = candidate;
    }
  }
  return { id, logProbability: logSoftmaxAt(logits, id) };
}

function roundTiesToEven(value: number): number {
  const lower = Math.floor(value);
  const fraction = value - lower;
  if (fraction < 0.5) return lower;
  if (fraction > 0.5) return lower + 1;
  return (lower & 1) === 0 ? lower : lower + 1;
}
