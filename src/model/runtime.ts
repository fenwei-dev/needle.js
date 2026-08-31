import type { InferenceBackend } from "../backends/backend.js";
import { denseMatvec } from "../backends/cq.js";
import { NeedleError, invariant } from "../errors.js";
import type { CactProbeHead, CactWeights } from "./cact.js";
import {
  applyRope,
  hadamardMlp,
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
  readonly onLayer?: (event: { readonly position: number; readonly layer: number; readonly layers: number }) => void;
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
          this.#weighted[weightedOffset + index] = (this.#weighted[weightedOffset + index] ?? 0) + weight * (cell[index] ?? 0);
        }
      } else {
        const oldWeight = Number.isFinite(previousMaximum) ? Math.exp(previousMaximum - score) : 0;
        this.#denominator[probe] = (this.#denominator[probe] ?? 0) * oldWeight + 1;
        for (let index = 0; index < this.#dimension; index++) {
          this.#weighted[weightedOffset + index] = (this.#weighted[weightedOffset + index] ?? 0) * oldWeight + (cell[index] ?? 0);
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

  constructor(weights: CactWeights, backend: InferenceBackend, options: RuntimeOptions = {}) {
    this.weights = weights;
    this.backend = backend;
    this.options = options;
  }

  get position(): number {
    return this.#position;
  }

  get maximumLength(): number {
    return this.#maximumLength;
  }

  reset(options: RuntimeResetOptions): void {
    const geometry = this.weights.geometry;
    invariant(Number.isInteger(options.maximumLength) && options.maximumLength > 0 && options.maximumLength <= geometry.maximumSequenceLength, "CONTEXT_OVERFLOW", `Runtime length ${options.maximumLength} is outside model limit ${geometry.maximumSequenceLength}`);
    this.#maximumLength = options.maximumLength;
    this.#sinkLength = Math.min(Math.max(options.sinkLength ?? 0, 0), options.maximumLength);
    this.#position = 0;
    this.#history = [];

    const window = geometry.kvWindow;
    this.#kvAllocation = window > 0
      ? Math.min(options.maximumLength, this.#sinkLength + window)
      : options.maximumLength;
    const kvElements = geometry.numberOfLayers * geometry.numberOfKVHeads * this.#kvAllocation * geometry.headDimension;
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

    const maximumOrder = Math.max(1, ...geometry.engramOrders);
    this.#engramDepth = (geometry.engramConvolutionTaps - 1) * maximumOrder + 1;
    this.#engramRing = new Float32Array(geometry.engramLayers.length * this.#engramDepth * geometry.modelDimension);
    this.#engramValid = new Uint8Array(geometry.engramLayers.length * this.#engramDepth);
    this.#engramPosition = 0;

    const confidence = this.weights.heads.get("confidence");
    this.#confidencePool = this.options.collectConfidence !== false && confidence
      ? new OnlineProbePool(confidence, geometry.modelDimension)
      : undefined;
  }

  async prefill(tokens: readonly number[], signal?: AbortSignal): Promise<Float32Array> {
    invariant(tokens.length > 0, "INVALID_CACT", "Cannot prefill an empty token sequence");
    invariant(this.#maximumLength >= tokens.length, "CONTEXT_OVERFLOW", `Prompt has ${tokens.length} tokens but runtime holds ${this.#maximumLength}`);
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

  async step(
    token: number,
    options: { readonly wantLogits?: boolean; readonly signal?: AbortSignal | undefined } = {},
  ): Promise<Float32Array | null> {
    if (options.signal?.aborted) throw new NeedleError("GENERATION_ABORTED", "Needle generation was aborted", { cause: options.signal.reason });
    const geometry = this.weights.geometry;
    invariant(this.#maximumLength > 0, "INVALID_CACT", "Call runtime.reset() before decoding");
    invariant(this.#position < this.#maximumLength, "CONTEXT_OVERFLOW", `Position ${this.#position} reached runtime limit ${this.#maximumLength}`);
    invariant(Number.isInteger(token) && token >= 0 && token < geometry.vocabularySize, "INVALID_CACT", `Token ${token} is outside the vocabulary`);

    const position = this.#position;
    this.#history.push(token);
    const dimension = geometry.modelDimension;
    const lanes = geometry.mhcLanes;
    const embedding = this.backend.row(this.weights.embedding, token);
    const scaledEmbedding = new Float32Array(dimension);
    const embeddingScale = Math.sqrt(dimension);
    for (let index = 0; index < dimension; index++) scaledEmbedding[index] = (embedding[index] ?? 0) * embeddingScale;
    this.#confidencePool?.add(scaledEmbedding);

    let x = new Float32Array(lanes * dimension);
    for (let lane = 0; lane < lanes; lane++) x.set(scaledEmbedding, lane * dimension);
    const [engramKeys, engramValues] = await this.#engramStep();
    const attentionWidth = geometry.numberOfHeads * geometry.headDimension;

    for (let layerIndex = 0; layerIndex < geometry.numberOfLayers; layerIndex++) {
      if (options.signal?.aborted) throw new NeedleError("GENERATION_ABORTED", "Needle generation was aborted", { cause: options.signal.reason });
      this.options.onLayer?.({ position, layer: layerIndex, layers: geometry.numberOfLayers });
      const layer = this.weights.layers[layerIndex];
      invariant(layer !== undefined, "INVALID_CACT", `Missing layer ${layerIndex}`);

      const normalizedLanes = rmsUnit(x);
      const phiPre = await this.backend.matvec(this.weights.mhcPhiPre, normalizedLanes, {
        rowStart: layerIndex * lanes,
        rowCount: lanes,
      });
      const phiPost = await this.backend.matvec(this.weights.mhcPhiPost, normalizedLanes, {
        rowStart: layerIndex * lanes,
        rowCount: lanes,
      });
      const phiResidual = await this.backend.matvec(this.weights.mhcPhiResidual, normalizedLanes, {
        rowStart: layerIndex * lanes * lanes,
        rowCount: lanes * lanes,
      });

      const hPre = new Float32Array(lanes);
      const ownLane = layerIndex % lanes;
      for (let lane = 0; lane < lanes; lane++) {
        const offset = lane === ownLane ? 4 : -4;
        hPre[lane] = sigmoid(
          (this.weights.mhcAPre[layerIndex] ?? 0) * (phiPre[lane] ?? 0)
            + (this.weights.mhcBPre[layerIndex * lanes + lane] ?? 0)
            + offset,
        );
      }
      const updateInput = new Float32Array(dimension);
      for (let column = 0; column < dimension; column++) {
        let sum = 0;
        for (let lane = 0; lane < lanes; lane++) sum += (hPre[lane] ?? 0) * (x[lane * dimension + column] ?? 0);
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
          for (let index = 0; index < dimension; index++) dot += (normalizedInput[index] ?? 0) * (normalizedKey[index] ?? 0);
          const alpha = sigmoid(dot / Math.sqrt(dimension));
          blockInput = new Float32Array(dimension);
          for (let index = 0; index < dimension; index++) blockInput[index] = (updateInput[index] ?? 0) + alpha * (value[index] ?? 0);
        }
      }

      // Sandwich-normalized GQA attention.
      const attentionInput = rmsNorm(blockInput, layer.normInput);
      const query = await this.backend.matvec(layer.queryProjection, attentionInput);
      const key = await this.backend.matvec(layer.keyProjection, attentionInput);
      const value = await this.backend.matvec(layer.valueProjection, attentionInput);
      for (let head = 0; head < geometry.numberOfHeads; head++) {
        this.#rmsNormSlice(query, head * geometry.headDimension, geometry.headDimension, layer.queryNorm);
      }
      for (let head = 0; head < geometry.numberOfKVHeads; head++) {
        this.#rmsNormSlice(key, head * geometry.headDimension, geometry.headDimension, layer.keyNorm);
      }
      applyRope(query, geometry.numberOfHeads, geometry.headDimension, position, geometry.ropeTheta);
      applyRope(key, geometry.numberOfKVHeads, geometry.headDimension, position, geometry.ropeTheta);
      this.#storeKV(layerIndex, position, key, value);
      const attentionOutput = this.#attention(layerIndex, position, query);
      const gate = await this.backend.matvec(layer.gateProjection, attentionInput);
      for (let index = 0; index < attentionWidth; index++) attentionOutput[index] = (attentionOutput[index] ?? 0) * sigmoid(gate[index] ?? 0);
      const projected = await this.backend.matvec(layer.outputProjection, attentionOutput);
      const normalizedProjected = rmsNorm(projected, layer.postAttentionNorm);
      const attentionScale = sigmoid(layer.attentionGate);
      const afterAttention = new Float32Array(dimension);
      for (let index = 0; index < dimension; index++) afterAttention[index] = (blockInput[index] ?? 0) + attentionScale * (normalizedProjected[index] ?? 0);

      const preHadamard = rmsNorm(afterAttention, layer.preHadamardNorm);
      const mlp = hadamardMlp(
        preHadamard,
        layer.hadamardD1,
        layer.hadamardD2,
        layer.hadamardD3,
        geometry.hadamardDimension,
      );
      const delta = new Float32Array(dimension);
      for (let index = 0; index < dimension; index++) {
        delta[index] = (afterAttention[index] ?? 0) + (mlp[index] ?? 0) - (updateInput[index] ?? 0);
      }

      const hPost = new Float32Array(lanes);
      for (let lane = 0; lane < lanes; lane++) {
        const offset = lane === ownLane ? 0 : -4;
        hPost[lane] = 2 * sigmoid(
          (this.weights.mhcAPost[layerIndex] ?? 0) * (phiPost[lane] ?? 0)
            + (this.weights.mhcBPost[layerIndex * lanes + lane] ?? 0)
            + offset,
        );
      }
      const routing = new Float32Array(lanes * lanes);
      for (let index = 0; index < routing.length; index++) {
        routing[index] = (this.weights.mhcAResidual[layerIndex] ?? 0) * (phiResidual[index] ?? 0)
          + (this.weights.mhcBResidual[layerIndex * lanes * lanes + index] ?? 0);
      }
      sinkhorn(routing, lanes);
      const nextX = new Float32Array(x.length);
      for (let lane = 0; lane < lanes; lane++) {
        for (let column = 0; column < dimension; column++) {
          let sum = 0;
          for (let sourceLane = 0; sourceLane < lanes; sourceLane++) {
            sum += (routing[lane * lanes + sourceLane] ?? 0) * (x[sourceLane * dimension + column] ?? 0);
          }
          nextX[lane * dimension + column] = sum + (hPost[lane] ?? 0) * (delta[column] ?? 0);
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

    this.#position++;
    if (options.wantLogits === false) return null;
    const mean = new Float32Array(dimension);
    for (let column = 0; column < dimension; column++) {
      let sum = 0;
      for (let lane = 0; lane < lanes; lane++) sum += x[lane * dimension + column] ?? 0;
      mean[column] = sum / lanes;
    }
    const final = rmsNorm(mean, this.weights.finalNorm);
    return this.backend.matvec(this.weights.embedding, final);
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
    return (((layer * geometry.numberOfKVHeads + head) * this.#kvAllocation + slot) * geometry.headDimension);
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
          keyCache[cacheOffset + index] = roundTiesToEven((key[vectorOffset + index] ?? 0) / keyScale);
          valueCache[cacheOffset + index] = roundTiesToEven((value[vectorOffset + index] ?? 0) / valueScale);
        }
      }
    }
  }

  #attention(layer: number, position: number, query: Float32Array): Float32Array {
    const geometry = this.weights.geometry;
    const repetitions = geometry.numberOfHeads / geometry.numberOfKVHeads;
    const output = new Float32Array(geometry.numberOfHeads * geometry.headDimension);
    const prefixCount = Math.min(this.#sinkLength, position + 1);
    const recentLow = geometry.kvWindow > 0
      ? Math.max(prefixCount, position + 1 - geometry.kvWindow)
      : prefixCount;
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
              dot += (query[queryOffset + index] ?? 0) * (this.#keyCacheFloat[cacheOffset + index] ?? 0);
            }
          } else {
            const cache = this.#keyCacheInt8;
            invariant(cache, "INVALID_CACT", "Int8 key cache is missing");
            for (let index = 0; index < geometry.headDimension; index++) {
              dot += (query[queryOffset + index] ?? 0) * (cache[cacheOffset + index] ?? 0) * keyScale;
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
              output[outputOffset + index] = (output[outputOffset + index] ?? 0)
                + weight * (this.#valueCacheFloat[cacheOffset + index] ?? 0);
            }
          } else {
            const cache = this.#valueCacheInt8;
            invariant(cache, "INVALID_CACT", "Int8 value cache is missing");
            const valueScale = this.#valueScale[scaleOffset] ?? 1;
            for (let index = 0; index < geometry.headDimension; index++) {
              output[outputOffset + index] = (output[outputOffset + index] ?? 0)
                + weight * (cache[cacheOffset + index] ?? 0) * valueScale;
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
    invariant(Number.isInteger(headsPerOrder), "INVALID_CACT", "Engram table count is not divisible by order count");
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

    const keys: Float32Array[] = [];
    const values: Float32Array[] = [];
    const maximumOrder = Math.max(...geometry.engramOrders);
    for (let siteIndex = 0; siteIndex < this.weights.engrams.length; siteIndex++) {
      const site = this.weights.engrams[siteIndex];
      invariant(site, "INVALID_CACT", `Missing engram site ${siteIndex}`);
      const concatenated = new Float32Array(geometry.numberOfEngramTables * geometry.engramSubDimension);
      for (let table = 0; table < geometry.numberOfEngramTables; table++) {
        if (valid[table]) {
          const row = table * geometry.engramSlots + (indices[table] ?? 0);
          concatenated.set(this.backend.row(site.tables, row), table * geometry.engramSubDimension);
        }
      }
      const key = await this.backend.matvec(site.keyProjection, concatenated);
      const valueNow = await this.backend.matvec(site.valueProjection, concatenated);
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
        const previousOffset = (siteIndex * this.#engramDepth + previousSlot) * geometry.modelDimension;
        const tapOffset = tap * geometry.modelDimension;
        for (let index = 0; index < geometry.modelDimension; index++) {
          mixed[index] = (mixed[index] ?? 0)
            + (site.taps[tapOffset + index] ?? 0) * (this.#engramRing[previousOffset + index] ?? 0);
        }
      }
      values.push(mixed);
    }
    this.#engramPosition++;
    return [keys, values];
  }
}

function roundTiesToEven(value: number): number {
  const lower = Math.floor(value);
  const fraction = value - lower;
  if (fraction < 0.5) return lower;
  if (fraction > 0.5) return lower + 1;
  return (lower & 1) === 0 ? lower : lower + 1;
}
