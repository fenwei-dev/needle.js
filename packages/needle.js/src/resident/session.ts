/// <reference types="@webgpu/types" preserve="true" />

import type {
  FusedAttentionRequest,
  FusedAttentionResetOptions,
  FusedAttentionResult,
  FusedAttentionSession,
  ResidentEngramRequest,
  ResidentLayerRequest,
  ResidentTokenRequest,
  ResidentTokenSelection,
} from "../backends/backend.js";
import { prepareCqActivation } from "../backends/cq.js";
import { matrixParameters, webGpuMatrixData } from "../backends/webgpu-kernel.js";
import { invariant } from "../errors.js";
import type { CactLayer, CactWeights, CqMatrix } from "../model/cact.js";
import { supportsResidentExecution } from "./compatibility.js";
import { ResidentConfidence } from "./confidence.js";
import {
  type AttentionParameters,
  createRawParameterFactory,
  type KvParameters,
  type QueryParameters,
  type ResidentParameter,
  type ResidentParameterFactory,
} from "./parameters.js";
import { createResidentPipelines, type ResidentPipelines as Pipelines } from "./pipelines.js";
import { ResidentTokenSelector } from "./selection.js";
import {
  bufferSource,
  bufferWithData,
  COPY_DST,
  COPY_SRC,
  floatBits,
  MAP_READ,
  STORAGE,
  storageBuffer,
} from "./webgpu.js";

interface GpuMatrix {
  readonly packed: GPUBuffer;
  readonly norms: GPUBuffer;
}

interface LayerNormBuffers {
  readonly query: GPUBuffer;
  readonly key: GPUBuffer;
  readonly input: GPUBuffer;
  readonly postAttention: GPUBuffer;
  readonly preHadamard: GPUBuffer;
  readonly d1: GPUBuffer;
  readonly d2: GPUBuffer;
  readonly d3: GPUBuffer;
}

interface ResidentEngramGroups {
  readonly gather: GPUBindGroup;
  readonly prepare: GPUBindGroup;
  readonly key: GPUBindGroup;
  readonly value: GPUBindGroup;
  readonly convolve: GPUBindGroup;
  readonly inject: GPUBindGroup;
}

interface ResidentLayerSlot {
  readonly layer: CactLayer;
  readonly projectionParams: readonly GPUBuffer[];
  readonly queryParams: ResidentParameter<QueryParameters>;
  readonly kvParams: ResidentParameter<KvParameters>;
  readonly attentionParams: ResidentParameter<AttentionParameters>;
  readonly projectionGroups: readonly GPUBindGroup[];
  readonly phiGroups: readonly GPUBindGroup[];
  readonly queryGroup: GPUBindGroup;
  readonly kvGroup: GPUBindGroup;
  readonly scoreGroup: GPUBindGroup;
  readonly attentionGroup: GPUBindGroup;
  readonly mhcPreGroup: GPUBindGroup;
  readonly inputNormGroup: GPUBindGroup;
  readonly sandwichGroup: GPUBindGroup;
  readonly mlpGroup: GPUBindGroup;
  readonly routingGroup: GPUBindGroup;
  readonly engramKey?: GPUBuffer;
  readonly engramValue?: GPUBuffer;
  readonly engramGroup?: GPUBindGroup;
}

export interface WebGpuResidentOptions {
  readonly fuseMlp: boolean;
  readonly fuseRouting: boolean;
  readonly residentLayers: boolean;
  readonly singleTokenSubmission: boolean;
  readonly parameterFactory?: ResidentParameterFactory;
}

export class WebGpuResidentSession implements FusedAttentionSession {
  readonly #device: GPUDevice;
  readonly #weights: CactWeights;
  readonly #fuseMlp: boolean;
  readonly #fuseRouting: boolean;
  readonly #residentLayers: boolean;
  readonly #singleTokenSubmission: boolean;
  readonly #parameterFactory: ResidentParameterFactory;
  readonly #codebook: GPUBuffer;
  readonly #preparedInput: GPUBuffer;
  readonly #normalizedLanes: GPUBuffer;
  readonly #preparedLanes: GPUBuffer;
  readonly #attentionInput: GPUBuffer;
  readonly #query: GPUBuffer;
  readonly #key: GPUBuffer;
  readonly #value: GPUBuffer;
  readonly #gate: GPUBuffer;
  readonly #attentionOutput: GPUBuffer;
  readonly #scores: GPUBuffer;
  readonly #preparedAttention: GPUBuffer;
  readonly #projected: GPUBuffer;
  readonly #blockInput: GPUBuffer;
  readonly #updateInput: GPUBuffer;
  readonly #afterAttention: GPUBuffer;
  readonly #mlpInput: GPUBuffer;
  readonly #delta: GPUBuffer;
  readonly #lanesInput: GPUBuffer;
  readonly #phiPre: GPUBuffer;
  readonly #phiPost: GPUBuffer;
  readonly #phiResidual: GPUBuffer;
  readonly #routingParams: GPUBuffer;
  readonly #hPreParams: GPUBuffer;
  readonly #engramKey: GPUBuffer;
  readonly #engramValue: GPUBuffer;
  readonly #engramRowIds: GPUBuffer;
  readonly #engramRowValid: GPUBuffer;
  readonly #residentEngramConcatenated: GPUBuffer[];
  readonly #residentEngramPrepared: GPUBuffer[];
  readonly #residentEngramKeys: GPUBuffer[];
  readonly #residentEngramValueNow: GPUBuffer[];
  readonly #residentEngramMixed: GPUBuffer[];
  readonly #residentEngramRing: GPUBuffer;
  readonly #residentEngramRingValid: GPUBuffer;
  readonly #residentEngramConvolutionParams: GPUBuffer[];
  readonly #residentEngramProjectionParams: GPUBuffer[];
  readonly #nextLanes: GPUBuffer;
  readonly #finalNorm: GPUBuffer;
  readonly #finalHidden: GPUBuffer;
  readonly #preparedFinal: GPUBuffer;
  readonly #logits: GPUBuffer;
  readonly #confidence: ResidentConfidence;
  readonly #selector: ResidentTokenSelector;
  readonly #staging: GPUBuffer;
  readonly #queryParams: ResidentParameter<QueryParameters>;
  readonly #kvParams: ResidentParameter<KvParameters>;
  readonly #attentionParams: ResidentParameter<AttentionParameters>;
  readonly #layerParams: GPUBuffer;
  readonly #projectionParams: GPUBuffer[];
  readonly #matrixCache = new WeakMap<CqMatrix, GpuMatrix>();
  readonly #allocatedMatrices: GpuMatrix[] = [];
  readonly #projectionGroups = new WeakMap<CqMatrix, GPUBindGroup>();
  readonly #layerNorms = new WeakMap<CactLayer, LayerNormBuffers>();
  readonly #allocatedNorms: LayerNormBuffers[] = [];
  readonly #pipelines: Promise<Pipelines>;

  #keyCache: GPUBuffer | undefined;
  #valueCache: GPUBuffer | undefined;
  #keyScales: GPUBuffer | undefined;
  #valueScales: GPUBuffer | undefined;
  #queryGroups = new WeakMap<CactLayer, GPUBindGroup>();
  #kvGroups = new WeakMap<CactLayer, GPUBindGroup>();
  #scoreGroup: GPUBindGroup | undefined;
  #attentionGroup: GPUBindGroup | undefined;
  #prepareGroup: GPUBindGroup | undefined;
  #sandwichGroups = new WeakMap<CactLayer, GPUBindGroup>();
  #mlpGroups = new WeakMap<CactLayer, GPUBindGroup>();
  #routingGroup: GPUBindGroup | undefined;
  #rmsLanesGroup: GPUBindGroup | undefined;
  #prepareLanesGroup: GPUBindGroup | undefined;
  #mhcPreGroup: GPUBindGroup | undefined;
  #engramGroup: GPUBindGroup | undefined;
  #residentEngramExtraBuffers: GPUBuffer[] = [];
  #residentEngramGroups: ResidentEngramGroups[] = [];
  #residentEngramReadyPosition = -1;
  #prepareInputGroup: GPUBindGroup | undefined;
  #finalNormGroup: GPUBindGroup | undefined;
  #prepareFinalGroup: GPUBindGroup | undefined;
  #inputNormGroups = new WeakMap<CactLayer, GPUBindGroup>();
  #residentSlots: ResidentLayerSlot[] = [];
  #residentSlotBuffers: GPUBuffer[] = [];
  #residentSlotParameters: Array<{ destroy(): void }> = [];
  #residentSlotAllocation = -1;
  #residentEnabled = false;
  #collectConfidence = false;
  #kvAllocation = 0;
  #sinkLength = 0;
  #enabled = false;
  #disposed = false;

  constructor(device: GPUDevice, weights: CactWeights, options: WebGpuResidentOptions) {
    this.#device = device;
    this.#weights = weights;
    this.#fuseMlp = options.fuseMlp;
    this.#fuseRouting = options.fuseRouting;
    this.#residentLayers = options.residentLayers;
    this.#singleTokenSubmission = options.singleTokenSubmission;
    const geometry = weights.geometry;
    const dimensionBytes = geometry.modelDimension * 4;
    const lanesBytes = geometry.mhcLanes * dimensionBytes;
    const keyValueBytes = geometry.numberOfKVHeads * geometry.headDimension * 4;
    this.#codebook = bufferWithData(device, "needle.attention.codebook", weights.codebook, STORAGE);
    this.#preparedInput = storageBuffer(device, "needle.attention.input", dimensionBytes, COPY_DST);
    this.#normalizedLanes = storageBuffer(device, "needle.attention.normalized-lanes", lanesBytes);
    this.#preparedLanes = storageBuffer(device, "needle.attention.prepared-lanes", lanesBytes);
    this.#attentionInput = storageBuffer(
      device,
      "needle.attention.normalized-input",
      dimensionBytes,
    );
    this.#query = storageBuffer(device, "needle.attention.query", dimensionBytes);
    this.#key = storageBuffer(device, "needle.attention.key", keyValueBytes);
    this.#value = storageBuffer(device, "needle.attention.value", keyValueBytes);
    this.#gate = storageBuffer(device, "needle.attention.gate", dimensionBytes);
    this.#attentionOutput = storageBuffer(device, "needle.attention.output", dimensionBytes);
    this.#scores = storageBuffer(
      device,
      "needle.attention.scores",
      geometry.numberOfHeads * 512 * 4,
    );
    this.#preparedAttention = storageBuffer(device, "needle.attention.prepared", dimensionBytes);
    this.#projected = storageBuffer(device, "needle.attention.projected", dimensionBytes, COPY_SRC);
    this.#blockInput = storageBuffer(
      device,
      "needle.attention.block-input",
      dimensionBytes,
      COPY_DST,
    );
    this.#updateInput = storageBuffer(
      device,
      "needle.attention.update-input",
      dimensionBytes,
      COPY_DST | COPY_SRC,
    );
    this.#afterAttention = storageBuffer(device, "needle.attention.after", dimensionBytes);
    this.#mlpInput = storageBuffer(device, "needle.attention.mlp-input", dimensionBytes);
    this.#delta = storageBuffer(device, "needle.attention.delta", dimensionBytes, COPY_SRC);
    this.#lanesInput = storageBuffer(
      device,
      "needle.attention.lanes",
      lanesBytes,
      COPY_DST | COPY_SRC,
    );
    this.#phiPre = storageBuffer(device, "needle.attention.phi-pre", 4 * 4);
    this.#phiPost = storageBuffer(device, "needle.attention.phi-post", 4 * 4, COPY_DST);
    this.#phiResidual = storageBuffer(device, "needle.attention.phi-residual", 16 * 4, COPY_DST);
    this.#routingParams = storageBuffer(
      device,
      "needle.attention.routing-params",
      23 * 4,
      COPY_DST,
    );
    this.#hPreParams = storageBuffer(device, "needle.attention.h-pre-params", 6 * 4, COPY_DST);
    this.#engramKey = storageBuffer(
      device,
      "needle.attention.engram-key",
      dimensionBytes,
      COPY_DST,
    );
    this.#engramValue = storageBuffer(
      device,
      "needle.attention.engram-value",
      dimensionBytes,
      COPY_DST,
    );
    this.#engramRowIds = storageBuffer(device, "needle.engram.row-ids", 4 * 4, COPY_DST);
    this.#engramRowValid = storageBuffer(device, "needle.engram.row-valid", 4 * 4, COPY_DST);
    this.#residentEngramConcatenated = [];
    this.#residentEngramPrepared = [];
    this.#residentEngramKeys = [];
    this.#residentEngramValueNow = [];
    this.#residentEngramMixed = [];
    this.#residentEngramConvolutionParams = [];
    this.#residentEngramProjectionParams = [];
    const maximumOrder = Math.max(1, ...geometry.engramOrders);
    const engramDepth = (geometry.engramConvolutionTaps - 1) * maximumOrder + 1;
    this.#residentEngramRing = storageBuffer(
      device,
      "needle.engram.ring",
      Math.max(4, geometry.engramLayers.length * engramDepth * dimensionBytes),
      COPY_DST,
    );
    this.#residentEngramRingValid = storageBuffer(
      device,
      "needle.engram.ring-valid",
      Math.max(4, geometry.engramLayers.length * engramDepth * 4),
      COPY_DST,
    );
    for (let site = 0; site < weights.engrams.length; site++) {
      const engram = weights.engrams[site];
      invariant(engram, "INVALID_CACT", `Missing engram site ${site}`);
      this.#residentEngramConcatenated.push(
        storageBuffer(device, `needle.engram.${site}.concatenated`, dimensionBytes),
      );
      this.#residentEngramPrepared.push(
        storageBuffer(device, `needle.engram.${site}.prepared`, dimensionBytes),
      );
      this.#residentEngramKeys.push(
        storageBuffer(device, `needle.engram.${site}.key`, dimensionBytes),
      );
      this.#residentEngramValueNow.push(
        storageBuffer(device, `needle.engram.${site}.value-now`, dimensionBytes),
      );
      this.#residentEngramMixed.push(
        storageBuffer(device, `needle.engram.${site}.mixed`, dimensionBytes),
      );
      this.#residentEngramConvolutionParams.push(
        storageBuffer(device, `needle.engram.${site}.convolution-params`, 16, COPY_DST),
      );
      const keyParams = bufferWithData(
        device,
        `needle.engram.${site}.key-params`,
        matrixParameters(engram.keyProjection, 0, engram.keyProjection.outputSize),
        STORAGE,
      );
      const valueParams = bufferWithData(
        device,
        `needle.engram.${site}.value-params`,
        matrixParameters(engram.valueProjection, 0, engram.valueProjection.outputSize),
        STORAGE,
      );
      this.#residentEngramProjectionParams.push(keyParams, valueParams);
    }
    this.#nextLanes = storageBuffer(device, "needle.attention.next-lanes", lanesBytes, COPY_SRC);
    this.#finalNorm = bufferWithData(
      device,
      "needle.attention.final-norm",
      weights.finalNorm,
      STORAGE,
    );
    this.#finalHidden = storageBuffer(device, "needle.attention.final-hidden", dimensionBytes);
    this.#preparedFinal = storageBuffer(device, "needle.attention.prepared-final", dimensionBytes);
    this.#logits = storageBuffer(
      device,
      "needle.attention.logits",
      geometry.vocabularySize * 4,
      COPY_SRC,
    );
    this.#confidence = new ResidentConfidence(device, weights);
    this.#selector = new ResidentTokenSelector(device, this.#logits, geometry.vocabularySize);
    this.#staging = device.createBuffer({
      label: "needle.attention.readback",
      size: Math.max(lanesBytes, geometry.vocabularySize * 4),
      usage: MAP_READ | COPY_DST,
    });
    const parameterFactory = options.parameterFactory ?? createRawParameterFactory(device);
    this.#parameterFactory = parameterFactory;
    this.#queryParams = parameterFactory.query("needle.attention.query-params");
    this.#kvParams = parameterFactory.kv("needle.attention.kv-params");
    this.#attentionParams = parameterFactory.attention("needle.attention.params");
    this.#layerParams = storageBuffer(device, "needle.attention.layer-params", 4, COPY_DST);
    this.#projectionParams = Array.from({ length: 9 }, (_, index) =>
      storageBuffer(device, `needle.attention.projection-params.${index}`, 32, COPY_DST),
    );
    this.#pipelines = createResidentPipelines(device);
  }

  reset(options: FusedAttentionResetOptions): boolean {
    invariant(!this.#disposed, "BACKEND_UNAVAILABLE", "Fused attention was disposed");
    const geometry = this.#weights.geometry;
    this.#enabled = supportsResidentExecution(this.#weights, options);
    const confidenceAvailable = this.#confidence.available;
    this.#residentEnabled =
      this.#enabled &&
      this.#residentLayers &&
      this.#fuseRouting &&
      (!options.collectConfidence || confidenceAvailable);
    this.#collectConfidence = this.#residentEnabled && options.collectConfidence;
    if (!this.#enabled) return false;
    if (this.#collectConfidence) this.#confidence.reset();
    if (this.#residentEnabled) {
      this.#device.queue.writeBuffer(
        this.#residentEngramRingValid,
        0,
        bufferSource(
          new Uint32Array(
            this.#weights.geometry.engramLayers.length *
              ((this.#weights.geometry.engramConvolutionTaps - 1) *
                Math.max(1, ...this.#weights.geometry.engramOrders) +
                1),
          ),
        ),
      );
      this.#residentEngramReadyPosition = -1;
    }

    this.#sinkLength = options.sinkLength;
    const allocation = Math.min(options.maximumLength, options.sinkLength + geometry.kvWindow);
    if (!this.#keyCache || allocation !== this.#kvAllocation) {
      this.#destroyCache();
      this.#kvAllocation = allocation;
      const cacheElements =
        geometry.numberOfLayers *
        geometry.numberOfKVHeads *
        this.#kvAllocation *
        geometry.headDimension;
      const scaleElements = geometry.numberOfLayers * geometry.numberOfKVHeads * this.#kvAllocation;
      this.#keyCache = storageBuffer(this.#device, "needle.attention.key-cache", cacheElements * 4);
      this.#valueCache = storageBuffer(
        this.#device,
        "needle.attention.value-cache",
        cacheElements * 4,
      );
      this.#keyScales = storageBuffer(
        this.#device,
        "needle.attention.key-scales",
        scaleElements * 4,
      );
      this.#valueScales = storageBuffer(
        this.#device,
        "needle.attention.value-scales",
        scaleElements * 4,
      );
      this.#queryGroups = new WeakMap();
      this.#kvGroups = new WeakMap();
      this.#scoreGroup = undefined;
      this.#attentionGroup = undefined;
    }
    return true;
  }

  async beginResidentToken(lanes: Float32Array): Promise<boolean> {
    if (!this.#residentEnabled || this.#disposed) return false;
    invariant(
      lanes.length === 2048,
      "INVALID_CACT",
      "Resident lane input must contain 2048 values",
    );
    this.#device.queue.writeBuffer(this.#lanesInput, 0, bufferSource(lanes));
    if (this.#collectConfidence) {
      const pipelines = await this.#pipelines;
      const encoder = this.#device.createCommandEncoder({ label: "needle.confidence.embedding" });
      this.#confidence.encodePool(
        encoder,
        pipelines.confidencePool,
        this.#lanesInput,
        "needle.confidence.pool-embedding",
      );
      this.#device.queue.submit([encoder.finish()]);
    }
    return true;
  }

  async prepareResidentEngrams(request: ResidentEngramRequest): Promise<boolean> {
    if (!this.#residentEnabled || this.#disposed || this.#weights.engrams.length === 0)
      return false;
    invariant(
      request.indices.length === 4 && request.valid.length === 4,
      "INVALID_CACT",
      "Resident engram lookup requires four row IDs and validity flags",
    );
    this.#device.queue.writeBuffer(this.#engramRowIds, 0, bufferSource(request.indices));
    this.#device.queue.writeBuffer(this.#engramRowValid, 0, bufferSource(request.valid));
    const pipelines = await this.#pipelines;
    const groups = this.#ensureResidentEngramGroups(pipelines);
    const geometry = this.#weights.geometry;
    const maximumOrder = Math.max(1, ...geometry.engramOrders);
    const depth = (geometry.engramConvolutionTaps - 1) * maximumOrder + 1;
    const encoder = this.#device.createCommandEncoder({ label: "needle.engram.resident" });
    for (let site = 0; site < groups.length; site++) {
      const group = groups[site];
      const params = this.#residentEngramConvolutionParams[site];
      invariant(group && params, "INVALID_CACT", `Missing resident engram group ${site}`);
      this.#device.queue.writeBuffer(
        params,
        0,
        bufferSource(new Uint32Array([site, request.position, depth, maximumOrder])),
      );
      const gather = encoder.beginComputePass({ label: `needle.engram.${site}.gather` });
      gather.setPipeline(pipelines.engramGather);
      gather.setBindGroup(0, group.gather);
      gather.dispatchWorkgroups(4);
      gather.end();
      const prepare = encoder.beginComputePass({ label: `needle.engram.${site}.prepare` });
      prepare.setPipeline(pipelines.prepare);
      prepare.setBindGroup(0, group.prepare);
      prepare.dispatchWorkgroups(4);
      prepare.end();
      const projections = encoder.beginComputePass({ label: `needle.engram.${site}.project` });
      projections.setPipeline(pipelines.cq);
      projections.setBindGroup(0, group.key);
      projections.dispatchWorkgroups(512);
      projections.setBindGroup(0, group.value);
      projections.dispatchWorkgroups(512);
      projections.end();
      const convolve = encoder.beginComputePass({ label: `needle.engram.${site}.convolve` });
      convolve.setPipeline(pipelines.engramConvolve);
      convolve.setBindGroup(0, group.convolve);
      convolve.dispatchWorkgroups(1);
      convolve.end();
    }
    this.#device.queue.submit([encoder.finish()]);
    this.#residentEngramReadyPosition = request.position;
    return true;
  }

  residentLayersEnabled(): boolean {
    return this.#residentEnabled;
  }

  async forwardResidentToken(request: ResidentTokenRequest): Promise<void> {
    invariant(
      this.#residentEnabled && !this.#disposed,
      "BACKEND_UNAVAILABLE",
      "Resident layers are off",
    );
    if (!this.#singleTokenSubmission) {
      for (let layerIndex = 0; layerIndex < this.#weights.geometry.numberOfLayers; layerIndex++) {
        const engramSite = this.#weights.geometry.engramLayers.indexOf(layerIndex);
        await this.forwardResidentLayer({
          layerIndex,
          position: request.position,
          ...(engramSite >= 0
            ? {
                engramKey: request.engramKeys[engramSite],
                engramValue: request.engramValues[engramSite],
              }
            : {}),
        });
      }
      return;
    }
    const pipelines = await this.#pipelines;
    const slots = this.#ensureResidentSlots(pipelines);
    const geometry = this.#weights.geometry;
    const thetaBits = floatBits(geometry.ropeTheta);
    const encoder = this.#device.createCommandEncoder({ label: "needle.resident-token.layers" });

    for (let layerIndex = 0; layerIndex < slots.length; layerIndex++) {
      const slot = slots[layerIndex];
      invariant(slot, "INVALID_CACT", `Missing resident slot ${layerIndex}`);
      slot.queryParams.write({ position: request.position, thetaBits });
      slot.kvParams.write({
        layer: layerIndex,
        position: request.position,
        allocation: this.#kvAllocation,
        sinkLength: this.#sinkLength,
        window: geometry.kvWindow,
        kvHeads: geometry.numberOfKVHeads,
        thetaBits,
        reserved: 0,
      });
      slot.attentionParams.write({
        layer: layerIndex,
        position: request.position,
        allocation: this.#kvAllocation,
        sinkLength: this.#sinkLength,
        window: geometry.kvWindow,
        kvHeads: geometry.numberOfKVHeads,
        heads: geometry.numberOfHeads,
        reserved: 0,
      });
      const engramSite = geometry.engramLayers.indexOf(layerIndex);
      if (engramSite >= 0 && slot.engramKey && slot.engramValue) {
        const key = request.engramKeys[engramSite];
        const value = request.engramValues[engramSite];
        invariant(key && value, "INVALID_CACT", `Missing engram values for layer ${layerIndex}`);
        this.#device.queue.writeBuffer(slot.engramKey, 0, bufferSource(key));
        this.#device.queue.writeBuffer(slot.engramValue, 0, bufferSource(value));
      }
      this.#encodeResidentSlot(encoder, slot, request.position, engramSite >= 0, pipelines);
    }
    this.#device.queue.submit([encoder.finish()]);
  }

  async forwardResidentLayer(request: ResidentLayerRequest): Promise<void> {
    invariant(
      this.#residentEnabled && !this.#disposed,
      "BACKEND_UNAVAILABLE",
      "Resident layers are off",
    );
    const { layerIndex, position, engramKey, engramValue } = request;
    const layer = this.#weights.layers[layerIndex];
    invariant(layer, "INVALID_CACT", `Missing resident layer ${layerIndex}`);
    const pipelines = await this.#pipelines;

    const matrices = [
      layer.queryProjection,
      layer.keyProjection,
      layer.valueProjection,
      layer.gateProjection,
      layer.outputProjection,
    ] as const;
    for (let index = 0; index < matrices.length; index++) {
      const matrix = matrices[index];
      const params = this.#projectionParams[index];
      invariant(matrix && params, "INVALID_CACT", "Resident projection is missing");
      this.#device.queue.writeBuffer(
        params,
        0,
        bufferSource(matrixParameters(matrix, 0, matrix.outputSize)),
      );
    }
    const phiMatrices = [
      this.#weights.mhcPhiPre,
      this.#weights.mhcPhiPost,
      this.#weights.mhcPhiResidual,
    ] as const;
    const phiRanges = [
      { rowStart: layerIndex * 4, rowCount: 4 },
      { rowStart: layerIndex * 4, rowCount: 4 },
      { rowStart: layerIndex * 16, rowCount: 16 },
    ] as const;
    for (let index = 0; index < phiMatrices.length; index++) {
      const matrix = phiMatrices[index];
      const range = phiRanges[index];
      const params = this.#projectionParams[index + 5];
      invariant(matrix && range && params, "INVALID_CACT", "Resident mHC projection is missing");
      this.#device.queue.writeBuffer(
        params,
        0,
        bufferSource(matrixParameters(matrix, range.rowStart, range.rowCount)),
      );
    }
    const hPreParams = new Float32Array(6);
    hPreParams[0] = this.#weights.mhcAPre[layerIndex] ?? 0;
    hPreParams[1] = layerIndex % 4;
    hPreParams.set(this.#weights.mhcBPre.subarray(layerIndex * 4, layerIndex * 4 + 4), 2);
    this.#device.queue.writeBuffer(this.#hPreParams, 0, bufferSource(hPreParams));
    const routingParams = new Float32Array(23);
    routingParams[0] = this.#weights.mhcAPost[layerIndex] ?? 0;
    routingParams[1] = this.#weights.mhcAResidual[layerIndex] ?? 0;
    routingParams[2] = layerIndex % 4;
    routingParams.set(this.#weights.mhcBPost.subarray(layerIndex * 4, layerIndex * 4 + 4), 3);
    routingParams.set(
      this.#weights.mhcBResidual.subarray(layerIndex * 16, layerIndex * 16 + 16),
      7,
    );
    this.#device.queue.writeBuffer(this.#routingParams, 0, bufferSource(routingParams));
    if (engramKey && engramValue) {
      this.#device.queue.writeBuffer(this.#engramKey, 0, bufferSource(engramKey));
      this.#device.queue.writeBuffer(this.#engramValue, 0, bufferSource(engramValue));
    }
    const engramSite = this.#weights.geometry.engramLayers.indexOf(layerIndex);
    const residentEngramGroup =
      this.#residentEngramReadyPosition === position && engramSite >= 0
        ? this.#residentEngramGroups[engramSite]?.inject
        : undefined;
    this.#writeAttentionParams(layer, layerIndex, position);

    const encoder = this.#device.createCommandEncoder({ label: "needle.resident-layer.encoder" });
    const rmsPass = encoder.beginComputePass({ label: "needle.resident-layer.rms-lanes" });
    rmsPass.setPipeline(pipelines.rmsLanes);
    rmsPass.setBindGroup(0, this.#getRmsLanesGroup(pipelines.rmsLanes));
    rmsPass.dispatchWorkgroups(1);
    rmsPass.end();

    const prepareLanesPass = encoder.beginComputePass({
      label: "needle.resident-layer.prepare-lanes",
    });
    prepareLanesPass.setPipeline(pipelines.prepare);
    prepareLanesPass.setBindGroup(0, this.#getPrepareLanesGroup(pipelines.prepare));
    prepareLanesPass.dispatchWorkgroups(16);
    prepareLanesPass.end();

    const phiPass = encoder.beginComputePass({ label: "needle.resident-layer.mhc-projections" });
    phiPass.setPipeline(pipelines.cq);
    const phiOutputs = [this.#phiPre, this.#phiPost, this.#phiResidual] as const;
    for (let index = 0; index < phiMatrices.length; index++) {
      const matrix = phiMatrices[index];
      const output = phiOutputs[index];
      const params = this.#projectionParams[index + 5];
      const range = phiRanges[index];
      invariant(matrix && output && params && range, "INVALID_CACT", "Resident mHC job is missing");
      phiPass.setBindGroup(
        0,
        this.#projectionGroup(matrix, this.#preparedLanes, params, output, pipelines.cq),
      );
      phiPass.dispatchWorkgroups(range.rowCount);
    }
    phiPass.end();

    const prePass = encoder.beginComputePass({ label: "needle.resident-layer.mhc-pre" });
    prePass.setPipeline(pipelines.mhcPre);
    prePass.setBindGroup(0, this.#getMhcPreGroup(pipelines.mhcPre));
    prePass.dispatchWorkgroups(1);
    prePass.end();

    if ((engramKey && engramValue) || residentEngramGroup) {
      const engramPass = encoder.beginComputePass({ label: "needle.resident-layer.engram" });
      engramPass.setPipeline(pipelines.engram);
      engramPass.setBindGroup(0, residentEngramGroup ?? this.#getEngramGroup(pipelines.engram));
      engramPass.dispatchWorkgroups(1);
      engramPass.end();
    } else {
      encoder.copyBufferToBuffer(this.#updateInput, 0, this.#blockInput, 0, 512 * 4);
    }

    const inputNormPass = encoder.beginComputePass({ label: "needle.resident-layer.input-norm" });
    inputNormPass.setPipeline(pipelines.norm512);
    inputNormPass.setBindGroup(0, this.#inputNormGroup(layer, pipelines.norm512));
    inputNormPass.dispatchWorkgroups(1);
    inputNormPass.end();

    const prepareInputPass = encoder.beginComputePass({
      label: "needle.resident-layer.prepare-input",
    });
    prepareInputPass.setPipeline(pipelines.prepare);
    prepareInputPass.setBindGroup(0, this.#getPrepareInputGroup(pipelines.prepare));
    prepareInputPass.dispatchWorkgroups(4);
    prepareInputPass.end();

    this.#encodeAttentionAndPost(encoder, layer, layerIndex, position, pipelines);
    if (this.#collectConfidence) {
      this.#confidence.encodePool(
        encoder,
        pipelines.confidencePool,
        this.#nextLanes,
        "needle.confidence.pool-layer",
      );
    }
    encoder.copyBufferToBuffer(this.#nextLanes, 0, this.#lanesInput, 0, 2048 * 4);
    this.#device.queue.submit([encoder.finish()]);
  }

  async finishResidentToken(wantLogits: boolean): Promise<Float32Array | null> {
    invariant(
      this.#residentEnabled && !this.#disposed,
      "BACKEND_UNAVAILABLE",
      "Resident layers are off",
    );
    if (!wantLogits) return null;
    const pipelines = await this.#pipelines;
    const embedding = this.#weights.embedding;
    const encoder = this.#device.createCommandEncoder({ label: "needle.resident-token.final" });
    this.#encodeResidentFinal(encoder, pipelines);
    encoder.copyBufferToBuffer(this.#logits, 0, this.#staging, 0, embedding.outputSize * 4);
    this.#device.queue.submit([encoder.finish()]);
    await this.#staging.mapAsync(MAP_READ, 0, embedding.outputSize * 4);
    try {
      return new Float32Array(this.#staging.getMappedRange(0, embedding.outputSize * 4)).slice();
    } finally {
      this.#staging.unmap();
    }
  }

  async finishResidentTokenForSelection(): Promise<void> {
    invariant(
      this.#residentEnabled && !this.#disposed,
      "BACKEND_UNAVAILABLE",
      "Resident layers are off",
    );
    const pipelines = await this.#pipelines;
    const encoder = this.#device.createCommandEncoder({
      label: "needle.resident-token.final-select",
    });
    this.#encodeResidentFinal(encoder, pipelines);
    this.#device.queue.submit([encoder.finish()]);
  }

  async selectResidentToken(allowedTokenIds?: Uint32Array): Promise<ResidentTokenSelection> {
    invariant(
      this.#residentEnabled && !this.#disposed,
      "BACKEND_UNAVAILABLE",
      "Resident layers are off",
    );
    const pipelines = await this.#pipelines;
    return this.#selector.select(pipelines.selectToken, allowedTokenIds);
  }

  async residentConfidence(): Promise<number | undefined> {
    if (!this.#collectConfidence || !this.#residentEnabled || this.#disposed) return undefined;
    const pipelines = await this.#pipelines;
    return this.#confidence.resolve(pipelines.confidenceHead);
  }

  async readResidentLanes(): Promise<Float32Array> {
    invariant(
      this.#residentEnabled && !this.#disposed,
      "BACKEND_UNAVAILABLE",
      "Resident layers are off",
    );
    const encoder = this.#device.createCommandEncoder({ label: "needle.resident-layer.readback" });
    encoder.copyBufferToBuffer(this.#lanesInput, 0, this.#staging, 0, 2048 * 4);
    this.#device.queue.submit([encoder.finish()]);
    await this.#staging.mapAsync(MAP_READ, 0, 2048 * 4);
    try {
      return new Float32Array(this.#staging.getMappedRange(0, 2048 * 4)).slice();
    } finally {
      this.#staging.unmap();
    }
  }

  async forward(request: FusedAttentionRequest): Promise<FusedAttentionResult> {
    invariant(this.#enabled && !this.#disposed, "BACKEND_UNAVAILABLE", "Fused attention is off");
    const keyCache = this.#keyCache;
    const valueCache = this.#valueCache;
    const keyScales = this.#keyScales;
    const valueScales = this.#valueScales;
    invariant(
      keyCache && valueCache && keyScales && valueScales,
      "BACKEND_UNAVAILABLE",
      "KV cache is missing",
    );
    const { layer, layerIndex, position, input, blockInput, updateInput, x, phiPost, phiResidual } =
      request;
    const geometry = this.#weights.geometry;
    const pipelines = await this.#pipelines;
    const prepared = prepareCqActivation(layer.queryProjection, input);
    this.#device.queue.writeBuffer(this.#preparedInput, 0, bufferSource(prepared));
    if (this.#fuseMlp) {
      this.#device.queue.writeBuffer(this.#blockInput, 0, bufferSource(blockInput));
      this.#device.queue.writeBuffer(this.#updateInput, 0, bufferSource(updateInput));
      this.#device.queue.writeBuffer(
        this.#layerParams,
        0,
        bufferSource(new Uint32Array([floatBits(layer.attentionGate)])),
      );
      if (this.#fuseRouting) {
        this.#device.queue.writeBuffer(this.#lanesInput, 0, bufferSource(x));
        this.#device.queue.writeBuffer(this.#phiPost, 0, bufferSource(phiPost));
        this.#device.queue.writeBuffer(this.#phiResidual, 0, bufferSource(phiResidual));
        const routingParams = new Float32Array(23);
        routingParams[0] = this.#weights.mhcAPost[layerIndex] ?? 0;
        routingParams[1] = this.#weights.mhcAResidual[layerIndex] ?? 0;
        routingParams[2] = layerIndex % 4;
        routingParams.set(this.#weights.mhcBPost.subarray(layerIndex * 4, layerIndex * 4 + 4), 3);
        routingParams.set(
          this.#weights.mhcBResidual.subarray(layerIndex * 16, layerIndex * 16 + 16),
          7,
        );
        this.#device.queue.writeBuffer(this.#routingParams, 0, bufferSource(routingParams));
      }
    }

    const matrices = [
      layer.queryProjection,
      layer.keyProjection,
      layer.valueProjection,
      layer.gateProjection,
      layer.outputProjection,
    ] as const;
    const outputs = [this.#query, this.#key, this.#value, this.#gate, this.#projected] as const;
    const inputs = [
      this.#preparedInput,
      this.#preparedInput,
      this.#preparedInput,
      this.#preparedInput,
      this.#preparedAttention,
    ] as const;
    for (let index = 0; index < matrices.length; index++) {
      const matrix = matrices[index];
      const params = this.#projectionParams[index];
      invariant(matrix && params, "INVALID_CACT", "Attention projection is missing");
      this.#device.queue.writeBuffer(
        params,
        0,
        bufferSource(matrixParameters(matrix, 0, matrix.outputSize)),
      );
    }
    this.#writeAttentionParams(layer, layerIndex, position);

    const encoder = this.#device.createCommandEncoder({ label: "needle.attention.encoder" });
    const projections = encoder.beginComputePass({ label: "needle.attention.projections" });
    projections.setPipeline(pipelines.cq);
    for (let index = 0; index < 4; index++) {
      const matrix = matrices[index];
      const output = outputs[index];
      const projectionInput = inputs[index];
      const params = this.#projectionParams[index];
      invariant(
        matrix && output && projectionInput && params,
        "INVALID_CACT",
        "Projection is missing",
      );
      projections.setBindGroup(
        0,
        this.#projectionGroup(matrix, projectionInput, params, output, pipelines.cq),
      );
      projections.dispatchWorkgroups(matrix.outputSize);
    }
    projections.end();

    const queryPass = encoder.beginComputePass({ label: "needle.attention.query-norm-rope" });
    queryPass.setPipeline(pipelines.query);
    queryPass.setBindGroup(0, this.#queryGroup(layer, pipelines.query));
    queryPass.dispatchWorkgroups(geometry.numberOfHeads);
    queryPass.end();

    const kvPass = encoder.beginComputePass({ label: "needle.attention.kv-store" });
    kvPass.setPipeline(pipelines.kv);
    kvPass.setBindGroup(0, this.#kvGroup(layer, pipelines.kv));
    kvPass.dispatchWorkgroups(geometry.numberOfKVHeads);
    kvPass.end();

    const prefixCount = Math.min(this.#sinkLength, position + 1);
    const recentLow = Math.max(prefixCount, position + 1 - geometry.kvWindow);
    const attentionCount = prefixCount + position + 1 - recentLow;
    const scorePass = encoder.beginComputePass({ label: "needle.attention.scores" });
    scorePass.setPipeline(pipelines.scores);
    scorePass.setBindGroup(0, this.#getScoreGroup(pipelines.scores));
    scorePass.dispatchWorkgroups(geometry.numberOfHeads, attentionCount);
    scorePass.end();

    const attentionPass = encoder.beginComputePass({ label: "needle.attention.softmax" });
    attentionPass.setPipeline(pipelines.attention);
    attentionPass.setBindGroup(0, this.#getAttentionGroup(pipelines.attention));
    attentionPass.dispatchWorkgroups(geometry.numberOfHeads);
    attentionPass.end();

    const preparePass = encoder.beginComputePass({ label: "needle.attention.prepare-output" });
    preparePass.setPipeline(pipelines.prepare);
    preparePass.setBindGroup(0, this.#getPrepareGroup(pipelines.prepare));
    preparePass.dispatchWorkgroups(geometry.modelDimension / 128);
    preparePass.end();

    const outputParams = this.#projectionParams[4];
    invariant(outputParams, "INVALID_CACT", "Output projection parameters are missing");
    const outputPass = encoder.beginComputePass({ label: "needle.attention.output-projection" });
    outputPass.setPipeline(pipelines.cq);
    outputPass.setBindGroup(
      0,
      this.#projectionGroup(
        layer.outputProjection,
        this.#preparedAttention,
        outputParams,
        this.#projected,
        pipelines.cq,
      ),
    );
    outputPass.dispatchWorkgroups(layer.outputProjection.outputSize);
    outputPass.end();

    let readback = this.#projected;
    let kind: FusedAttentionResult["kind"] = "projected";
    if (this.#fuseMlp) {
      const sandwichPass = encoder.beginComputePass({ label: "needle.attention.sandwich" });
      sandwichPass.setPipeline(pipelines.sandwich);
      sandwichPass.setBindGroup(0, this.#sandwichGroup(layer, pipelines.sandwich));
      sandwichPass.dispatchWorkgroups(1);
      sandwichPass.end();

      const mlpPass = encoder.beginComputePass({ label: "needle.attention.hadamard-mlp" });
      mlpPass.setPipeline(pipelines.mlp);
      mlpPass.setBindGroup(0, this.#mlpGroup(layer, pipelines.mlp));
      mlpPass.dispatchWorkgroups(1);
      mlpPass.end();
      readback = this.#delta;
      kind = "delta";

      if (this.#fuseRouting) {
        const routingPass = encoder.beginComputePass({ label: "needle.attention.post-mhc" });
        routingPass.setPipeline(pipelines.routing);
        routingPass.setBindGroup(0, this.#getRoutingGroup(pipelines.routing));
        routingPass.dispatchWorkgroups(1);
        routingPass.end();
        readback = this.#nextLanes;
        kind = "nextX";
      }
    }

    const readbackElements = kind === "nextX" ? 2048 : geometry.modelDimension;

    encoder.copyBufferToBuffer(readback, 0, this.#staging, 0, readbackElements * 4);
    this.#device.queue.submit([encoder.finish()]);
    await this.#staging.mapAsync(MAP_READ, 0, readbackElements * 4);
    try {
      const mapped = this.#staging.getMappedRange(0, readbackElements * 4);
      return { kind, values: new Float32Array(mapped).slice() };
    } finally {
      this.#staging.unmap();
    }
  }

  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    this.#destroyCache();
    this.#selector.dispose();
    this.#confidence.dispose();
    this.#queryParams.destroy();
    this.#kvParams.destroy();
    this.#attentionParams.destroy();
    for (const buffer of [
      this.#codebook,
      this.#preparedInput,
      this.#normalizedLanes,
      this.#preparedLanes,
      this.#attentionInput,
      this.#query,
      this.#key,
      this.#value,
      this.#gate,
      this.#attentionOutput,
      this.#scores,
      this.#preparedAttention,
      this.#projected,
      this.#blockInput,
      this.#updateInput,
      this.#afterAttention,
      this.#mlpInput,
      this.#delta,
      this.#lanesInput,
      this.#phiPre,
      this.#phiPost,
      this.#phiResidual,
      this.#routingParams,
      this.#hPreParams,
      this.#engramKey,
      this.#engramValue,
      this.#engramRowIds,
      this.#engramRowValid,
      this.#residentEngramRing,
      this.#residentEngramRingValid,
      ...this.#residentEngramConcatenated,
      ...this.#residentEngramPrepared,
      ...this.#residentEngramKeys,
      ...this.#residentEngramValueNow,
      ...this.#residentEngramMixed,
      ...this.#residentEngramConvolutionParams,
      ...this.#residentEngramProjectionParams,
      ...this.#residentEngramExtraBuffers,
      this.#nextLanes,
      this.#finalNorm,
      this.#finalHidden,
      this.#preparedFinal,
      this.#logits,
      this.#staging,
      this.#layerParams,
      ...this.#projectionParams,
    ]) {
      buffer.destroy();
    }
    for (const buffer of this.#residentSlotBuffers) buffer.destroy();
    for (const parameter of this.#residentSlotParameters) parameter.destroy();
    for (const matrix of this.#allocatedMatrices) {
      matrix.packed.destroy();
      matrix.norms.destroy();
    }
    for (const norms of this.#allocatedNorms) {
      norms.query.destroy();
      norms.key.destroy();
      norms.input.destroy();
      norms.postAttention.destroy();
      norms.preHadamard.destroy();
      norms.d1.destroy();
      norms.d2.destroy();
      norms.d3.destroy();
    }
  }

  #ensureResidentEngramGroups(pipelines: Pipelines): readonly ResidentEngramGroups[] {
    if (this.#residentEngramGroups.length === this.#weights.engrams.length) {
      return this.#residentEngramGroups;
    }
    for (const buffer of this.#residentEngramExtraBuffers) buffer.destroy();
    this.#residentEngramExtraBuffers = [];
    this.#residentEngramGroups = [];
    for (let site = 0; site < this.#weights.engrams.length; site++) {
      const engram = this.#weights.engrams[site];
      const concatenated = this.#residentEngramConcatenated[site];
      const prepared = this.#residentEngramPrepared[site];
      const keyOutput = this.#residentEngramKeys[site];
      const valueOutput = this.#residentEngramValueNow[site];
      const mixed = this.#residentEngramMixed[site];
      const convolutionParams = this.#residentEngramConvolutionParams[site];
      const keyParams = this.#residentEngramProjectionParams[site * 2];
      const valueParams = this.#residentEngramProjectionParams[site * 2 + 1];
      invariant(
        engram &&
          concatenated &&
          prepared &&
          keyOutput &&
          valueOutput &&
          mixed &&
          convolutionParams &&
          keyParams &&
          valueParams,
        "INVALID_CACT",
        `Missing resident engram resources ${site}`,
      );
      const table = this.#gpuMatrix(engram.tables);
      const gather = this.#device.createBindGroup({
        layout: pipelines.engramGather.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: table.packed } },
          { binding: 1, resource: { buffer: table.norms } },
          { binding: 2, resource: { buffer: this.#codebook } },
          { binding: 3, resource: { buffer: this.#engramRowIds } },
          { binding: 4, resource: { buffer: this.#engramRowValid } },
          { binding: 5, resource: { buffer: concatenated } },
        ],
      });
      const prepare = this.#device.createBindGroup({
        layout: pipelines.prepare.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: concatenated } },
          { binding: 1, resource: { buffer: prepared } },
        ],
      });
      const key = this.#createProjectionGroup(
        engram.keyProjection,
        prepared,
        keyParams,
        keyOutput,
        pipelines.cq,
      );
      const value = this.#createProjectionGroup(
        engram.valueProjection,
        prepared,
        valueParams,
        valueOutput,
        pipelines.cq,
      );
      const taps = bufferWithData(this.#device, `needle.engram.${site}.taps`, engram.taps, STORAGE);
      this.#residentEngramExtraBuffers.push(taps);
      const convolve = this.#device.createBindGroup({
        layout: pipelines.engramConvolve.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: valueOutput } },
          { binding: 1, resource: { buffer: this.#residentEngramRing } },
          { binding: 2, resource: { buffer: taps } },
          { binding: 3, resource: { buffer: this.#residentEngramRingValid } },
          { binding: 4, resource: { buffer: convolutionParams } },
          { binding: 5, resource: { buffer: mixed } },
        ],
      });
      const inject = this.#device.createBindGroup({
        layout: pipelines.engram.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.#updateInput } },
          { binding: 1, resource: { buffer: keyOutput } },
          { binding: 2, resource: { buffer: mixed } },
          { binding: 3, resource: { buffer: this.#blockInput } },
        ],
      });
      this.#residentEngramGroups.push({ gather, prepare, key, value, convolve, inject });
    }
    return this.#residentEngramGroups;
  }

  #ensureResidentSlots(pipelines: Pipelines): readonly ResidentLayerSlot[] {
    if (
      this.#residentSlots.length === this.#weights.geometry.numberOfLayers &&
      this.#residentSlotAllocation === this.#kvAllocation
    ) {
      return this.#residentSlots;
    }
    for (const buffer of this.#residentSlotBuffers) buffer.destroy();
    for (const parameter of this.#residentSlotParameters) parameter.destroy();
    this.#residentSlotBuffers = [];
    this.#residentSlotParameters = [];
    this.#residentSlots = [];
    this.#residentSlotAllocation = this.#kvAllocation;
    const keyCache = this.#keyCache;
    const valueCache = this.#valueCache;
    const keyScales = this.#keyScales;
    const valueScales = this.#valueScales;
    invariant(
      keyCache && valueCache && keyScales && valueScales,
      "BACKEND_UNAVAILABLE",
      "Resident KV buffers are missing",
    );
    const geometry = this.#weights.geometry;
    const makeBuffer = (label: string, data: ArrayBufferView): GPUBuffer => {
      const buffer = bufferWithData(this.#device, label, data, STORAGE);
      this.#residentSlotBuffers.push(buffer);
      return buffer;
    };
    const makeDynamic = (label: string, bytes: number): GPUBuffer => {
      const buffer = storageBuffer(this.#device, label, bytes, COPY_DST);
      this.#residentSlotBuffers.push(buffer);
      return buffer;
    };

    for (let layerIndex = 0; layerIndex < geometry.numberOfLayers; layerIndex++) {
      const layer = this.#weights.layers[layerIndex];
      invariant(layer, "INVALID_CACT", `Missing resident layer ${layerIndex}`);
      const projectionMatrices = [
        layer.queryProjection,
        layer.keyProjection,
        layer.valueProjection,
        layer.gateProjection,
        layer.outputProjection,
      ] as const;
      const projectionInputs = [
        this.#preparedInput,
        this.#preparedInput,
        this.#preparedInput,
        this.#preparedInput,
        this.#preparedAttention,
      ] as const;
      const projectionOutputs = [
        this.#query,
        this.#key,
        this.#value,
        this.#gate,
        this.#projected,
      ] as const;
      const projectionParams: GPUBuffer[] = [];
      const projectionGroups: GPUBindGroup[] = [];
      for (let index = 0; index < projectionMatrices.length; index++) {
        const matrix = projectionMatrices[index];
        const input = projectionInputs[index];
        const output = projectionOutputs[index];
        invariant(matrix && input && output, "INVALID_CACT", "Resident projection is missing");
        const params = makeBuffer(
          `needle.resident.${layerIndex}.projection.${index}`,
          matrixParameters(matrix, 0, matrix.outputSize),
        );
        projectionParams.push(params);
        projectionGroups.push(
          this.#createProjectionGroup(matrix, input, params, output, pipelines.cq),
        );
      }
      const phiMatrices = [
        this.#weights.mhcPhiPre,
        this.#weights.mhcPhiPost,
        this.#weights.mhcPhiResidual,
      ] as const;
      const phiRanges = [
        { rowStart: layerIndex * 4, rowCount: 4 },
        { rowStart: layerIndex * 4, rowCount: 4 },
        { rowStart: layerIndex * 16, rowCount: 16 },
      ] as const;
      const phiOutputs = [this.#phiPre, this.#phiPost, this.#phiResidual] as const;
      const phiGroups: GPUBindGroup[] = [];
      for (let index = 0; index < phiMatrices.length; index++) {
        const matrix = phiMatrices[index];
        const range = phiRanges[index];
        const output = phiOutputs[index];
        invariant(matrix && range && output, "INVALID_CACT", "Resident phi projection is missing");
        const params = makeBuffer(
          `needle.resident.${layerIndex}.phi.${index}`,
          matrixParameters(matrix, range.rowStart, range.rowCount),
        );
        projectionParams.push(params);
        phiGroups.push(
          this.#createProjectionGroup(matrix, this.#preparedLanes, params, output, pipelines.cq),
        );
      }
      const queryParams = this.#parameterFactory.query(
        `needle.resident.${layerIndex}.query-params`,
      );
      const kvParams = this.#parameterFactory.kv(`needle.resident.${layerIndex}.kv-params`);
      const attentionParams = this.#parameterFactory.attention(
        `needle.resident.${layerIndex}.attention-params`,
      );
      this.#residentSlotParameters.push(queryParams, kvParams, attentionParams);
      const hPre = new Float32Array(6);
      hPre[0] = this.#weights.mhcAPre[layerIndex] ?? 0;
      hPre[1] = layerIndex % 4;
      hPre.set(this.#weights.mhcBPre.subarray(layerIndex * 4, layerIndex * 4 + 4), 2);
      const hPreParams = makeBuffer(`needle.resident.${layerIndex}.h-pre`, hPre);
      const routing = new Float32Array(23);
      routing[0] = this.#weights.mhcAPost[layerIndex] ?? 0;
      routing[1] = this.#weights.mhcAResidual[layerIndex] ?? 0;
      routing[2] = layerIndex % 4;
      routing.set(this.#weights.mhcBPost.subarray(layerIndex * 4, layerIndex * 4 + 4), 3);
      routing.set(this.#weights.mhcBResidual.subarray(layerIndex * 16, layerIndex * 16 + 16), 7);
      const routingParams = makeBuffer(`needle.resident.${layerIndex}.routing`, routing);
      const layerParams = makeBuffer(
        `needle.resident.${layerIndex}.layer`,
        new Uint32Array([floatBits(layer.attentionGate)]),
      );
      const norms = this.#norms(layer);
      const queryGroup = this.#device.createBindGroup({
        layout: pipelines.query.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.#query } },
          { binding: 1, resource: { buffer: norms.query } },
          { binding: 2, resource: { buffer: queryParams.buffer } },
        ],
      });
      const kvGroup = this.#device.createBindGroup({
        layout: pipelines.kv.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.#key } },
          { binding: 1, resource: { buffer: this.#value } },
          { binding: 2, resource: { buffer: norms.key } },
          { binding: 3, resource: { buffer: keyCache } },
          { binding: 4, resource: { buffer: valueCache } },
          { binding: 5, resource: { buffer: keyScales } },
          { binding: 6, resource: { buffer: valueScales } },
          { binding: 7, resource: { buffer: kvParams.buffer } },
        ],
      });
      const scoreGroup = this.#device.createBindGroup({
        layout: pipelines.scores.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.#query } },
          { binding: 1, resource: { buffer: keyCache } },
          { binding: 2, resource: { buffer: keyScales } },
          { binding: 3, resource: { buffer: attentionParams.buffer } },
          { binding: 4, resource: { buffer: this.#scores } },
        ],
      });
      const attentionGroup = this.#device.createBindGroup({
        layout: pipelines.attention.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.#gate } },
          { binding: 1, resource: { buffer: valueCache } },
          { binding: 2, resource: { buffer: valueScales } },
          { binding: 3, resource: { buffer: attentionParams.buffer } },
          { binding: 4, resource: { buffer: this.#scores } },
          { binding: 5, resource: { buffer: this.#attentionOutput } },
        ],
      });
      const mhcPreGroup = this.#device.createBindGroup({
        layout: pipelines.mhcPre.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.#lanesInput } },
          { binding: 1, resource: { buffer: this.#phiPre } },
          { binding: 2, resource: { buffer: hPreParams } },
          { binding: 3, resource: { buffer: this.#updateInput } },
        ],
      });
      const inputNormGroup = this.#device.createBindGroup({
        layout: pipelines.norm512.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.#blockInput } },
          { binding: 1, resource: { buffer: norms.input } },
          { binding: 2, resource: { buffer: this.#attentionInput } },
        ],
      });
      const sandwichGroup = this.#device.createBindGroup({
        layout: pipelines.sandwich.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.#projected } },
          { binding: 1, resource: { buffer: norms.postAttention } },
          { binding: 2, resource: { buffer: norms.preHadamard } },
          { binding: 3, resource: { buffer: this.#blockInput } },
          { binding: 4, resource: { buffer: norms.d1 } },
          { binding: 5, resource: { buffer: layerParams } },
          { binding: 6, resource: { buffer: this.#afterAttention } },
          { binding: 7, resource: { buffer: this.#mlpInput } },
        ],
      });
      const mlpGroup = this.#device.createBindGroup({
        layout: pipelines.mlp.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.#mlpInput } },
          { binding: 1, resource: { buffer: norms.d2 } },
          { binding: 2, resource: { buffer: norms.d3 } },
          { binding: 3, resource: { buffer: this.#afterAttention } },
          { binding: 4, resource: { buffer: this.#updateInput } },
          { binding: 5, resource: { buffer: this.#delta } },
        ],
      });
      const routingGroup = this.#device.createBindGroup({
        layout: pipelines.routing.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.#lanesInput } },
          { binding: 1, resource: { buffer: this.#phiPost } },
          { binding: 2, resource: { buffer: this.#phiResidual } },
          { binding: 3, resource: { buffer: this.#delta } },
          { binding: 4, resource: { buffer: routingParams } },
          { binding: 5, resource: { buffer: this.#nextLanes } },
        ],
      });
      let engramKey: GPUBuffer | undefined;
      let engramValue: GPUBuffer | undefined;
      let engramGroup: GPUBindGroup | undefined;
      const engramSite = geometry.engramLayers.indexOf(layerIndex);
      if (engramSite >= 0) {
        const residentEngram = this.#residentEngramGroups[engramSite];
        if (residentEngram) {
          engramGroup = residentEngram.inject;
        } else {
          engramKey = makeDynamic(`needle.resident.${layerIndex}.engram-key`, 512 * 4);
          engramValue = makeDynamic(`needle.resident.${layerIndex}.engram-value`, 512 * 4);
          engramGroup = this.#device.createBindGroup({
            layout: pipelines.engram.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: this.#updateInput } },
              { binding: 1, resource: { buffer: engramKey } },
              { binding: 2, resource: { buffer: engramValue } },
              { binding: 3, resource: { buffer: this.#blockInput } },
            ],
          });
        }
      }
      this.#residentSlots.push({
        layer,
        projectionParams,
        queryParams,
        kvParams,
        attentionParams,
        projectionGroups,
        phiGroups,
        queryGroup,
        kvGroup,
        scoreGroup,
        attentionGroup,
        mhcPreGroup,
        inputNormGroup,
        sandwichGroup,
        mlpGroup,
        routingGroup,
        ...(engramGroup
          ? {
              engramGroup,
              ...(engramKey && engramValue ? { engramKey, engramValue } : {}),
            }
          : {}),
      });
    }
    return this.#residentSlots;
  }

  #encodeResidentSlot(
    encoder: GPUCommandEncoder,
    slot: ResidentLayerSlot,
    position: number,
    hasEngram: boolean,
    pipelines: Pipelines,
  ): void {
    const geometry = this.#weights.geometry;
    const rmsPass = encoder.beginComputePass({ label: "needle.resident-token.rms-lanes" });
    rmsPass.setPipeline(pipelines.rmsLanes);
    rmsPass.setBindGroup(0, this.#getRmsLanesGroup(pipelines.rmsLanes));
    rmsPass.dispatchWorkgroups(1);
    rmsPass.end();
    const prepareLanes = encoder.beginComputePass({ label: "needle.resident-token.prepare-lanes" });
    prepareLanes.setPipeline(pipelines.prepare);
    prepareLanes.setBindGroup(0, this.#getPrepareLanesGroup(pipelines.prepare));
    prepareLanes.dispatchWorkgroups(16);
    prepareLanes.end();
    const phiPass = encoder.beginComputePass({ label: "needle.resident-token.phi" });
    phiPass.setPipeline(pipelines.cq);
    for (let index = 0; index < slot.phiGroups.length; index++) {
      phiPass.setBindGroup(0, slot.phiGroups[index] as GPUBindGroup);
      phiPass.dispatchWorkgroups(index === 2 ? 16 : 4);
    }
    phiPass.end();
    const prePass = encoder.beginComputePass({ label: "needle.resident-token.mhc-pre" });
    prePass.setPipeline(pipelines.mhcPre);
    prePass.setBindGroup(0, slot.mhcPreGroup);
    prePass.dispatchWorkgroups(1);
    prePass.end();
    if (hasEngram && slot.engramGroup) {
      const engramPass = encoder.beginComputePass({ label: "needle.resident-token.engram" });
      engramPass.setPipeline(pipelines.engram);
      engramPass.setBindGroup(0, slot.engramGroup);
      engramPass.dispatchWorkgroups(1);
      engramPass.end();
    } else {
      encoder.copyBufferToBuffer(this.#updateInput, 0, this.#blockInput, 0, 512 * 4);
    }
    const normPass = encoder.beginComputePass({ label: "needle.resident-token.input-norm" });
    normPass.setPipeline(pipelines.norm512);
    normPass.setBindGroup(0, slot.inputNormGroup);
    normPass.dispatchWorkgroups(1);
    normPass.end();
    const prepareInput = encoder.beginComputePass({ label: "needle.resident-token.prepare-input" });
    prepareInput.setPipeline(pipelines.prepare);
    prepareInput.setBindGroup(0, this.#getPrepareInputGroup(pipelines.prepare));
    prepareInput.dispatchWorkgroups(4);
    prepareInput.end();
    const projections = encoder.beginComputePass({ label: "needle.resident-token.projections" });
    projections.setPipeline(pipelines.cq);
    const projectionRows = [512, 256, 256, 512] as const;
    for (let index = 0; index < 4; index++) {
      projections.setBindGroup(0, slot.projectionGroups[index] as GPUBindGroup);
      projections.dispatchWorkgroups(projectionRows[index] ?? 0);
    }
    projections.end();
    const queryPass = encoder.beginComputePass({ label: "needle.resident-token.query" });
    queryPass.setPipeline(pipelines.query);
    queryPass.setBindGroup(0, slot.queryGroup);
    queryPass.dispatchWorkgroups(8);
    queryPass.end();
    const kvPass = encoder.beginComputePass({ label: "needle.resident-token.kv" });
    kvPass.setPipeline(pipelines.kv);
    kvPass.setBindGroup(0, slot.kvGroup);
    kvPass.dispatchWorkgroups(4);
    kvPass.end();
    const prefixCount = Math.min(this.#sinkLength, position + 1);
    const recentLow = Math.max(prefixCount, position + 1 - geometry.kvWindow);
    const attentionCount = prefixCount + position + 1 - recentLow;
    const scorePass = encoder.beginComputePass({ label: "needle.resident-token.scores" });
    scorePass.setPipeline(pipelines.scores);
    scorePass.setBindGroup(0, slot.scoreGroup);
    scorePass.dispatchWorkgroups(8, attentionCount);
    scorePass.end();
    const attentionPass = encoder.beginComputePass({ label: "needle.resident-token.attention" });
    attentionPass.setPipeline(pipelines.attention);
    attentionPass.setBindGroup(0, slot.attentionGroup);
    attentionPass.dispatchWorkgroups(8);
    attentionPass.end();
    const prepareOutput = encoder.beginComputePass({
      label: "needle.resident-token.prepare-output",
    });
    prepareOutput.setPipeline(pipelines.prepare);
    prepareOutput.setBindGroup(0, this.#getPrepareGroup(pipelines.prepare));
    prepareOutput.dispatchWorkgroups(4);
    prepareOutput.end();
    const outputPass = encoder.beginComputePass({ label: "needle.resident-token.output" });
    outputPass.setPipeline(pipelines.cq);
    outputPass.setBindGroup(0, slot.projectionGroups[4] as GPUBindGroup);
    outputPass.dispatchWorkgroups(512);
    outputPass.end();
    const sandwich = encoder.beginComputePass({ label: "needle.resident-token.sandwich" });
    sandwich.setPipeline(pipelines.sandwich);
    sandwich.setBindGroup(0, slot.sandwichGroup);
    sandwich.dispatchWorkgroups(1);
    sandwich.end();
    const mlp = encoder.beginComputePass({ label: "needle.resident-token.mlp" });
    mlp.setPipeline(pipelines.mlp);
    mlp.setBindGroup(0, slot.mlpGroup);
    mlp.dispatchWorkgroups(1);
    mlp.end();
    const routing = encoder.beginComputePass({ label: "needle.resident-token.routing" });
    routing.setPipeline(pipelines.routing);
    routing.setBindGroup(0, slot.routingGroup);
    routing.dispatchWorkgroups(1);
    routing.end();
    if (this.#collectConfidence) {
      this.#confidence.encodePool(
        encoder,
        pipelines.confidencePool,
        this.#nextLanes,
        "needle.resident-token.confidence",
      );
    }
    encoder.copyBufferToBuffer(this.#nextLanes, 0, this.#lanesInput, 0, 2048 * 4);
  }

  #createProjectionGroup(
    matrix: CqMatrix,
    input: GPUBuffer,
    params: GPUBuffer,
    output: GPUBuffer,
    pipeline: GPUComputePipeline,
  ): GPUBindGroup {
    const gpuMatrix = this.#gpuMatrix(matrix);
    return this.#device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: gpuMatrix.packed } },
        { binding: 1, resource: { buffer: gpuMatrix.norms } },
        { binding: 2, resource: { buffer: input } },
        { binding: 3, resource: { buffer: this.#codebook } },
        { binding: 4, resource: { buffer: params } },
        { binding: 5, resource: { buffer: output } },
      ],
    });
  }

  #encodeResidentFinal(encoder: GPUCommandEncoder, pipelines: Pipelines): void {
    const embedding = this.#weights.embedding;
    const params = this.#projectionParams[8];
    invariant(params, "INVALID_CACT", "Final projection parameters are missing");
    this.#device.queue.writeBuffer(
      params,
      0,
      bufferSource(matrixParameters(embedding, 0, embedding.outputSize)),
    );
    const normPass = encoder.beginComputePass({ label: "needle.resident-token.final-norm" });
    normPass.setPipeline(pipelines.finalNorm);
    normPass.setBindGroup(0, this.#getFinalNormGroup(pipelines.finalNorm));
    normPass.dispatchWorkgroups(1);
    normPass.end();

    const preparePass = encoder.beginComputePass({ label: "needle.resident-token.prepare-final" });
    preparePass.setPipeline(pipelines.prepare);
    preparePass.setBindGroup(0, this.#getPrepareFinalGroup(pipelines.prepare));
    preparePass.dispatchWorkgroups(4);
    preparePass.end();

    const logitsPass = encoder.beginComputePass({ label: "needle.resident-token.logits" });
    logitsPass.setPipeline(pipelines.cq);
    logitsPass.setBindGroup(
      0,
      this.#projectionGroup(embedding, this.#preparedFinal, params, this.#logits, pipelines.cq),
    );
    logitsPass.dispatchWorkgroups(embedding.outputSize);
    logitsPass.end();
  }

  #writeAttentionParams(layer: CactLayer, layerIndex: number, position: number): void {
    const geometry = this.#weights.geometry;
    const thetaBits = floatBits(geometry.ropeTheta);
    this.#queryParams.write({ position, thetaBits });
    this.#kvParams.write({
      layer: layerIndex,
      position,
      allocation: this.#kvAllocation,
      sinkLength: this.#sinkLength,
      window: geometry.kvWindow,
      kvHeads: geometry.numberOfKVHeads,
      thetaBits,
      reserved: 0,
    });
    this.#attentionParams.write({
      layer: layerIndex,
      position,
      allocation: this.#kvAllocation,
      sinkLength: this.#sinkLength,
      window: geometry.kvWindow,
      kvHeads: geometry.numberOfKVHeads,
      heads: geometry.numberOfHeads,
      reserved: 0,
    });
    this.#device.queue.writeBuffer(
      this.#layerParams,
      0,
      bufferSource(new Uint32Array([floatBits(layer.attentionGate)])),
    );
  }

  #encodeAttentionAndPost(
    encoder: GPUCommandEncoder,
    layer: CactLayer,
    _layerIndex: number,
    position: number,
    pipelines: Pipelines,
  ): void {
    const geometry = this.#weights.geometry;
    const matrices = [
      layer.queryProjection,
      layer.keyProjection,
      layer.valueProjection,
      layer.gateProjection,
    ] as const;
    const outputs = [this.#query, this.#key, this.#value, this.#gate] as const;
    const projections = encoder.beginComputePass({ label: "needle.resident-layer.projections" });
    projections.setPipeline(pipelines.cq);
    for (let index = 0; index < matrices.length; index++) {
      const matrix = matrices[index];
      const output = outputs[index];
      const params = this.#projectionParams[index];
      invariant(matrix && output && params, "INVALID_CACT", "Resident projection is missing");
      projections.setBindGroup(
        0,
        this.#projectionGroup(matrix, this.#preparedInput, params, output, pipelines.cq),
      );
      projections.dispatchWorkgroups(matrix.outputSize);
    }
    projections.end();

    const queryPass = encoder.beginComputePass({ label: "needle.resident-layer.query" });
    queryPass.setPipeline(pipelines.query);
    queryPass.setBindGroup(0, this.#queryGroup(layer, pipelines.query));
    queryPass.dispatchWorkgroups(geometry.numberOfHeads);
    queryPass.end();

    const kvPass = encoder.beginComputePass({ label: "needle.resident-layer.kv" });
    kvPass.setPipeline(pipelines.kv);
    kvPass.setBindGroup(0, this.#kvGroup(layer, pipelines.kv));
    kvPass.dispatchWorkgroups(geometry.numberOfKVHeads);
    kvPass.end();

    const prefixCount = Math.min(this.#sinkLength, position + 1);
    const recentLow = Math.max(prefixCount, position + 1 - geometry.kvWindow);
    const attentionCount = prefixCount + position + 1 - recentLow;
    const scorePass = encoder.beginComputePass({ label: "needle.resident-layer.scores" });
    scorePass.setPipeline(pipelines.scores);
    scorePass.setBindGroup(0, this.#getScoreGroup(pipelines.scores));
    scorePass.dispatchWorkgroups(geometry.numberOfHeads, attentionCount);
    scorePass.end();

    const attentionPass = encoder.beginComputePass({ label: "needle.resident-layer.attention" });
    attentionPass.setPipeline(pipelines.attention);
    attentionPass.setBindGroup(0, this.#getAttentionGroup(pipelines.attention));
    attentionPass.dispatchWorkgroups(geometry.numberOfHeads);
    attentionPass.end();

    const preparePass = encoder.beginComputePass({ label: "needle.resident-layer.prepare-output" });
    preparePass.setPipeline(pipelines.prepare);
    preparePass.setBindGroup(0, this.#getPrepareGroup(pipelines.prepare));
    preparePass.dispatchWorkgroups(4);
    preparePass.end();

    const outputParams = this.#projectionParams[4];
    invariant(outputParams, "INVALID_CACT", "Output projection parameters are missing");
    const outputPass = encoder.beginComputePass({ label: "needle.resident-layer.output" });
    outputPass.setPipeline(pipelines.cq);
    outputPass.setBindGroup(
      0,
      this.#projectionGroup(
        layer.outputProjection,
        this.#preparedAttention,
        outputParams,
        this.#projected,
        pipelines.cq,
      ),
    );
    outputPass.dispatchWorkgroups(512);
    outputPass.end();

    const sandwichPass = encoder.beginComputePass({ label: "needle.resident-layer.sandwich" });
    sandwichPass.setPipeline(pipelines.sandwich);
    sandwichPass.setBindGroup(0, this.#sandwichGroup(layer, pipelines.sandwich));
    sandwichPass.dispatchWorkgroups(1);
    sandwichPass.end();

    const mlpPass = encoder.beginComputePass({ label: "needle.resident-layer.mlp" });
    mlpPass.setPipeline(pipelines.mlp);
    mlpPass.setBindGroup(0, this.#mlpGroup(layer, pipelines.mlp));
    mlpPass.dispatchWorkgroups(1);
    mlpPass.end();

    const routingPass = encoder.beginComputePass({ label: "needle.resident-layer.routing" });
    routingPass.setPipeline(pipelines.routing);
    routingPass.setBindGroup(0, this.#getRoutingGroup(pipelines.routing));
    routingPass.dispatchWorkgroups(1);
    routingPass.end();
  }

  #destroyCache(): void {
    this.#keyCache?.destroy();
    this.#valueCache?.destroy();
    this.#keyScales?.destroy();
    this.#valueScales?.destroy();
    this.#keyCache = undefined;
    this.#valueCache = undefined;
    this.#keyScales = undefined;
    this.#valueScales = undefined;
  }

  #gpuMatrix(matrix: CqMatrix): GpuMatrix {
    const cached = this.#matrixCache.get(matrix);
    if (cached) return cached;
    const data = webGpuMatrixData(matrix);
    const result = {
      packed: bufferWithData(
        this.#device,
        "needle.attention.matrix.packed",
        data.packedWords,
        STORAGE,
      ),
      norms: bufferWithData(this.#device, "needle.attention.matrix.norms", data.norms, STORAGE),
    };
    this.#matrixCache.set(matrix, result);
    this.#allocatedMatrices.push(result);
    return result;
  }

  #projectionGroup(
    matrix: CqMatrix,
    input: GPUBuffer,
    params: GPUBuffer,
    output: GPUBuffer,
    pipeline: GPUComputePipeline,
  ): GPUBindGroup {
    const cached = this.#projectionGroups.get(matrix);
    if (cached) return cached;
    const gpuMatrix = this.#gpuMatrix(matrix);
    const group = this.#device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: gpuMatrix.packed } },
        { binding: 1, resource: { buffer: gpuMatrix.norms } },
        { binding: 2, resource: { buffer: input } },
        { binding: 3, resource: { buffer: this.#codebook } },
        { binding: 4, resource: { buffer: params } },
        { binding: 5, resource: { buffer: output } },
      ],
    });
    this.#projectionGroups.set(matrix, group);
    return group;
  }

  #norms(layer: CactLayer): LayerNormBuffers {
    const cached = this.#layerNorms.get(layer);
    if (cached) return cached;
    const result = {
      query: bufferWithData(this.#device, "needle.attention.query-norm", layer.queryNorm, STORAGE),
      key: bufferWithData(this.#device, "needle.attention.key-norm", layer.keyNorm, STORAGE),
      input: bufferWithData(this.#device, "needle.attention.input-norm", layer.normInput, STORAGE),
      postAttention: bufferWithData(
        this.#device,
        "needle.attention.post-norm",
        layer.postAttentionNorm,
        STORAGE,
      ),
      preHadamard: bufferWithData(
        this.#device,
        "needle.attention.pre-hadamard-norm",
        layer.preHadamardNorm,
        STORAGE,
      ),
      d1: bufferWithData(this.#device, "needle.attention.hadamard-d1", layer.hadamardD1, STORAGE),
      d2: bufferWithData(this.#device, "needle.attention.hadamard-d2", layer.hadamardD2, STORAGE),
      d3: bufferWithData(this.#device, "needle.attention.hadamard-d3", layer.hadamardD3, STORAGE),
    };
    this.#layerNorms.set(layer, result);
    this.#allocatedNorms.push(result);
    return result;
  }

  #getRmsLanesGroup(pipeline: GPUComputePipeline): GPUBindGroup {
    if (this.#rmsLanesGroup) return this.#rmsLanesGroup;
    this.#rmsLanesGroup = this.#device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.#lanesInput } },
        { binding: 1, resource: { buffer: this.#normalizedLanes } },
      ],
    });
    return this.#rmsLanesGroup;
  }

  #getPrepareLanesGroup(pipeline: GPUComputePipeline): GPUBindGroup {
    if (this.#prepareLanesGroup) return this.#prepareLanesGroup;
    this.#prepareLanesGroup = this.#device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.#normalizedLanes } },
        { binding: 1, resource: { buffer: this.#preparedLanes } },
      ],
    });
    return this.#prepareLanesGroup;
  }

  #getMhcPreGroup(pipeline: GPUComputePipeline): GPUBindGroup {
    if (this.#mhcPreGroup) return this.#mhcPreGroup;
    this.#mhcPreGroup = this.#device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.#lanesInput } },
        { binding: 1, resource: { buffer: this.#phiPre } },
        { binding: 2, resource: { buffer: this.#hPreParams } },
        { binding: 3, resource: { buffer: this.#updateInput } },
      ],
    });
    return this.#mhcPreGroup;
  }

  #getEngramGroup(pipeline: GPUComputePipeline): GPUBindGroup {
    if (this.#engramGroup) return this.#engramGroup;
    this.#engramGroup = this.#device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.#updateInput } },
        { binding: 1, resource: { buffer: this.#engramKey } },
        { binding: 2, resource: { buffer: this.#engramValue } },
        { binding: 3, resource: { buffer: this.#blockInput } },
      ],
    });
    return this.#engramGroup;
  }

  #inputNormGroup(layer: CactLayer, pipeline: GPUComputePipeline): GPUBindGroup {
    const cached = this.#inputNormGroups.get(layer);
    if (cached) return cached;
    const group = this.#device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.#blockInput } },
        { binding: 1, resource: { buffer: this.#norms(layer).input } },
        { binding: 2, resource: { buffer: this.#attentionInput } },
      ],
    });
    this.#inputNormGroups.set(layer, group);
    return group;
  }

  #getPrepareInputGroup(pipeline: GPUComputePipeline): GPUBindGroup {
    if (this.#prepareInputGroup) return this.#prepareInputGroup;
    this.#prepareInputGroup = this.#device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.#attentionInput } },
        { binding: 1, resource: { buffer: this.#preparedInput } },
      ],
    });
    return this.#prepareInputGroup;
  }

  #getFinalNormGroup(pipeline: GPUComputePipeline): GPUBindGroup {
    if (this.#finalNormGroup) return this.#finalNormGroup;
    this.#finalNormGroup = this.#device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.#lanesInput } },
        { binding: 1, resource: { buffer: this.#finalNorm } },
        { binding: 2, resource: { buffer: this.#finalHidden } },
      ],
    });
    return this.#finalNormGroup;
  }

  #getPrepareFinalGroup(pipeline: GPUComputePipeline): GPUBindGroup {
    if (this.#prepareFinalGroup) return this.#prepareFinalGroup;
    this.#prepareFinalGroup = this.#device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.#finalHidden } },
        { binding: 1, resource: { buffer: this.#preparedFinal } },
      ],
    });
    return this.#prepareFinalGroup;
  }

  #queryGroup(layer: CactLayer, pipeline: GPUComputePipeline): GPUBindGroup {
    const cached = this.#queryGroups.get(layer);
    if (cached) return cached;
    const group = this.#device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.#query } },
        { binding: 1, resource: { buffer: this.#norms(layer).query } },
        { binding: 2, resource: { buffer: this.#queryParams.buffer } },
      ],
    });
    this.#queryGroups.set(layer, group);
    return group;
  }

  #kvGroup(layer: CactLayer, pipeline: GPUComputePipeline): GPUBindGroup {
    const cached = this.#kvGroups.get(layer);
    if (cached) return cached;
    const keyCache = this.#keyCache;
    const valueCache = this.#valueCache;
    const keyScales = this.#keyScales;
    const valueScales = this.#valueScales;
    invariant(
      keyCache && valueCache && keyScales && valueScales,
      "BACKEND_UNAVAILABLE",
      "KV cache is missing",
    );
    const group = this.#device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.#key } },
        { binding: 1, resource: { buffer: this.#value } },
        { binding: 2, resource: { buffer: this.#norms(layer).key } },
        { binding: 3, resource: { buffer: keyCache } },
        { binding: 4, resource: { buffer: valueCache } },
        { binding: 5, resource: { buffer: keyScales } },
        { binding: 6, resource: { buffer: valueScales } },
        { binding: 7, resource: { buffer: this.#kvParams.buffer } },
      ],
    });
    this.#kvGroups.set(layer, group);
    return group;
  }

  #getScoreGroup(pipeline: GPUComputePipeline): GPUBindGroup {
    if (this.#scoreGroup) return this.#scoreGroup;
    const keyCache = this.#keyCache;
    const keyScales = this.#keyScales;
    invariant(keyCache && keyScales, "BACKEND_UNAVAILABLE", "KV cache is missing");
    this.#scoreGroup = this.#device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.#query } },
        { binding: 1, resource: { buffer: keyCache } },
        { binding: 2, resource: { buffer: keyScales } },
        { binding: 3, resource: { buffer: this.#attentionParams.buffer } },
        { binding: 4, resource: { buffer: this.#scores } },
      ],
    });
    return this.#scoreGroup;
  }

  #getAttentionGroup(pipeline: GPUComputePipeline): GPUBindGroup {
    if (this.#attentionGroup) return this.#attentionGroup;
    const valueCache = this.#valueCache;
    const valueScales = this.#valueScales;
    invariant(valueCache && valueScales, "BACKEND_UNAVAILABLE", "KV cache is missing");
    this.#attentionGroup = this.#device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.#gate } },
        { binding: 1, resource: { buffer: valueCache } },
        { binding: 2, resource: { buffer: valueScales } },
        { binding: 3, resource: { buffer: this.#attentionParams.buffer } },
        { binding: 4, resource: { buffer: this.#scores } },
        { binding: 5, resource: { buffer: this.#attentionOutput } },
      ],
    });
    return this.#attentionGroup;
  }

  #getPrepareGroup(pipeline: GPUComputePipeline): GPUBindGroup {
    if (this.#prepareGroup) return this.#prepareGroup;
    this.#prepareGroup = this.#device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.#attentionOutput } },
        { binding: 1, resource: { buffer: this.#preparedAttention } },
      ],
    });
    return this.#prepareGroup;
  }

  #sandwichGroup(layer: CactLayer, pipeline: GPUComputePipeline): GPUBindGroup {
    const cached = this.#sandwichGroups.get(layer);
    if (cached) return cached;
    const norms = this.#norms(layer);
    const group = this.#device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.#projected } },
        { binding: 1, resource: { buffer: norms.postAttention } },
        { binding: 2, resource: { buffer: norms.preHadamard } },
        { binding: 3, resource: { buffer: this.#blockInput } },
        { binding: 4, resource: { buffer: norms.d1 } },
        { binding: 5, resource: { buffer: this.#layerParams } },
        { binding: 6, resource: { buffer: this.#afterAttention } },
        { binding: 7, resource: { buffer: this.#mlpInput } },
      ],
    });
    this.#sandwichGroups.set(layer, group);
    return group;
  }

  #mlpGroup(layer: CactLayer, pipeline: GPUComputePipeline): GPUBindGroup {
    const cached = this.#mlpGroups.get(layer);
    if (cached) return cached;
    const norms = this.#norms(layer);
    const group = this.#device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.#mlpInput } },
        { binding: 1, resource: { buffer: norms.d2 } },
        { binding: 2, resource: { buffer: norms.d3 } },
        { binding: 3, resource: { buffer: this.#afterAttention } },
        { binding: 4, resource: { buffer: this.#updateInput } },
        { binding: 5, resource: { buffer: this.#delta } },
      ],
    });
    this.#mlpGroups.set(layer, group);
    return group;
  }

  #getRoutingGroup(pipeline: GPUComputePipeline): GPUBindGroup {
    if (this.#routingGroup) return this.#routingGroup;
    this.#routingGroup = this.#device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.#lanesInput } },
        { binding: 1, resource: { buffer: this.#phiPost } },
        { binding: 2, resource: { buffer: this.#phiResidual } },
        { binding: 3, resource: { buffer: this.#delta } },
        { binding: 4, resource: { buffer: this.#routingParams } },
        { binding: 5, resource: { buffer: this.#nextLanes } },
      ],
    });
    return this.#routingGroup;
  }
}
