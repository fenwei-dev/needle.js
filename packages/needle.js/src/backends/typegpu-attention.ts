/// <reference types="@webgpu/types" preserve="true" />

import { invariant } from "../errors.js";
import type { CactLayer, CactWeights, CqMatrix } from "../model/cact.js";
import type {
  FusedAttentionRequest,
  FusedAttentionResetOptions,
  FusedAttentionResult,
  FusedAttentionSession,
  ResidentLayerRequest,
  ResidentTokenSelection,
} from "./backend.js";
import { prepareCqActivation } from "./cq.js";
import {
  ATTENTION_SCORES_WGSL,
  ATTENTION_SOFTMAX_GATE_WGSL,
  CONFIDENCE_HEAD_WGSL,
  CONFIDENCE_POOL_WGSL,
  ENGRAM_INJECT_WGSL,
  FINAL_NORM_WGSL,
  HADAMARD_MLP_DELTA_WGSL,
  KV_NORM_ROPE_STORE_WGSL,
  MHC_PRE_WGSL,
  POST_MHC_ROUTING_WGSL,
  PREPARE_ATTENTION_WGSL,
  QUERY_NORM_ROPE_WGSL,
  RMS_LANES_WGSL,
  RMS_NORM_512_WGSL,
  SANDWICH_PREPARE_MLP_WGSL,
  SELECT_TOKEN_WGSL,
} from "./typegpu-attention-kernel.js";
import { CQ_MATVEC_WGSL, matrixParameters, webGpuMatrixData } from "./webgpu-kernel.js";

const MAP_READ = 0x0001;
const COPY_SRC = 0x0004;
const COPY_DST = 0x0008;
const STORAGE = 0x0080;

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

interface Pipelines {
  readonly cq: GPUComputePipeline;
  readonly query: GPUComputePipeline;
  readonly kv: GPUComputePipeline;
  readonly scores: GPUComputePipeline;
  readonly attention: GPUComputePipeline;
  readonly prepare: GPUComputePipeline;
  readonly sandwich: GPUComputePipeline;
  readonly mlp: GPUComputePipeline;
  readonly routing: GPUComputePipeline;
  readonly rmsLanes: GPUComputePipeline;
  readonly mhcPre: GPUComputePipeline;
  readonly engram: GPUComputePipeline;
  readonly norm512: GPUComputePipeline;
  readonly finalNorm: GPUComputePipeline;
  readonly confidencePool: GPUComputePipeline;
  readonly confidenceHead: GPUComputePipeline;
  readonly selectToken: GPUComputePipeline;
}

export class TypeGPUFusedAttentionSession implements FusedAttentionSession {
  readonly #device: GPUDevice;
  readonly #weights: CactWeights;
  readonly #fuseMlp: boolean;
  readonly #fuseRouting: boolean;
  readonly #residentLayers: boolean;
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
  readonly #nextLanes: GPUBuffer;
  readonly #finalNorm: GPUBuffer;
  readonly #finalHidden: GPUBuffer;
  readonly #preparedFinal: GPUBuffer;
  readonly #logits: GPUBuffer;
  readonly #confidenceProbes: GPUBuffer | undefined;
  readonly #confidenceProjection: GPUBuffer | undefined;
  readonly #confidenceBias: GPUBuffer | undefined;
  readonly #confidenceMaxima: GPUBuffer | undefined;
  readonly #confidenceDenominators: GPUBuffer | undefined;
  readonly #confidenceWeighted: GPUBuffer | undefined;
  readonly #confidenceResult: GPUBuffer | undefined;
  readonly #allowedTokens: GPUBuffer;
  readonly #selectionParams: GPUBuffer;
  readonly #selectionResult: GPUBuffer;
  readonly #staging: GPUBuffer;
  readonly #queryParams: GPUBuffer;
  readonly #kvParams: GPUBuffer;
  readonly #attentionParams: GPUBuffer;
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
  #prepareInputGroup: GPUBindGroup | undefined;
  #finalNormGroup: GPUBindGroup | undefined;
  #prepareFinalGroup: GPUBindGroup | undefined;
  #confidenceInputGroup: GPUBindGroup | undefined;
  #confidenceNextGroup: GPUBindGroup | undefined;
  #confidenceHeadGroup: GPUBindGroup | undefined;
  #selectionGroup: GPUBindGroup | undefined;
  #inputNormGroups = new WeakMap<CactLayer, GPUBindGroup>();
  #residentEnabled = false;
  #collectConfidence = false;
  #kvAllocation = 0;
  #sinkLength = 0;
  #enabled = false;
  #disposed = false;

  constructor(
    device: GPUDevice,
    weights: CactWeights,
    fuseMlp: boolean,
    fuseRouting: boolean,
    residentLayers: boolean,
  ) {
    this.#device = device;
    this.#weights = weights;
    this.#fuseMlp = fuseMlp;
    this.#fuseRouting = fuseRouting;
    this.#residentLayers = residentLayers;
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
    const confidence = weights.heads.get("confidence");
    const supportedConfidence =
      confidence?.probeCount === 8 &&
      confidence.outputSize === 1 &&
      confidence.probes.length === 4096 &&
      confidence.projection.length === 4096;
    this.#confidenceProbes = supportedConfidence
      ? bufferWithData(device, "needle.confidence.probes", confidence.probes, STORAGE)
      : undefined;
    this.#confidenceProjection = supportedConfidence
      ? bufferWithData(device, "needle.confidence.projection", confidence.projection, STORAGE)
      : undefined;
    this.#confidenceBias = supportedConfidence
      ? bufferWithData(device, "needle.confidence.bias", confidence.bias, STORAGE)
      : undefined;
    this.#confidenceMaxima = supportedConfidence
      ? storageBuffer(device, "needle.confidence.maxima", 8 * 4, COPY_DST)
      : undefined;
    this.#confidenceDenominators = supportedConfidence
      ? storageBuffer(device, "needle.confidence.denominators", 8 * 4, COPY_DST)
      : undefined;
    this.#confidenceWeighted = supportedConfidence
      ? storageBuffer(device, "needle.confidence.weighted", 4096 * 4, COPY_DST)
      : undefined;
    this.#confidenceResult = supportedConfidence
      ? storageBuffer(device, "needle.confidence.result", 4, COPY_SRC)
      : undefined;
    this.#allowedTokens = storageBuffer(
      device,
      "needle.selection.allowed",
      geometry.vocabularySize * 4,
      COPY_DST,
    );
    this.#selectionParams = storageBuffer(device, "needle.selection.params", 8, COPY_DST);
    this.#selectionResult = storageBuffer(device, "needle.selection.result", 8, COPY_SRC);
    this.#staging = device.createBuffer({
      label: "needle.attention.readback",
      size: Math.max(lanesBytes, geometry.vocabularySize * 4),
      usage: MAP_READ | COPY_DST,
    });
    this.#queryParams = storageBuffer(device, "needle.attention.query-params", 8, COPY_DST);
    this.#kvParams = storageBuffer(device, "needle.attention.kv-params", 32, COPY_DST);
    this.#attentionParams = storageBuffer(device, "needle.attention.params", 32, COPY_DST);
    this.#layerParams = storageBuffer(device, "needle.attention.layer-params", 4, COPY_DST);
    this.#projectionParams = Array.from({ length: 9 }, (_, index) =>
      storageBuffer(device, `needle.attention.projection-params.${index}`, 32, COPY_DST),
    );
    this.#pipelines = createPipelines(device);
  }

  reset(options: FusedAttentionResetOptions): boolean {
    invariant(!this.#disposed, "BACKEND_UNAVAILABLE", "Fused attention was disposed");
    const geometry = this.#weights.geometry;
    this.#enabled =
      options.kvCache === "int8" &&
      geometry.modelDimension === 512 &&
      geometry.headDimension === 64 &&
      geometry.numberOfHeads === 8 &&
      geometry.numberOfKVHeads === 4 &&
      geometry.mhcLanes === 4 &&
      geometry.hadamardDimension === 512 &&
      geometry.kvWindow > 0 &&
      this.#weights.layers.every(
        (layer) =>
          layer.queryProjection.inputSize === 512 &&
          layer.queryProjection.outputSize === 512 &&
          layer.keyProjection.inputSize === 512 &&
          layer.keyProjection.outputSize === 256 &&
          layer.keyProjection.groupSize === layer.queryProjection.groupSize &&
          layer.valueProjection.inputSize === 512 &&
          layer.valueProjection.outputSize === 256 &&
          layer.valueProjection.groupSize === layer.queryProjection.groupSize &&
          layer.gateProjection.inputSize === 512 &&
          layer.gateProjection.outputSize === 512 &&
          layer.gateProjection.groupSize === layer.queryProjection.groupSize &&
          layer.outputProjection.inputSize === 512 &&
          layer.outputProjection.outputSize === 512 &&
          layer.outputProjection.groupSize === 128 &&
          layer.hadamardD1.length === 512 &&
          layer.hadamardD2.length === 512 &&
          layer.hadamardD3.length === 512,
      ) &&
      options.sinkLength + geometry.kvWindow <= 512;
    const confidenceAvailable = Boolean(
      this.#confidenceMaxima &&
        this.#confidenceDenominators &&
        this.#confidenceWeighted &&
        this.#confidenceResult,
    );
    this.#residentEnabled =
      this.#enabled &&
      this.#residentLayers &&
      this.#fuseRouting &&
      (!options.collectConfidence || confidenceAvailable);
    this.#collectConfidence = this.#residentEnabled && options.collectConfidence;
    if (!this.#enabled) return false;
    if (this.#collectConfidence) {
      const maxima = new Float32Array(8);
      maxima.fill(Number.NEGATIVE_INFINITY);
      this.#device.queue.writeBuffer(this.#confidenceMaxima as GPUBuffer, 0, bufferSource(maxima));
      this.#device.queue.writeBuffer(
        this.#confidenceDenominators as GPUBuffer,
        0,
        bufferSource(new Float32Array(8)),
      );
      this.#device.queue.writeBuffer(
        this.#confidenceWeighted as GPUBuffer,
        0,
        bufferSource(new Float32Array(4096)),
      );
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
      const pass = encoder.beginComputePass({ label: "needle.confidence.pool-embedding" });
      pass.setPipeline(pipelines.confidencePool);
      pass.setBindGroup(0, this.#getConfidenceInputGroup(pipelines.confidencePool));
      pass.dispatchWorkgroups(1);
      pass.end();
      this.#device.queue.submit([encoder.finish()]);
    }
    return true;
  }

  residentLayersEnabled(): boolean {
    return this.#residentEnabled;
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

    if (engramKey && engramValue) {
      const engramPass = encoder.beginComputePass({ label: "needle.resident-layer.engram" });
      engramPass.setPipeline(pipelines.engram);
      engramPass.setBindGroup(0, this.#getEngramGroup(pipelines.engram));
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
      const confidencePass = encoder.beginComputePass({ label: "needle.confidence.pool-layer" });
      confidencePass.setPipeline(pipelines.confidencePool);
      confidencePass.setBindGroup(0, this.#getConfidenceNextGroup(pipelines.confidencePool));
      confidencePass.dispatchWorkgroups(1);
      confidencePass.end();
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
    const count = allowedTokenIds?.length ?? this.#weights.geometry.vocabularySize;
    invariant(
      count > 0 && count <= 8192,
      "INVALID_CACT",
      "Allowed token set is empty or too large",
    );
    if (allowedTokenIds) {
      this.#device.queue.writeBuffer(this.#allowedTokens, 0, bufferSource(allowedTokenIds));
    }
    this.#device.queue.writeBuffer(
      this.#selectionParams,
      0,
      bufferSource(new Uint32Array([count, allowedTokenIds ? 1 : 0])),
    );
    const pipelines = await this.#pipelines;
    const encoder = this.#device.createCommandEncoder({ label: "needle.selection.encoder" });
    const pass = encoder.beginComputePass({ label: "needle.selection.argmax" });
    pass.setPipeline(pipelines.selectToken);
    pass.setBindGroup(0, this.#getSelectionGroup(pipelines.selectToken));
    pass.dispatchWorkgroups(1);
    pass.end();
    encoder.copyBufferToBuffer(this.#selectionResult, 0, this.#staging, 0, 8);
    this.#device.queue.submit([encoder.finish()]);
    await this.#staging.mapAsync(MAP_READ, 0, 8);
    try {
      const view = new DataView(this.#staging.getMappedRange(0, 8));
      return { id: view.getUint32(0, true), logProbability: view.getFloat32(4, true) };
    } finally {
      this.#staging.unmap();
    }
  }

  async residentConfidence(): Promise<number | undefined> {
    if (!this.#collectConfidence || !this.#residentEnabled || this.#disposed) return undefined;
    const result = this.#confidenceResult;
    invariant(result, "BACKEND_UNAVAILABLE", "Resident confidence output is missing");
    const pipelines = await this.#pipelines;
    const encoder = this.#device.createCommandEncoder({ label: "needle.confidence.final" });
    const pass = encoder.beginComputePass({ label: "needle.confidence.head" });
    pass.setPipeline(pipelines.confidenceHead);
    pass.setBindGroup(0, this.#getConfidenceHeadGroup(pipelines.confidenceHead));
    pass.dispatchWorkgroups(1);
    pass.end();
    encoder.copyBufferToBuffer(result, 0, this.#staging, 0, 4);
    this.#device.queue.submit([encoder.finish()]);
    await this.#staging.mapAsync(MAP_READ, 0, 4);
    try {
      return new Float32Array(this.#staging.getMappedRange(0, 4))[0];
    } finally {
      this.#staging.unmap();
    }
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
    const thetaBits = floatBits(geometry.ropeTheta);
    this.#device.queue.writeBuffer(
      this.#queryParams,
      0,
      bufferSource(new Uint32Array([position, thetaBits])),
    );
    this.#device.queue.writeBuffer(
      this.#kvParams,
      0,
      bufferSource(
        new Uint32Array([
          layerIndex,
          position,
          this.#kvAllocation,
          this.#sinkLength,
          geometry.kvWindow,
          geometry.numberOfKVHeads,
          thetaBits,
          0,
        ]),
      ),
    );
    this.#device.queue.writeBuffer(
      this.#attentionParams,
      0,
      bufferSource(
        new Uint32Array([
          layerIndex,
          position,
          this.#kvAllocation,
          this.#sinkLength,
          geometry.kvWindow,
          geometry.numberOfKVHeads,
          geometry.numberOfHeads,
          0,
        ]),
      ),
    );

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
      this.#nextLanes,
      this.#finalNorm,
      this.#finalHidden,
      this.#preparedFinal,
      this.#logits,
      this.#allowedTokens,
      this.#selectionParams,
      this.#selectionResult,
      this.#staging,
      this.#queryParams,
      this.#kvParams,
      this.#attentionParams,
      this.#layerParams,
      ...this.#projectionParams,
    ]) {
      buffer.destroy();
    }
    for (const buffer of [
      this.#confidenceProbes,
      this.#confidenceProjection,
      this.#confidenceBias,
      this.#confidenceMaxima,
      this.#confidenceDenominators,
      this.#confidenceWeighted,
      this.#confidenceResult,
    ]) {
      buffer?.destroy();
    }
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
    this.#device.queue.writeBuffer(
      this.#queryParams,
      0,
      bufferSource(new Uint32Array([position, thetaBits])),
    );
    this.#device.queue.writeBuffer(
      this.#kvParams,
      0,
      bufferSource(
        new Uint32Array([
          layerIndex,
          position,
          this.#kvAllocation,
          this.#sinkLength,
          geometry.kvWindow,
          geometry.numberOfKVHeads,
          thetaBits,
          0,
        ]),
      ),
    );
    this.#device.queue.writeBuffer(
      this.#attentionParams,
      0,
      bufferSource(
        new Uint32Array([
          layerIndex,
          position,
          this.#kvAllocation,
          this.#sinkLength,
          geometry.kvWindow,
          geometry.numberOfKVHeads,
          geometry.numberOfHeads,
          0,
        ]),
      ),
    );
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

  #getConfidenceInputGroup(pipeline: GPUComputePipeline): GPUBindGroup {
    if (this.#confidenceInputGroup) return this.#confidenceInputGroup;
    this.#confidenceInputGroup = this.#createConfidencePoolGroup(this.#lanesInput, pipeline);
    return this.#confidenceInputGroup;
  }

  #getConfidenceNextGroup(pipeline: GPUComputePipeline): GPUBindGroup {
    if (this.#confidenceNextGroup) return this.#confidenceNextGroup;
    this.#confidenceNextGroup = this.#createConfidencePoolGroup(this.#nextLanes, pipeline);
    return this.#confidenceNextGroup;
  }

  #createConfidencePoolGroup(source: GPUBuffer, pipeline: GPUComputePipeline): GPUBindGroup {
    const probes = this.#confidenceProbes;
    const maxima = this.#confidenceMaxima;
    const denominators = this.#confidenceDenominators;
    const weighted = this.#confidenceWeighted;
    invariant(
      probes && maxima && denominators && weighted,
      "BACKEND_UNAVAILABLE",
      "Resident confidence buffers are missing",
    );
    return this.#device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: source } },
        { binding: 1, resource: { buffer: probes } },
        { binding: 2, resource: { buffer: maxima } },
        { binding: 3, resource: { buffer: denominators } },
        { binding: 4, resource: { buffer: weighted } },
      ],
    });
  }

  #getConfidenceHeadGroup(pipeline: GPUComputePipeline): GPUBindGroup {
    if (this.#confidenceHeadGroup) return this.#confidenceHeadGroup;
    const denominators = this.#confidenceDenominators;
    const weighted = this.#confidenceWeighted;
    const projection = this.#confidenceProjection;
    const bias = this.#confidenceBias;
    const result = this.#confidenceResult;
    invariant(
      denominators && weighted && projection && bias && result,
      "BACKEND_UNAVAILABLE",
      "Resident confidence head is missing",
    );
    this.#confidenceHeadGroup = this.#device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: denominators } },
        { binding: 1, resource: { buffer: weighted } },
        { binding: 2, resource: { buffer: projection } },
        { binding: 3, resource: { buffer: bias } },
        { binding: 4, resource: { buffer: result } },
      ],
    });
    return this.#confidenceHeadGroup;
  }

  #getSelectionGroup(pipeline: GPUComputePipeline): GPUBindGroup {
    if (this.#selectionGroup) return this.#selectionGroup;
    this.#selectionGroup = this.#device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.#logits } },
        { binding: 1, resource: { buffer: this.#allowedTokens } },
        { binding: 2, resource: { buffer: this.#selectionParams } },
        { binding: 3, resource: { buffer: this.#selectionResult } },
      ],
    });
    return this.#selectionGroup;
  }

  #queryGroup(layer: CactLayer, pipeline: GPUComputePipeline): GPUBindGroup {
    const cached = this.#queryGroups.get(layer);
    if (cached) return cached;
    const group = this.#device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.#query } },
        { binding: 1, resource: { buffer: this.#norms(layer).query } },
        { binding: 2, resource: { buffer: this.#queryParams } },
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
        { binding: 7, resource: { buffer: this.#kvParams } },
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
        { binding: 3, resource: { buffer: this.#attentionParams } },
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
        { binding: 3, resource: { buffer: this.#attentionParams } },
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

async function createPipelines(device: GPUDevice): Promise<Pipelines> {
  const descriptors = [
    ["cq", CQ_MATVEC_WGSL, "main"],
    ["query", QUERY_NORM_ROPE_WGSL, "query_norm_rope"],
    ["kv", KV_NORM_ROPE_STORE_WGSL, "kv_norm_rope_store"],
    ["scores", ATTENTION_SCORES_WGSL, "attention_scores"],
    ["attention", ATTENTION_SOFTMAX_GATE_WGSL, "attention_softmax_gate"],
    ["prepare", PREPARE_ATTENTION_WGSL, "prepare_attention"],
    ["sandwich", SANDWICH_PREPARE_MLP_WGSL, "sandwich_prepare_mlp"],
    ["mlp", HADAMARD_MLP_DELTA_WGSL, "hadamard_mlp_delta"],
    ["routing", POST_MHC_ROUTING_WGSL, "post_mhc_routing"],
    ["rms-lanes", RMS_LANES_WGSL, "rms_lanes"],
    ["mhc-pre", MHC_PRE_WGSL, "mhc_pre"],
    ["engram", ENGRAM_INJECT_WGSL, "engram_inject"],
    ["norm-512", RMS_NORM_512_WGSL, "rms_norm_512"],
    ["final-norm", FINAL_NORM_WGSL, "final_norm"],
    ["confidence-pool", CONFIDENCE_POOL_WGSL, "confidence_pool"],
    ["confidence-head", CONFIDENCE_HEAD_WGSL, "confidence_head"],
    ["select-token", SELECT_TOKEN_WGSL, "select_token"],
  ] as const;
  const pipelines = await Promise.all(
    descriptors.map(async ([label, source, entryPoint]) => {
      const module = device.createShaderModule({
        label: `needle.attention.${label}.shader`,
        code: source,
      });
      const descriptor: GPUComputePipelineDescriptor = {
        label: `needle.attention.${label}.pipeline`,
        layout: "auto",
        compute: { module, entryPoint },
      };
      return device.createComputePipelineAsync
        ? device.createComputePipelineAsync(descriptor)
        : device.createComputePipeline(descriptor);
    }),
  );
  const [
    cq,
    query,
    kv,
    scores,
    attention,
    prepare,
    sandwich,
    mlp,
    routing,
    rmsLanes,
    mhcPre,
    engram,
    norm512,
    finalNorm,
    confidencePool,
    confidenceHead,
    selectToken,
  ] = pipelines;
  invariant(
    cq &&
      query &&
      kv &&
      scores &&
      attention &&
      prepare &&
      sandwich &&
      mlp &&
      routing &&
      rmsLanes &&
      mhcPre &&
      engram &&
      norm512 &&
      finalNorm &&
      confidencePool &&
      confidenceHead &&
      selectToken,
    "WEBGPU_UNAVAILABLE",
    "Attention pipeline failed",
  );
  return {
    cq,
    query,
    kv,
    scores,
    attention,
    prepare,
    sandwich,
    mlp,
    routing,
    rmsLanes,
    mhcPre,
    engram,
    norm512,
    finalNorm,
    confidencePool,
    confidenceHead,
    selectToken,
  };
}

function storageBuffer(device: GPUDevice, label: string, size: number, extraUsage = 0): GPUBuffer {
  return device.createBuffer({ label, size: align4(size), usage: STORAGE | extraUsage });
}

function bufferWithData(
  device: GPUDevice,
  label: string,
  data: ArrayBufferView,
  extraUsage = 0,
): GPUBuffer {
  const buffer = storageBuffer(device, label, data.byteLength, COPY_DST | extraUsage);
  device.queue.writeBuffer(buffer, 0, bufferSource(data));
  return buffer;
}

function floatBits(value: number): number {
  const view = new DataView(new ArrayBuffer(4));
  view.setFloat32(0, value, true);
  return view.getUint32(0, true);
}

function align4(value: number): number {
  return Math.max(4, (value + 3) & ~3);
}

function bufferSource(data: ArrayBufferView): GPUAllowSharedBufferSource {
  if (data.buffer instanceof ArrayBuffer) return data as ArrayBufferView<ArrayBuffer>;
  return new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice();
}
