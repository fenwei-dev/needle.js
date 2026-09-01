/// <reference types="@webgpu/types" preserve="true" />

import { invariant } from "../errors.js";
import type { CactLayer, CactWeights, CqMatrix } from "../model/cact.js";
import type {
  FusedAttentionRequest,
  FusedAttentionResetOptions,
  FusedAttentionSession,
} from "./backend.js";
import { prepareCqActivation } from "./cq.js";
import {
  ATTENTION_SCORES_WGSL,
  ATTENTION_SOFTMAX_GATE_WGSL,
  KV_NORM_ROPE_STORE_WGSL,
  PREPARE_ATTENTION_WGSL,
  QUERY_NORM_ROPE_WGSL,
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
}

interface Pipelines {
  readonly cq: GPUComputePipeline;
  readonly query: GPUComputePipeline;
  readonly kv: GPUComputePipeline;
  readonly scores: GPUComputePipeline;
  readonly attention: GPUComputePipeline;
  readonly prepare: GPUComputePipeline;
}

export class TypeGPUFusedAttentionSession implements FusedAttentionSession {
  readonly #device: GPUDevice;
  readonly #weights: CactWeights;
  readonly #codebook: GPUBuffer;
  readonly #preparedInput: GPUBuffer;
  readonly #query: GPUBuffer;
  readonly #key: GPUBuffer;
  readonly #value: GPUBuffer;
  readonly #gate: GPUBuffer;
  readonly #attentionOutput: GPUBuffer;
  readonly #scores: GPUBuffer;
  readonly #preparedAttention: GPUBuffer;
  readonly #projected: GPUBuffer;
  readonly #staging: GPUBuffer;
  readonly #queryParams: GPUBuffer;
  readonly #kvParams: GPUBuffer;
  readonly #attentionParams: GPUBuffer;
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
  #kvAllocation = 0;
  #sinkLength = 0;
  #enabled = false;
  #disposed = false;

  constructor(device: GPUDevice, weights: CactWeights) {
    this.#device = device;
    this.#weights = weights;
    const geometry = weights.geometry;
    const dimensionBytes = geometry.modelDimension * 4;
    const keyValueBytes = geometry.numberOfKVHeads * geometry.headDimension * 4;
    this.#codebook = bufferWithData(device, "needle.attention.codebook", weights.codebook, STORAGE);
    this.#preparedInput = storageBuffer(device, "needle.attention.input", dimensionBytes, COPY_DST);
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
    this.#staging = device.createBuffer({
      label: "needle.attention.readback",
      size: dimensionBytes,
      usage: MAP_READ | COPY_DST,
    });
    this.#queryParams = storageBuffer(device, "needle.attention.query-params", 8, COPY_DST);
    this.#kvParams = storageBuffer(device, "needle.attention.kv-params", 32, COPY_DST);
    this.#attentionParams = storageBuffer(device, "needle.attention.params", 32, COPY_DST);
    this.#projectionParams = Array.from({ length: 5 }, (_, index) =>
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
          layer.outputProjection.groupSize === 128,
      ) &&
      options.sinkLength + geometry.kvWindow <= 512;
    if (!this.#enabled) return false;

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

  async forward(request: FusedAttentionRequest): Promise<Float32Array> {
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
    const { layer, layerIndex, position, input } = request;
    const geometry = this.#weights.geometry;
    const pipelines = await this.#pipelines;
    const prepared = prepareCqActivation(layer.queryProjection, input);
    this.#device.queue.writeBuffer(this.#preparedInput, 0, bufferSource(prepared));

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
    encoder.copyBufferToBuffer(this.#projected, 0, this.#staging, 0, geometry.modelDimension * 4);
    this.#device.queue.submit([encoder.finish()]);
    await this.#staging.mapAsync(MAP_READ, 0, geometry.modelDimension * 4);
    try {
      const mapped = this.#staging.getMappedRange(0, geometry.modelDimension * 4);
      return new Float32Array(mapped).slice();
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
      this.#query,
      this.#key,
      this.#value,
      this.#gate,
      this.#attentionOutput,
      this.#scores,
      this.#preparedAttention,
      this.#projected,
      this.#staging,
      this.#queryParams,
      this.#kvParams,
      this.#attentionParams,
      ...this.#projectionParams,
    ]) {
      buffer.destroy();
    }
    for (const matrix of this.#allocatedMatrices) {
      matrix.packed.destroy();
      matrix.norms.destroy();
    }
    for (const norms of this.#allocatedNorms) {
      norms.query.destroy();
      norms.key.destroy();
    }
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
    };
    this.#layerNorms.set(layer, result);
    this.#allocatedNorms.push(result);
    return result;
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
}

async function createPipelines(device: GPUDevice): Promise<Pipelines> {
  const descriptors = [
    ["cq", CQ_MATVEC_WGSL, "main"],
    ["query", QUERY_NORM_ROPE_WGSL, "query_norm_rope"],
    ["kv", KV_NORM_ROPE_STORE_WGSL, "kv_norm_rope_store"],
    ["scores", ATTENTION_SCORES_WGSL, "attention_scores"],
    ["attention", ATTENTION_SOFTMAX_GATE_WGSL, "attention_softmax_gate"],
    ["prepare", PREPARE_ATTENTION_WGSL, "prepare_attention"],
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
  const [cq, query, kv, scores, attention, prepare] = pipelines;
  invariant(
    cq && query && kv && scores && attention && prepare,
    "WEBGPU_UNAVAILABLE",
    "Attention pipeline failed",
  );
  return { cq, query, kv, scores, attention, prepare };
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
