/// <reference types="@webgpu/types" preserve="true" />

import type { ResidentEngramRequest } from "../backends/backend.js";
import { matrixParameters } from "../backends/webgpu-kernel.js";
import { invariant } from "../errors.js";
import type { CactWeights, CqMatrix } from "../model/cact.js";
import type { ResidentPipelines } from "./pipelines.js";
import type { ResidentResourceFactory } from "./resources.js";
import { bufferSource, COPY_DST, STORAGE } from "./webgpu.js";

export interface ResidentGpuMatrix {
  readonly packed: GPUBuffer;
  readonly norms: GPUBuffer;
}

export interface ResidentEngramGpuAccess {
  readonly codebook: GPUBuffer;
  readonly updateInput: GPUBuffer;
  readonly blockInput: GPUBuffer;
  matrix(matrix: CqMatrix): ResidentGpuMatrix;
  projectionGroup(
    matrix: CqMatrix,
    input: GPUBuffer,
    params: GPUBuffer,
    output: GPUBuffer,
    pipeline: GPUComputePipeline,
  ): GPUBindGroup;
}

interface EngramGroups {
  readonly gather: GPUBindGroup;
  readonly prepare: GPUBindGroup;
  readonly key: GPUBindGroup;
  readonly value: GPUBindGroup;
  readonly convolve: GPUBindGroup;
  readonly inject: GPUBindGroup;
}

export class ResidentEngramState {
  readonly #device: GPUDevice;
  readonly #weights: CactWeights;
  readonly #resources: ResidentResourceFactory;
  readonly #gpu: ResidentEngramGpuAccess;
  readonly #owned: GPUBuffer[] = [];
  readonly #rowIds: GPUBuffer;
  readonly #rowValid: GPUBuffer;
  readonly #concatenated: GPUBuffer[] = [];
  readonly #prepared: GPUBuffer[] = [];
  readonly #keys: GPUBuffer[] = [];
  readonly #valueNow: GPUBuffer[] = [];
  readonly #mixed: GPUBuffer[] = [];
  readonly #convolutionParams: GPUBuffer[] = [];
  readonly #keyParams: GPUBuffer[] = [];
  readonly #valueParams: GPUBuffer[] = [];
  readonly #ring: GPUBuffer;
  readonly #ringValid: GPUBuffer;
  #groups: EngramGroups[] = [];
  #readyPosition = -1;

  constructor(
    device: GPUDevice,
    weights: CactWeights,
    resources: ResidentResourceFactory,
    gpu: ResidentEngramGpuAccess,
  ) {
    this.#device = device;
    this.#weights = weights;
    this.#resources = resources;
    this.#gpu = gpu;
    const geometry = weights.geometry;
    const dimension = geometry.modelDimension;
    const maximumOrder = Math.max(1, ...geometry.engramOrders);
    const depth = (geometry.engramConvolutionTaps - 1) * maximumOrder + 1;
    this.#rowIds = this.#create("u32", "needle.engram.row-ids", 4, COPY_DST);
    this.#rowValid = this.#create("u32", "needle.engram.row-valid", 4, COPY_DST);
    this.#ring = this.#create(
      "f32",
      "needle.engram.ring",
      Math.max(1, geometry.engramLayers.length * depth * dimension),
      COPY_DST,
    );
    this.#ringValid = this.#create(
      "u32",
      "needle.engram.ring-valid",
      Math.max(1, geometry.engramLayers.length * depth),
      COPY_DST,
    );
    for (let site = 0; site < weights.engrams.length; site++) {
      const engram = weights.engrams[site];
      invariant(engram, "INVALID_CACT", `Missing engram site ${site}`);
      this.#concatenated.push(this.#create("f32", `needle.engram.${site}.concatenated`, dimension));
      this.#prepared.push(this.#create("f32", `needle.engram.${site}.prepared`, dimension));
      this.#keys.push(this.#create("f32", `needle.engram.${site}.key`, dimension));
      this.#valueNow.push(this.#create("f32", `needle.engram.${site}.value-now`, dimension));
      this.#mixed.push(this.#create("f32", `needle.engram.${site}.mixed`, dimension));
      this.#convolutionParams.push(
        this.#create("u32", `needle.engram.${site}.convolution-params`, 4, COPY_DST),
      );
      this.#keyParams.push(
        this.#createWithData(
          "u32",
          `needle.engram.${site}.key-params`,
          matrixParameters(engram.keyProjection, 0, engram.keyProjection.outputSize),
        ),
      );
      this.#valueParams.push(
        this.#createWithData(
          "u32",
          `needle.engram.${site}.value-params`,
          matrixParameters(engram.valueProjection, 0, engram.valueProjection.outputSize),
        ),
      );
    }
  }

  reset(): void {
    const geometry = this.#weights.geometry;
    const maximumOrder = Math.max(1, ...geometry.engramOrders);
    const depth = (geometry.engramConvolutionTaps - 1) * maximumOrder + 1;
    this.#device.queue.writeBuffer(
      this.#ringValid,
      0,
      bufferSource(new Uint32Array(geometry.engramLayers.length * depth)),
    );
    this.#readyPosition = -1;
  }

  async prepare(request: ResidentEngramRequest, pipelines: ResidentPipelines): Promise<boolean> {
    if (this.#weights.engrams.length === 0) return false;
    invariant(
      request.indices.length === 4 && request.valid.length === 4,
      "INVALID_CACT",
      "Resident engram lookup requires four row IDs and validity flags",
    );
    this.#device.queue.writeBuffer(this.#rowIds, 0, bufferSource(request.indices));
    this.#device.queue.writeBuffer(this.#rowValid, 0, bufferSource(request.valid));
    const groups = this.#ensureGroups(pipelines);
    const geometry = this.#weights.geometry;
    const maximumOrder = Math.max(1, ...geometry.engramOrders);
    const depth = (geometry.engramConvolutionTaps - 1) * maximumOrder + 1;
    const encoder = this.#device.createCommandEncoder({ label: "needle.engram.resident" });
    for (let site = 0; site < groups.length; site++) {
      const group = groups[site];
      const params = this.#convolutionParams[site];
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
    this.#readyPosition = request.position;
    return true;
  }

  injectGroup(site: number, position?: number): GPUBindGroup | undefined {
    if (position !== undefined && this.#readyPosition !== position) return undefined;
    return this.#groups[site]?.inject;
  }

  dispose(): void {
    for (const buffer of this.#owned) {
      if (!this.#resources.destroy(buffer)) buffer.destroy();
    }
  }

  #ensureGroups(pipelines: ResidentPipelines): readonly EngramGroups[] {
    if (this.#groups.length === this.#weights.engrams.length) return this.#groups;
    this.#groups = [];
    for (let site = 0; site < this.#weights.engrams.length; site++) {
      const engram = this.#weights.engrams[site];
      const concatenated = this.#concatenated[site];
      const prepared = this.#prepared[site];
      const keyOutput = this.#keys[site];
      const valueOutput = this.#valueNow[site];
      const mixed = this.#mixed[site];
      const convolutionParams = this.#convolutionParams[site];
      const keyParams = this.#keyParams[site];
      const valueParams = this.#valueParams[site];
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
      const table = this.#gpu.matrix(engram.tables);
      const gather = this.#device.createBindGroup({
        layout: pipelines.engramGather.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: table.packed } },
          { binding: 1, resource: { buffer: table.norms } },
          { binding: 2, resource: { buffer: this.#gpu.codebook } },
          { binding: 3, resource: { buffer: this.#rowIds } },
          { binding: 4, resource: { buffer: this.#rowValid } },
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
      const key = this.#gpu.projectionGroup(
        engram.keyProjection,
        prepared,
        keyParams,
        keyOutput,
        pipelines.cq,
      );
      const value = this.#gpu.projectionGroup(
        engram.valueProjection,
        prepared,
        valueParams,
        valueOutput,
        pipelines.cq,
      );
      const taps = this.#createWithData("f32", `needle.engram.${site}.taps`, engram.taps);
      const convolve = this.#device.createBindGroup({
        layout: pipelines.engramConvolve.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: valueOutput } },
          { binding: 1, resource: { buffer: this.#ring } },
          { binding: 2, resource: { buffer: taps } },
          { binding: 3, resource: { buffer: this.#ringValid } },
          { binding: 4, resource: { buffer: convolutionParams } },
          { binding: 5, resource: { buffer: mixed } },
        ],
      });
      const inject = this.#device.createBindGroup({
        layout: pipelines.engram.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.#gpu.updateInput } },
          { binding: 1, resource: { buffer: keyOutput } },
          { binding: 2, resource: { buffer: mixed } },
          { binding: 3, resource: { buffer: this.#gpu.blockInput } },
        ],
      });
      this.#groups.push({ gather, prepare, key, value, convolve, inject });
    }
    return this.#groups;
  }

  #create(scalar: "f32" | "u32", label: string, elementCount: number, extraUsage = 0): GPUBuffer {
    const buffer = this.#resources.create(scalar, label, elementCount, extraUsage);
    this.#owned.push(buffer);
    return buffer;
  }

  #createWithData(
    scalar: "f32" | "u32",
    label: string,
    data: Float32Array | Uint32Array,
  ): GPUBuffer {
    const buffer = this.#create(scalar, label, data.length, COPY_DST | STORAGE);
    this.#device.queue.writeBuffer(buffer, 0, bufferSource(data));
    return buffer;
  }
}
