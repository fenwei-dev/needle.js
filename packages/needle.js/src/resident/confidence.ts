/// <reference types="@webgpu/types" preserve="true" />

import { invariant } from "../errors.js";
import type { CactWeights } from "../model/cact.js";
import type { ResidentBindingFactory } from "./bindings.js";
import { supportsResidentConfidence } from "./compatibility.js";
import type { ResidentPipelines } from "./pipelines.js";
import { createRawResourceFactory, type ResidentResourceFactory } from "./resources.js";
import { bufferSource, COPY_DST, COPY_SRC, MAP_READ } from "./webgpu.js";

export class ResidentConfidence {
  readonly #device: GPUDevice;
  readonly #resources: ResidentResourceFactory;
  readonly #bindings: ResidentBindingFactory;
  readonly #probes: GPUBuffer | undefined;
  readonly #projection: GPUBuffer | undefined;
  readonly #bias: GPUBuffer | undefined;
  readonly #maxima: GPUBuffer | undefined;
  readonly #denominators: GPUBuffer | undefined;
  readonly #weighted: GPUBuffer | undefined;
  readonly #result: GPUBuffer | undefined;
  readonly #staging: GPUBuffer | undefined;
  #groups = new WeakMap<GPUBuffer, GPUBindGroup>();
  #headGroup: GPUBindGroup | undefined;

  constructor(
    device: GPUDevice,
    weights: CactWeights,
    resources: ResidentResourceFactory = createRawResourceFactory(device),
    bindings: ResidentBindingFactory = {},
  ) {
    this.#device = device;
    this.#resources = resources;
    this.#bindings = bindings;
    const head = weights.heads.get("confidence");
    const supported = supportsResidentConfidence(weights) && head !== undefined;
    const create = (label: string, values: Float32Array, extraUsage = 0) => {
      const buffer = resources.create("f32", label, values.length, COPY_DST | extraUsage);
      device.queue.writeBuffer(buffer, 0, bufferSource(values));
      return buffer;
    };
    this.#probes = supported ? create("needle.confidence.probes", head.probes) : undefined;
    this.#projection = supported
      ? create("needle.confidence.projection", head.projection)
      : undefined;
    this.#bias = supported ? create("needle.confidence.bias", head.bias) : undefined;
    this.#maxima = supported
      ? resources.create("f32", "needle.confidence.maxima", 8, COPY_DST)
      : undefined;
    this.#denominators = supported
      ? resources.create("f32", "needle.confidence.denominators", 8, COPY_DST)
      : undefined;
    this.#weighted = supported
      ? resources.create("f32", "needle.confidence.weighted", 4096, COPY_DST)
      : undefined;
    this.#result = supported
      ? resources.create("f32", "needle.confidence.result", 1, COPY_SRC)
      : undefined;
    this.#staging = supported
      ? device.createBuffer({
          label: "needle.confidence.readback",
          size: 4,
          usage: MAP_READ | COPY_DST,
        })
      : undefined;
  }

  get available(): boolean {
    return Boolean(
      this.#probes &&
        this.#projection &&
        this.#bias &&
        this.#maxima &&
        this.#denominators &&
        this.#weighted &&
        this.#result &&
        this.#staging,
    );
  }

  reset(): void {
    if (!this.available) return;
    const maxima = new Float32Array(8);
    maxima.fill(Number.NEGATIVE_INFINITY);
    this.#device.queue.writeBuffer(this.#maxima as GPUBuffer, 0, bufferSource(maxima));
    this.#device.queue.writeBuffer(
      this.#denominators as GPUBuffer,
      0,
      bufferSource(new Float32Array(8)),
    );
    this.#device.queue.writeBuffer(
      this.#weighted as GPUBuffer,
      0,
      bufferSource(new Float32Array(4096)),
    );
  }

  encodePool(
    encoder: GPUCommandEncoder,
    pipeline: ResidentPipelines["confidencePool"],
    source: GPUBuffer,
    label: string,
  ): void {
    invariant(this.available, "BACKEND_UNAVAILABLE", "Resident confidence is unavailable");
    const pass = encoder.beginComputePass({ label });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, this.#getPoolGroup(source, pipeline));
    pass.dispatchWorkgroups(1);
    pass.end();
  }

  async resolve(pipeline: ResidentPipelines["confidenceHead"]): Promise<number> {
    const result = this.#result;
    const staging = this.#staging;
    invariant(result && staging, "BACKEND_UNAVAILABLE", "Resident confidence output is missing");
    const encoder = this.#device.createCommandEncoder({ label: "needle.confidence.final" });
    const pass = encoder.beginComputePass({ label: "needle.confidence.head" });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, this.#getHeadGroup(pipeline));
    pass.dispatchWorkgroups(1);
    pass.end();
    encoder.copyBufferToBuffer(result, 0, staging, 0, 4);
    this.#device.queue.submit([encoder.finish()]);
    await staging.mapAsync(MAP_READ, 0, 4);
    try {
      return new Float32Array(staging.getMappedRange(0, 4))[0] ?? 0;
    } finally {
      staging.unmap();
    }
  }

  dispose(): void {
    for (const buffer of [
      this.#probes,
      this.#projection,
      this.#bias,
      this.#maxima,
      this.#denominators,
      this.#weighted,
      this.#result,
    ]) {
      if (buffer && !this.#resources.destroy(buffer)) buffer.destroy();
    }
    this.#staging?.destroy();
  }

  #getPoolGroup(source: GPUBuffer, pipeline: GPUComputePipeline): GPUBindGroup {
    const cached = this.#groups.get(source);
    if (cached) return cached;
    const probes = this.#probes;
    const maxima = this.#maxima;
    const denominators = this.#denominators;
    const weighted = this.#weighted;
    invariant(
      probes && maxima && denominators && weighted,
      "BACKEND_UNAVAILABLE",
      "Resident confidence buffers are missing",
    );
    const resources = { lanes: source, probes, maxima, denominators, weighted };
    const group =
      this.#bindings.createConfidencePool?.(resources) ??
      this.#device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: resources.lanes } },
          { binding: 1, resource: { buffer: resources.probes } },
          { binding: 2, resource: { buffer: resources.maxima } },
          { binding: 3, resource: { buffer: resources.denominators } },
          { binding: 4, resource: { buffer: resources.weighted } },
        ],
      });
    this.#groups.set(source, group);
    return group;
  }

  #getHeadGroup(pipeline: GPUComputePipeline): GPUBindGroup {
    if (this.#headGroup) return this.#headGroup;
    const denominators = this.#denominators;
    const weighted = this.#weighted;
    const projection = this.#projection;
    const bias = this.#bias;
    const result = this.#result;
    invariant(
      denominators && weighted && projection && bias && result,
      "BACKEND_UNAVAILABLE",
      "Resident confidence head is missing",
    );
    const resources = { denominators, weighted, projection, bias, result };
    this.#headGroup =
      this.#bindings.createConfidenceHead?.(resources) ??
      this.#device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: resources.denominators } },
          { binding: 1, resource: { buffer: resources.weighted } },
          { binding: 2, resource: { buffer: resources.projection } },
          { binding: 3, resource: { buffer: resources.bias } },
          { binding: 4, resource: { buffer: resources.result } },
        ],
      });
    return this.#headGroup;
  }
}
