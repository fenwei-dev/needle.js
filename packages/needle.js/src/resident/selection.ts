/// <reference types="@webgpu/types" preserve="true" />

import type { ResidentTokenSelection } from "../backends/backend.js";
import { invariant } from "../errors.js";
import type { ResidentBindingFactory } from "./bindings.js";
import type { ResidentPipelines } from "./pipelines.js";
import { createRawResourceFactory, type ResidentResourceFactory } from "./resources.js";
import { bufferSource, COPY_DST, COPY_SRC, MAP_READ } from "./webgpu.js";

export class ResidentTokenSelector {
  readonly #device: GPUDevice;
  readonly #vocabularySize: number;
  readonly #resources: ResidentResourceFactory;
  readonly #bindings: ResidentBindingFactory;
  readonly #logits: GPUBuffer;
  readonly #allowed: GPUBuffer;
  readonly #params: GPUBuffer;
  readonly #result: GPUBuffer;
  readonly #staging: GPUBuffer;
  #group: GPUBindGroup | undefined;

  constructor(
    device: GPUDevice,
    logits: GPUBuffer,
    vocabularySize: number,
    resources: ResidentResourceFactory = createRawResourceFactory(device),
    bindings: ResidentBindingFactory = {},
  ) {
    this.#device = device;
    this.#vocabularySize = vocabularySize;
    this.#resources = resources;
    this.#bindings = bindings;
    this.#logits = logits;
    this.#allowed = resources.create("u32", "needle.selection.allowed", vocabularySize, COPY_DST);
    this.#params = resources.create("u32", "needle.selection.params", 2, COPY_DST);
    this.#result = resources.create("u32", "needle.selection.result", 2, COPY_SRC);
    this.#staging = device.createBuffer({
      label: "needle.selection.readback",
      size: 8,
      usage: MAP_READ | COPY_DST,
    });
  }

  async select(
    pipeline: ResidentPipelines["selectToken"],
    allowedTokenIds?: Uint32Array,
  ): Promise<ResidentTokenSelection> {
    const count = allowedTokenIds?.length ?? this.#vocabularySize;
    invariant(
      count > 0 && count <= this.#vocabularySize,
      "INVALID_CACT",
      "Allowed token set is empty or too large",
    );
    if (allowedTokenIds) {
      this.#device.queue.writeBuffer(this.#allowed, 0, bufferSource(allowedTokenIds));
    }
    this.#device.queue.writeBuffer(
      this.#params,
      0,
      bufferSource(new Uint32Array([count, allowedTokenIds ? 1 : 0])),
    );
    const encoder = this.#device.createCommandEncoder({ label: "needle.selection.encoder" });
    const pass = encoder.beginComputePass({ label: "needle.selection.argmax" });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, this.#getGroup(pipeline));
    pass.dispatchWorkgroups(1);
    pass.end();
    encoder.copyBufferToBuffer(this.#result, 0, this.#staging, 0, 8);
    this.#device.queue.submit([encoder.finish()]);
    await this.#staging.mapAsync(MAP_READ, 0, 8);
    try {
      const view = new DataView(this.#staging.getMappedRange(0, 8));
      return { id: view.getUint32(0, true), logProbability: view.getFloat32(4, true) };
    } finally {
      this.#staging.unmap();
    }
  }

  dispose(): void {
    for (const buffer of [this.#allowed, this.#params, this.#result]) {
      if (!this.#resources.destroy(buffer)) buffer.destroy();
    }
    this.#staging.destroy();
  }

  #getGroup(pipeline: GPUComputePipeline): GPUBindGroup {
    if (this.#group) return this.#group;
    const resources = {
      logits: this.#logits,
      allowed: this.#allowed,
      params: this.#params,
      result: this.#result,
    };
    this.#group =
      this.#bindings.createSelection?.(resources) ??
      this.#device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: resources.logits } },
          { binding: 1, resource: { buffer: resources.allowed } },
          { binding: 2, resource: { buffer: resources.params } },
          { binding: 3, resource: { buffer: resources.result } },
        ],
      });
    return this.#group;
  }
}
