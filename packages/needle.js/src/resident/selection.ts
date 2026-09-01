/// <reference types="@webgpu/types" preserve="true" />

import type { ResidentTokenSelection } from "../backends/backend.js";
import { invariant } from "../errors.js";
import type { ResidentPipelines } from "./pipelines.js";
import { bufferSource, COPY_DST, COPY_SRC, MAP_READ, storageBuffer } from "./webgpu.js";

export class ResidentTokenSelector {
  readonly #device: GPUDevice;
  readonly #vocabularySize: number;
  readonly #logits: GPUBuffer;
  readonly #allowed: GPUBuffer;
  readonly #params: GPUBuffer;
  readonly #result: GPUBuffer;
  readonly #staging: GPUBuffer;
  #group: GPUBindGroup | undefined;

  constructor(device: GPUDevice, logits: GPUBuffer, vocabularySize: number) {
    this.#device = device;
    this.#vocabularySize = vocabularySize;
    this.#logits = logits;
    this.#allowed = storageBuffer(device, "needle.selection.allowed", vocabularySize * 4, COPY_DST);
    this.#params = storageBuffer(device, "needle.selection.params", 8, COPY_DST);
    this.#result = storageBuffer(device, "needle.selection.result", 8, COPY_SRC);
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
    this.#allowed.destroy();
    this.#params.destroy();
    this.#result.destroy();
    this.#staging.destroy();
  }

  #getGroup(pipeline: GPUComputePipeline): GPUBindGroup {
    if (this.#group) return this.#group;
    this.#group = this.#device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.#logits } },
        { binding: 1, resource: { buffer: this.#allowed } },
        { binding: 2, resource: { buffer: this.#params } },
        { binding: 3, resource: { buffer: this.#result } },
      ],
    });
    return this.#group;
  }
}
