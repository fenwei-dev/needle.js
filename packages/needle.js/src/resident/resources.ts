/// <reference types="@webgpu/types" preserve="true" />

import { storageBuffer } from "./webgpu.js";

export type ResidentScalar = "f32" | "u32" | "i32";

export interface ResidentResourceFactory {
  create(
    scalar: ResidentScalar,
    label: string,
    elementCount: number,
    extraUsage?: GPUBufferUsageFlags,
  ): GPUBuffer;
  /** Returns true when this factory owned and released the buffer. */
  destroy(buffer: GPUBuffer): boolean;
}

export function createRawResourceFactory(device: GPUDevice): ResidentResourceFactory {
  const owned = new WeakSet<GPUBuffer>();
  return {
    create(_scalar, label, elementCount, extraUsage = 0) {
      const buffer = storageBuffer(device, label, elementCount * 4, extraUsage);
      owned.add(buffer);
      return buffer;
    },
    destroy(buffer) {
      if (!owned.has(buffer)) return false;
      owned.delete(buffer);
      buffer.destroy();
      return true;
    },
  };
}
