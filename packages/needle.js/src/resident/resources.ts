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
}

export function createRawResourceFactory(device: GPUDevice): ResidentResourceFactory {
  return {
    create(_scalar, label, elementCount, extraUsage = 0) {
      return storageBuffer(device, label, elementCount * 4, extraUsage);
    },
  };
}
