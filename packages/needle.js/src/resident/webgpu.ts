/// <reference types="@webgpu/types" preserve="true" />

export const MAP_READ = 0x0001;
export const COPY_SRC = 0x0004;
export const COPY_DST = 0x0008;
export const STORAGE = 0x0080;

export function storageBuffer(
  device: GPUDevice,
  label: string,
  size: number,
  extraUsage = 0,
): GPUBuffer {
  return device.createBuffer({ label, size: align4(size), usage: STORAGE | extraUsage });
}

export function bufferWithData(
  device: GPUDevice,
  label: string,
  data: ArrayBufferView,
  extraUsage = 0,
): GPUBuffer {
  const buffer = storageBuffer(device, label, data.byteLength, COPY_DST | extraUsage);
  device.queue.writeBuffer(buffer, 0, bufferSource(data));
  return buffer;
}

export function floatBits(value: number): number {
  const view = new DataView(new ArrayBuffer(4));
  view.setFloat32(0, value, true);
  return view.getUint32(0, true);
}

export function align4(value: number): number {
  return Math.max(4, (value + 3) & ~3);
}

export function bufferSource(data: ArrayBufferView): GPUAllowSharedBufferSource {
  if (data.buffer instanceof ArrayBuffer) return data as ArrayBufferView<ArrayBuffer>;
  return new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice();
}
