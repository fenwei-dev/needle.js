/// <reference types="@webgpu/types" preserve="true" />

import { bufferSource, COPY_DST, storageBuffer } from "./webgpu.js";

export interface QueryParameters {
  readonly position: number;
  readonly thetaBits: number;
}

export interface KvParameters {
  readonly layer: number;
  readonly position: number;
  readonly allocation: number;
  readonly sinkLength: number;
  readonly window: number;
  readonly kvHeads: number;
  readonly thetaBits: number;
  readonly reserved: number;
}

export interface AttentionParameters {
  readonly layer: number;
  readonly position: number;
  readonly allocation: number;
  readonly sinkLength: number;
  readonly window: number;
  readonly kvHeads: number;
  readonly heads: number;
  readonly reserved: number;
}

export interface ResidentParameter<T> {
  readonly buffer: GPUBuffer;
  write(value: T): void;
  destroy(): void;
}

export interface ResidentParameterFactory {
  query(label: string): ResidentParameter<QueryParameters>;
  kv(label: string): ResidentParameter<KvParameters>;
  attention(label: string): ResidentParameter<AttentionParameters>;
}

export function createRawParameterFactory(device: GPUDevice): ResidentParameterFactory {
  const create = <T>(
    label: string,
    words: number,
    encode: (value: T) => readonly number[],
  ): ResidentParameter<T> => {
    const buffer = storageBuffer(device, label, words * 4, COPY_DST);
    return {
      buffer,
      write(value) {
        device.queue.writeBuffer(buffer, 0, bufferSource(Uint32Array.from(encode(value))));
      },
      destroy() {
        buffer.destroy();
      },
    };
  };
  return {
    query: (label) => create(label, 2, (value) => [value.position, value.thetaBits]),
    kv: (label) =>
      create(label, 8, (value) => [
        value.layer,
        value.position,
        value.allocation,
        value.sinkLength,
        value.window,
        value.kvHeads,
        value.thetaBits,
        value.reserved,
      ]),
    attention: (label) =>
      create(label, 8, (value) => [
        value.layer,
        value.position,
        value.allocation,
        value.sinkLength,
        value.window,
        value.kvHeads,
        value.heads,
        value.reserved,
      ]),
  };
}
