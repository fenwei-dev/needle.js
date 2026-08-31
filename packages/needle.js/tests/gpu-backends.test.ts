import { describe, expect, test } from "bun:test";
import { init } from "vgpu/mock";
import { TypeGPUBackend } from "../src/backends/typegpu.js";
import { VGPUBackend } from "../src/backends/vgpu.js";
import type { CactWeights, CqMatrix } from "../src/model/cact.js";

function fixture(): { weights: CactWeights; matrix: CqMatrix } {
  const codebook = new Float32Array(28);
  codebook.set([-0.5, -0.1, 0.1, 0.5]);
  const record = {
    index: 0,
    dtype: 3,
    shape: [1, 8],
    offset: 0,
    byteLength: 4,
    groupSize: 8,
    bits: 2,
  } as const;
  const matrix: CqMatrix = {
    kind: "cq",
    shape: [1, 8],
    outputSize: 1,
    inputSize: 8,
    inputSizePadded: 8,
    groupSize: 8,
    bits: 2,
    rowByteLength: 2,
    packed: new Uint8Array(2),
    norms: new DataView(new ArrayBuffer(2)),
    codebooks: new Map([
      [2, codebook.slice(0, 4)],
      [3, codebook.slice(4, 12)],
      [4, codebook.slice(12, 28)],
    ]),
    record,
  };
  return {
    matrix,
    // Only fields consumed while constructing an operator backend are needed
    // by this deterministic WebGPU API smoke fixture.
    weights: { tensors: [matrix], codebook } as unknown as CactWeights,
  };
}

describe("optional GPU backends", () => {
  test("TypeGPU initializes, resolves WGSL, dispatches, and reads back", async () => {
    const mock = await init();
    const { weights, matrix } = fixture();
    const backend = await TypeGPUBackend.create(weights, {
      device: mock.gpu,
      minimumGpuRows: 0,
    });
    try {
      expect(backend.kind).toBe("typegpu");
      expect(await backend.matvec(matrix, new Float32Array(8))).toHaveLength(1);
    } finally {
      backend.dispose();
      mock.dispose();
    }
  });

  test("vgpu initializes storage/compute, dispatches, and reads back", async () => {
    const mock = await init();
    const { weights, matrix } = fixture();
    const backend = await VGPUBackend.create(weights, {
      gpu: mock,
      node: true,
      minimumGpuRows: 0,
    });
    try {
      expect(backend.kind).toBe("vgpu");
      expect(await backend.matvec(matrix, new Float32Array(8))).toHaveLength(1);
    } finally {
      backend.dispose();
      mock.dispose();
    }
  });
});
