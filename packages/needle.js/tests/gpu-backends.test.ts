import { describe, expect, test } from "bun:test";
import { batchMatrixParameters, webGpuBatchMatrixData } from "../src/backends/webgpu-kernel.js";
import type { CqMatrix } from "../src/model/cact.js";

function fixture(): CqMatrix {
  const codebook = new Float32Array(28);
  codebook.set([-0.5, -0.1, 0.1, 0.5]);
  return {
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
    record: {
      index: 0,
      dtype: 3,
      shape: [1, 8],
      offset: 0,
      byteLength: 4,
      groupSize: 8,
      bits: 2,
    },
  };
}

describe("WebGPU matrix arenas", () => {
  test("packs multiple matrices and their dispatch metadata into one arena", () => {
    const matrix = fixture();
    const data = webGpuBatchMatrixData([matrix, matrix]);
    expect(data.packedWords).toHaveLength(2);
    expect(data.norms).toHaveLength(2);
    expect(data.entries).toEqual([
      { packedWordOffset: 0, normOffset: 0 },
      { packedWordOffset: 1, normOffset: 1 },
    ]);
    const entry = data.entries[0];
    if (!entry) throw new Error("Missing packed matrix entry");
    const parameters = batchMatrixParameters([
      {
        matrix,
        rowStart: 0,
        rowCount: 1,
        outputStart: 0,
        inputOffset: 0,
        ...entry,
      },
    ]);
    expect(parameters[0]).toBe(1);
    expect(parameters[1]).toBe(0);
    expect(parameters[2]).toBe(1);
    expect(parameters[9]).toBe(0);
  });
});
