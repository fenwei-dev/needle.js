import { describe, expect, test } from "bun:test";
import type { CactTensorRecord, CqMatrix } from "../src/model/cact.js";
import { CactDType } from "../src/model/cact.js";
import { cqMatvec, dequantizeCqRow, fastWalshHadamard } from "../src/backends/cq.js";
import { numberToFloat16 } from "../src/model/fp16.js";

function pack(indices: readonly number[], bits: number): Uint8Array {
  const output = new Uint8Array(Math.ceil(indices.length * bits / 8));
  indices.forEach((value, index) => {
    const bit = index * bits;
    const byte = bit >>> 3;
    const shift = bit & 7;
    output[byte] = (output[byte] ?? 0) | ((value << shift) & 0xff);
    if (shift + bits > 8) output[byte + 1] = (output[byte + 1] ?? 0) | (value >>> (8 - shift));
  });
  return output;
}

function matrix(bits: 2 | 3 | 4 | 5): CqMatrix {
  const group = 8;
  const rows = 2;
  const sourceBits = bits === 5 ? 2 : bits;
  const max = bits === 5 ? 3 : 1 << bits;
  const indices = Array.from({ length: rows * group }, (_, index) => index % max);
  // Ternary record crumbs represent trit indices 0,1,2 as 3,0,1.
  const encoded = bits === 5 ? indices.map((index) => index === 0 ? 3 : index - 1) : indices;
  const packed = pack(encoded, sourceBits);
  const normBytes = new ArrayBuffer(rows * 2);
  const normView = new DataView(normBytes);
  normView.setUint16(0, numberToFloat16(1.25), true);
  normView.setUint16(2, numberToFloat16(0.75), true);
  const codebooks = new Map<number, Float32Array>([
    [2, new Float32Array([-0.5, -0.15, 0.15, 0.5])],
    [3, new Float32Array([-0.6, -0.4, -0.2, -0.05, 0.05, 0.2, 0.4, 0.6])],
    [4, Float32Array.from({ length: 16 }, (_, index) => (index - 7.5) / 12)],
  ]);
  const record: CactTensorRecord = {
    index: 0,
    dtype: CactDType.CactusQuant,
    shape: [rows, group],
    offset: 0,
    byteLength: packed.byteLength + normBytes.byteLength,
    groupSize: group,
    bits,
  };
  return {
    kind: "cq",
    shape: [rows, group],
    outputSize: rows,
    inputSize: group,
    inputSizePadded: group,
    groupSize: group,
    bits,
    rowByteLength: group * sourceBits / 8,
    packed,
    norms: normView,
    codebooks,
    record,
  };
}

describe("Cactus Quants kernels", () => {
  test("FWHT is self-inverse up to n", () => {
    const values = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const original = values.slice();
    fastWalshHadamard(values);
    fastWalshHadamard(values);
    for (let index = 0; index < values.length; index++) {
      expect(values[index]).toBeCloseTo((original[index] ?? 0) * values.length, 5);
    }
  });

  for (const bits of [2, 3, 4, 5] as const) {
    test(`W${bits === 5 ? "1.58" : bits} direct matvec equals a dequantized row dot`, () => {
      const current = matrix(bits);
      const input = Float32Array.from({ length: 8 }, (_, index) => Math.sin(index + 0.25));
      const direct = cqMatvec(current, input);
      for (let row = 0; row < current.outputSize; row++) {
        const weights = dequantizeCqRow(current, row);
        let expected = 0;
        for (let index = 0; index < input.length; index++) expected += (weights[index] ?? 0) * (input[index] ?? 0);
        expect(direct[row]).toBeCloseTo(expected, 4);
      }
    });
  }
});
