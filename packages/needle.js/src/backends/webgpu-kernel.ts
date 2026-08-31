import type { CqMatrix } from "../model/cact.js";
import { float16ToNumber } from "../model/fp16.js";

export const CQ_MATVEC_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> packed: array<u32>;
@group(0) @binding(1) var<storage, read> norms: array<f32>;
@group(0) @binding(2) var<storage, read> input: array<f32>;
@group(0) @binding(3) var<storage, read> codebook: array<f32>;
@group(0) @binding(4) var<storage, read> params: array<u32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

fn packed_index(row: u32, column: u32, record_bits: u32, row_bytes: u32) -> u32 {
  var bits = record_bits;
  if (record_bits == 5u) { bits = 2u; }
  let bit_position = row * row_bytes * 8u + column * bits;
  let word_index = bit_position >> 5u;
  let shift = bit_position & 31u;
  var value = packed[word_index] >> shift;
  if (shift + bits > 32u) {
    value = value | (packed[word_index + 1u] << (32u - shift));
  }
  return value & ((1u << bits) - 1u);
}

fn weight_value(index: u32, bits: u32, group_size: u32) -> f32 {
  if (bits == 2u) { return codebook[index]; }
  if (bits == 3u) { return codebook[4u + index]; }
  if (bits == 4u) { return codebook[12u + index]; }
  let centroid = 1.2240064 / sqrt(f32(group_size));
  if (index == 3u) { return -centroid; }
  if (index == 1u) { return centroid; }
  return 0.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let output_count = params[0];
  if (id.x >= output_count) { return; }
  let row = params[1] + id.x;
  let input_padded = params[2];
  let group_size = params[3];
  let bits = params[4];
  let row_bytes = params[5];
  let group_count = params[6];
  var total = 0.0;
  for (var group = 0u; group < group_count; group++) {
    var dot = 0.0;
    let offset = group * group_size;
    for (var column = 0u; column < group_size; column++) {
      let absolute_column = offset + column;
      let index = packed_index(row, absolute_column, bits, row_bytes);
      dot = dot + weight_value(index, bits, group_size) * input[absolute_column];
    }
    total = total + norms[row * group_count + group] * dot;
  }
  output[id.x] = total;
}
`;

export interface WebGpuMatrixData {
  readonly packedWords: Uint32Array;
  readonly norms: Float32Array;
}

const matrixDataCache = new WeakMap<CqMatrix, WebGpuMatrixData>();

export function webGpuMatrixData(matrix: CqMatrix): WebGpuMatrixData {
  const cached = matrixDataCache.get(matrix);
  if (cached) return cached;
  const packedWords = new Uint32Array(Math.ceil(matrix.packed.byteLength / 4));
  const packedView = new DataView(packedWords.buffer);
  for (let offset = 0; offset < matrix.packed.byteLength; offset += 4) {
    const word = (matrix.packed[offset] ?? 0)
      | ((matrix.packed[offset + 1] ?? 0) << 8)
      | ((matrix.packed[offset + 2] ?? 0) << 16)
      | ((matrix.packed[offset + 3] ?? 0) << 24);
    packedView.setUint32(offset, word >>> 0, true);
  }
  const normCount = matrix.norms.byteLength / 2;
  const norms = new Float32Array(normCount);
  for (let index = 0; index < normCount; index++) {
    norms[index] = float16ToNumber(matrix.norms.getUint16(index * 2, true));
  }
  const result = { packedWords, norms };
  matrixDataCache.set(matrix, result);
  return result;
}

export function matrixParameters(matrix: CqMatrix, rowStart: number, rowCount: number): Uint32Array {
  return new Uint32Array([
    rowCount,
    rowStart,
    matrix.inputSizePadded,
    matrix.groupSize,
    matrix.bits,
    matrix.rowByteLength,
    matrix.inputSizePadded / matrix.groupSize,
    0,
  ]);
}
