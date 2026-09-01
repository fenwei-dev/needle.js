import type { CqMatrix } from "../model/cact.js";
import { float16ToNumber } from "../model/fp16.js";

export const DEFAULT_MINIMUM_GPU_ROWS = 1024;

export function normalizeMinimumGpuRows(value: number | undefined): number {
  if (value === undefined || !Number.isFinite(value)) return DEFAULT_MINIMUM_GPU_ROWS;
  return Math.max(0, Math.floor(value));
}

export const CQ_MATVEC_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> packed: array<u32>;
@group(0) @binding(1) var<storage, read> norms: array<f32>;
@group(0) @binding(2) var<storage, read> input: array<f32>;
@group(0) @binding(3) var<storage, read> codebook: array<f32>;
@group(0) @binding(4) var<storage, read> params: array<u32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

fn packed_index(
  row: u32,
  column: u32,
  record_bits: u32,
  row_bytes: u32,
  packed_base: u32,
) -> u32 {
  var bits = record_bits;
  if (record_bits == 5u) { bits = 2u; }
  let bit_position = row * row_bytes * 8u + column * bits;
  let word_index = packed_base + (bit_position >> 5u);
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

var<workgroup> partials: array<f32, 32>;

@compute @workgroup_size(32)
fn main(
  @builtin(workgroup_id) workgroup: vec3u,
  @builtin(local_invocation_id) local: vec3u,
) {
  let output_index = workgroup.x;
  let output_count = params[0];
  if (output_index >= output_count) { return; }
  let row = params[1] + output_index;
  let group_size = params[3];
  let bits = params[4];
  let row_bytes = params[5];
  let group_count = params[6];
  var total = 0.0;
  for (var group = 0u; group < group_count; group++) {
    var dot = 0.0;
    let offset = group * group_size;
    for (var column = local.x; column < group_size; column += 32u) {
      let absolute_column = offset + column;
      let index = packed_index(row, absolute_column, bits, row_bytes, 0u);
      dot = dot + weight_value(index, bits, group_size) * input[absolute_column];
    }
    partials[local.x] = dot;
    workgroupBarrier();
    for (var stride = 16u; stride > 0u; stride >>= 1u) {
      if (local.x < stride) {
        partials[local.x] = partials[local.x] + partials[local.x + stride];
      }
      workgroupBarrier();
    }
    if (local.x == 0u) {
      total = total + norms[row * group_count + group] * partials[0];
    }
    workgroupBarrier();
  }
  if (local.x == 0u) {
    output[output_index] = total;
  }
}

// params[0] is the job count. Each job then occupies 12 u32 values:
// output start/count, row start, input padded/group geometry, quantization
// geometry, packed/norm/input offsets, and one reserved value.
@compute @workgroup_size(32)
fn batch_main(
  @builtin(workgroup_id) workgroup: vec3u,
  @builtin(local_invocation_id) local: vec3u,
) {
  let output_index = workgroup.x;
  let job_count = params[0];
  var base = 0u;
  var found = false;
  for (var job = 0u; job < job_count; job++) {
    let candidate = 1u + job * 12u;
    let output_start = params[candidate];
    let output_count = params[candidate + 1u];
    if (output_index >= output_start && output_index < output_start + output_count) {
      base = candidate;
      found = true;
      break;
    }
  }
  if (!found) { return; }

  let output_start = params[base];
  let local_output = output_index - output_start;
  let row = params[base + 2u] + local_output;
  let group_size = params[base + 4u];
  let bits = params[base + 5u];
  let row_bytes = params[base + 6u];
  let group_count = params[base + 7u];
  let packed_base = params[base + 8u];
  let norm_base = params[base + 9u];
  let input_base = params[base + 10u];
  var total = 0.0;
  for (var group = 0u; group < group_count; group++) {
    var dot = 0.0;
    let offset = group * group_size;
    for (var column = local.x; column < group_size; column += 32u) {
      let absolute_column = offset + column;
      let index = packed_index(row, absolute_column, bits, row_bytes, packed_base);
      dot = dot + weight_value(index, bits, group_size) * input[input_base + absolute_column];
    }
    partials[local.x] = dot;
    workgroupBarrier();
    for (var stride = 16u; stride > 0u; stride >>= 1u) {
      if (local.x < stride) {
        partials[local.x] = partials[local.x] + partials[local.x + stride];
      }
      workgroupBarrier();
    }
    if (local.x == 0u) {
      total = total + norms[norm_base + row * group_count + group] * partials[0];
    }
    workgroupBarrier();
  }
  if (local.x == 0u) {
    output[output_index] = total;
  }
}
`;

export interface WebGpuMatrixData {
  readonly packedWords: Uint32Array;
  readonly norms: Float32Array;
}

export interface WebGpuBatchMatrixEntry {
  readonly packedWordOffset: number;
  readonly normOffset: number;
}

export interface WebGpuBatchMatrixData {
  readonly packedWords: Uint32Array;
  readonly norms: Float32Array;
  readonly entries: readonly WebGpuBatchMatrixEntry[];
}

const matrixDataCache = new WeakMap<CqMatrix, WebGpuMatrixData>();

export function webGpuMatrixData(matrix: CqMatrix): WebGpuMatrixData {
  const cached = matrixDataCache.get(matrix);
  if (cached) return cached;
  const packedWords = new Uint32Array(Math.ceil(matrix.packed.byteLength / 4));
  const packedView = new DataView(packedWords.buffer);
  for (let offset = 0; offset < matrix.packed.byteLength; offset += 4) {
    const word =
      (matrix.packed[offset] ?? 0) |
      ((matrix.packed[offset + 1] ?? 0) << 8) |
      ((matrix.packed[offset + 2] ?? 0) << 16) |
      ((matrix.packed[offset + 3] ?? 0) << 24);
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

export function webGpuBatchMatrixData(matrices: readonly CqMatrix[]): WebGpuBatchMatrixData {
  const sources = matrices.map(webGpuMatrixData);
  const packedLength = sources.reduce((total, source) => total + source.packedWords.length, 0);
  const normLength = sources.reduce((total, source) => total + source.norms.length, 0);
  const packedWords = new Uint32Array(packedLength);
  const norms = new Float32Array(normLength);
  const entries: WebGpuBatchMatrixEntry[] = [];
  let packedWordOffset = 0;
  let normOffset = 0;
  for (const source of sources) {
    entries.push({ packedWordOffset, normOffset });
    packedWords.set(source.packedWords, packedWordOffset);
    norms.set(source.norms, normOffset);
    packedWordOffset += source.packedWords.length;
    normOffset += source.norms.length;
  }
  return { packedWords, norms, entries };
}

export interface WebGpuBatchJobParameters extends WebGpuBatchMatrixEntry {
  readonly matrix: CqMatrix;
  readonly rowStart: number;
  readonly rowCount: number;
  readonly outputStart: number;
  readonly inputOffset: number;
}

export function batchMatrixParameters(jobs: readonly WebGpuBatchJobParameters[]): Uint32Array {
  const parameters = new Uint32Array(1 + jobs.length * 12);
  parameters[0] = jobs.length;
  for (let index = 0; index < jobs.length; index++) {
    const job = jobs[index];
    if (!job) continue;
    const base = 1 + index * 12;
    parameters.set(
      [
        job.outputStart,
        job.rowCount,
        job.rowStart,
        job.matrix.inputSizePadded,
        job.matrix.groupSize,
        job.matrix.bits,
        job.matrix.rowByteLength,
        job.matrix.inputSizePadded / job.matrix.groupSize,
        job.packedWordOffset,
        job.normOffset,
        job.inputOffset,
        0,
      ],
      base,
    );
  }
  return parameters;
}

export function matrixParameters(
  matrix: CqMatrix,
  rowStart: number,
  rowCount: number,
): Uint32Array {
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
