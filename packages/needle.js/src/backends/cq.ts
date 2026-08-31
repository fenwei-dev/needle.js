import { invariant } from "../errors.js";
import type { CqMatrix } from "../model/cact.js";
import { float16ToNumber } from "../model/fp16.js";

const TERNARY_RECORD_BITS = 5;
const TERNARY_CENTROID = 1.2240064;

export function fastWalshHadamard(values: Float32Array, offset = 0, length = values.length): void {
  invariant(
    length > 0 && (length & (length - 1)) === 0,
    "UNSUPPORTED_CACT",
    `Walsh-Hadamard length ${length} is not a power of two`,
  );
  for (let stride = 1; stride < length; stride <<= 1) {
    const block = stride << 1;
    for (let base = 0; base < length; base += block) {
      for (let inner = 0; inner < stride; inner++) {
        const leftIndex = offset + base + inner;
        const rightIndex = leftIndex + stride;
        const left = values[leftIndex] ?? 0;
        const right = values[rightIndex] ?? 0;
        values[leftIndex] = left + right;
        values[rightIndex] = left - right;
      }
    }
  }
}

/** Applies normalized H independently to each quantization group. */
export function prepareCqActivation(matrix: CqMatrix, input: Float32Array): Float32Array {
  invariant(
    input.length === matrix.inputSize,
    "INVALID_CACT",
    `Matrix expects ${matrix.inputSize} inputs, received ${input.length}`,
  );
  const prepared = new Float32Array(matrix.inputSizePadded);
  prepared.set(input);
  const inverseRoot = 1 / Math.sqrt(matrix.groupSize);
  for (let offset = 0; offset < prepared.length; offset += matrix.groupSize) {
    fastWalshHadamard(prepared, offset, matrix.groupSize);
    for (let index = 0; index < matrix.groupSize; index++) {
      prepared[offset + index] = (prepared[offset + index] ?? 0) * inverseRoot;
    }
  }
  return prepared;
}

interface CqLuts {
  readonly lut2: Float32Array;
  readonly lut4: Float32Array;
}

const lutCache = new WeakMap<object, CqLuts>();

function luts(matrix: CqMatrix): CqLuts {
  const key = matrix.codebooks as object;
  const cached = lutCache.get(key);
  if (cached) return cached;
  const codebook2 = matrix.codebooks.get(2);
  const codebook4 = matrix.codebooks.get(4);
  invariant(
    codebook2?.length === 4 && codebook4?.length === 16,
    "INVALID_CACT",
    "CQ2/CQ4 codebooks are missing",
  );
  const lut2 = new Float32Array(256 * 4);
  const lut4 = new Float32Array(256 * 2);
  for (let byte = 0; byte < 256; byte++) {
    for (let crumb = 0; crumb < 4; crumb++) {
      lut2[byte * 4 + crumb] = codebook2[(byte >>> (crumb * 2)) & 3] ?? 0;
    }
    lut4[byte * 2] = codebook4[byte & 15] ?? 0;
    lut4[byte * 2 + 1] = codebook4[(byte >>> 4) & 15] ?? 0;
  }
  const result = { lut2, lut4 };
  lutCache.set(key, result);
  return result;
}

function normAt(matrix: CqMatrix, row: number, group: number, groupsPerRow: number): number {
  return float16ToNumber(matrix.norms.getUint16((row * groupsPerRow + group) * 2, true));
}

/** Quantized matrix-vector multiply without expanding model weights. */
export function cqMatvecPrepared(
  matrix: CqMatrix,
  prepared: Float32Array,
  rowStart = 0,
  rowCount = matrix.outputSize - rowStart,
): Float32Array {
  invariant(
    prepared.length === matrix.inputSizePadded,
    "INVALID_CACT",
    "Prepared activation has the wrong padded length",
  );
  invariant(
    Number.isInteger(rowStart) &&
      Number.isInteger(rowCount) &&
      rowStart >= 0 &&
      rowCount >= 0 &&
      rowStart + rowCount <= matrix.outputSize,
    "INVALID_CACT",
    `Invalid matrix row range ${rowStart}:${rowStart + rowCount}`,
  );
  const output = new Float32Array(rowCount);
  const groupSize = matrix.groupSize;
  const groupsPerRow = matrix.inputSizePadded / groupSize;
  const packed = matrix.packed;

  if (matrix.bits === 2) {
    const { lut2 } = luts(matrix);
    for (let localRow = 0; localRow < rowCount; localRow++) {
      const row = rowStart + localRow;
      const rowOffset = row * matrix.rowByteLength;
      let total = 0;
      for (let group = 0; group < groupsPerRow; group++) {
        const inputOffset = group * groupSize;
        const packedOffset = rowOffset + (group * groupSize) / 4;
        let dot = 0;
        for (let column = 0; column < groupSize; column += 4) {
          const byte = packed[packedOffset + (column >>> 2)] ?? 0;
          const lookup = byte * 4;
          dot +=
            (lut2[lookup] ?? 0) * (prepared[inputOffset + column] ?? 0) +
            (lut2[lookup + 1] ?? 0) * (prepared[inputOffset + column + 1] ?? 0) +
            (lut2[lookup + 2] ?? 0) * (prepared[inputOffset + column + 2] ?? 0) +
            (lut2[lookup + 3] ?? 0) * (prepared[inputOffset + column + 3] ?? 0);
        }
        total += normAt(matrix, row, group, groupsPerRow) * dot;
      }
      output[localRow] = total;
    }
    return output;
  }

  if (matrix.bits === 4) {
    const { lut4 } = luts(matrix);
    for (let localRow = 0; localRow < rowCount; localRow++) {
      const row = rowStart + localRow;
      const rowOffset = row * matrix.rowByteLength;
      let total = 0;
      for (let group = 0; group < groupsPerRow; group++) {
        const inputOffset = group * groupSize;
        const packedOffset = rowOffset + (group * groupSize) / 2;
        let dot = 0;
        for (let column = 0; column < groupSize; column += 2) {
          const byte = packed[packedOffset + (column >>> 1)] ?? 0;
          const lookup = byte * 2;
          dot +=
            (lut4[lookup] ?? 0) * (prepared[inputOffset + column] ?? 0) +
            (lut4[lookup + 1] ?? 0) * (prepared[inputOffset + column + 1] ?? 0);
        }
        total += normAt(matrix, row, group, groupsPerRow) * dot;
      }
      output[localRow] = total;
    }
    return output;
  }

  if (matrix.bits === TERNARY_RECORD_BITS) {
    const coefficient = TERNARY_CENTROID / Math.sqrt(groupSize);
    const crumbs = [0, coefficient, 0, -coefficient] as const;
    for (let localRow = 0; localRow < rowCount; localRow++) {
      const row = rowStart + localRow;
      const rowOffset = row * matrix.rowByteLength;
      let total = 0;
      for (let group = 0; group < groupsPerRow; group++) {
        const inputOffset = group * groupSize;
        const packedOffset = rowOffset + (group * groupSize) / 4;
        let dot = 0;
        for (let column = 0; column < groupSize; column += 4) {
          const byte = packed[packedOffset + (column >>> 2)] ?? 0;
          dot +=
            (crumbs[byte & 3] ?? 0) * (prepared[inputOffset + column] ?? 0) +
            (crumbs[(byte >>> 2) & 3] ?? 0) * (prepared[inputOffset + column + 1] ?? 0) +
            (crumbs[(byte >>> 4) & 3] ?? 0) * (prepared[inputOffset + column + 2] ?? 0) +
            (crumbs[(byte >>> 6) & 3] ?? 0) * (prepared[inputOffset + column + 3] ?? 0);
        }
        total += normAt(matrix, row, group, groupsPerRow) * dot;
      }
      output[localRow] = total;
    }
    return output;
  }

  // Three-bit rows are an uninterrupted little-endian bitstream.
  const codebook3 = matrix.codebooks.get(3);
  invariant(codebook3?.length === 8, "INVALID_CACT", "CQ3 codebook is missing");
  for (let localRow = 0; localRow < rowCount; localRow++) {
    const row = rowStart + localRow;
    const rowOffset = row * matrix.rowByteLength;
    let total = 0;
    for (let group = 0; group < groupsPerRow; group++) {
      const inputOffset = group * groupSize;
      let dot = 0;
      for (let column = 0; column < groupSize; column++) {
        const absoluteColumn = group * groupSize + column;
        const bitPosition = absoluteColumn * 3;
        const bytePosition = rowOffset + (bitPosition >>> 3);
        const shift = bitPosition & 7;
        const word = (packed[bytePosition] ?? 0) | ((packed[bytePosition + 1] ?? 0) << 8);
        dot += (codebook3[(word >>> shift) & 7] ?? 0) * (prepared[inputOffset + column] ?? 0);
      }
      total += normAt(matrix, row, group, groupsPerRow) * dot;
    }
    output[localRow] = total;
  }
  return output;
}

export function cqMatvec(
  matrix: CqMatrix,
  input: Float32Array,
  rowStart = 0,
  rowCount = matrix.outputSize - rowStart,
): Float32Array {
  return cqMatvecPrepared(matrix, prepareCqActivation(matrix, input), rowStart, rowCount);
}

/** Dequantizes one matrix row, primarily for embedding and engram gathers. */
export function dequantizeCqRow(
  matrix: CqMatrix,
  row: number,
  output?: Float32Array,
): Float32Array {
  const result = output ?? new Float32Array(matrix.inputSize);
  invariant(
    Number.isInteger(row) && row >= 0 && row < matrix.outputSize,
    "INVALID_CACT",
    `Matrix row ${row} is out of range`,
  );
  invariant(result.length >= matrix.inputSize, "INVALID_CACT", "Row output buffer is too short");
  const groupsPerRow = matrix.inputSizePadded / matrix.groupSize;
  const temporary = new Float32Array(matrix.groupSize);
  const inverseRoot = 1 / Math.sqrt(matrix.groupSize);
  const rowOffset = row * matrix.rowByteLength;
  const codebook =
    matrix.bits === 5
      ? new Float32Array([-TERNARY_CENTROID * inverseRoot, 0, TERNARY_CENTROID * inverseRoot])
      : matrix.codebooks.get(matrix.bits);
  invariant(codebook !== undefined, "INVALID_CACT", `Codebook W${matrix.bits} is missing`);

  for (let group = 0; group < groupsPerRow; group++) {
    const norm = normAt(matrix, row, group, groupsPerRow);
    for (let column = 0; column < matrix.groupSize; column++) {
      const absoluteColumn = group * matrix.groupSize + column;
      let index: number;
      if (matrix.bits === 2 || matrix.bits === 5) {
        const byte = matrix.packed[rowOffset + (absoluteColumn >>> 2)] ?? 0;
        const crumb = (byte >>> ((absoluteColumn & 3) * 2)) & 3;
        index = matrix.bits === 5 ? (crumb === 3 ? 0 : crumb + 1) : crumb;
      } else if (matrix.bits === 4) {
        const byte = matrix.packed[rowOffset + (absoluteColumn >>> 1)] ?? 0;
        index = (byte >>> ((absoluteColumn & 1) * 4)) & 15;
      } else {
        const bitPosition = absoluteColumn * 3;
        const bytePosition = rowOffset + (bitPosition >>> 3);
        const shift = bitPosition & 7;
        const word =
          (matrix.packed[bytePosition] ?? 0) | ((matrix.packed[bytePosition + 1] ?? 0) << 8);
        index = (word >>> shift) & 7;
      }
      temporary[column] = (codebook[index] ?? 0) * norm;
    }
    fastWalshHadamard(temporary);
    const outputOffset = group * matrix.groupSize;
    const count = Math.min(matrix.groupSize, matrix.inputSize - outputOffset);
    for (let column = 0; column < count; column++) {
      result[outputOffset + column] = (temporary[column] ?? 0) * inverseRoot;
    }
  }
  return result;
}

export function denseMatvec(
  matrix: Float32Array,
  outputSize: number,
  input: Float32Array,
  bias?: Float32Array,
): Float32Array {
  invariant(
    matrix.length === outputSize * input.length,
    "INVALID_CACT",
    "Dense matrix shape does not match the vector",
  );
  const result = new Float32Array(outputSize);
  for (let row = 0; row < outputSize; row++) {
    let sum = bias?.[row] ?? 0;
    const offset = row * input.length;
    for (let column = 0; column < input.length; column++) {
      sum += (matrix[offset + column] ?? 0) * (input[column] ?? 0);
    }
    result[row] = sum;
  }
  return result;
}
