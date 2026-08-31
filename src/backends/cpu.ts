import type { CactWeights, CqMatrix } from "../model/cact.js";
import type { InferenceBackend, MatrixRowRange } from "./backend.js";
import { cqMatvec, dequantizeCqRow } from "./cq.js";

/** Portable, dependency-free TypeScript backend. */
export class CpuBackend implements InferenceBackend {
  readonly kind = "cpu" as const;
  readonly name = "Pure TypeScript CPU";

  constructor(readonly weights: CactWeights) {}

  matvec(matrix: CqMatrix, input: Float32Array, range: MatrixRowRange = {}): Float32Array {
    const rowStart = range.rowStart ?? 0;
    const rowCount = range.rowCount ?? matrix.outputSize - rowStart;
    return cqMatvec(matrix, input, rowStart, rowCount);
  }

  row(matrix: CqMatrix, row: number, output?: Float32Array): Float32Array {
    return dequantizeCqRow(matrix, row, output);
  }

  dispose(): void {
    // The CPU backend owns no resources outside normal garbage-collected arrays.
  }
}

export function createCpuBackend(weights: CactWeights): CpuBackend {
  return new CpuBackend(weights);
}
