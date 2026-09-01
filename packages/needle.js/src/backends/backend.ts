/// <reference types="@webgpu/types" preserve="true" />
import type { CactWeights, CqMatrix } from "../model/cact.js";

export type BackendKind = "cpu" | "typegpu" | "vgpu";

export interface MatrixRowRange {
  readonly rowStart?: number;
  readonly rowCount?: number;
}

export interface MatvecRequest {
  readonly matrix: CqMatrix;
  readonly input: Float32Array;
  readonly range?: MatrixRowRange;
}

/** Minimal operator surface needed by the incremental Needle 2 decoder. */
export interface InferenceBackend {
  readonly kind: BackendKind;
  readonly name: string;
  readonly weights: CactWeights;

  /** Computes selected rows of `matrix @ input`. */
  matvec(
    matrix: CqMatrix,
    input: Float32Array,
    range?: MatrixRowRange,
  ): Float32Array | Promise<Float32Array>;

  /**
   * Computes independent matvecs together. GPU implementations use one host
   * synchronization for the group; the default runtime falls back to matvec().
   */
  matvecBatch?(
    requests: readonly MatvecRequest[],
  ): readonly Float32Array[] | Promise<readonly Float32Array[]>;

  /** Dequantizes one row. Embedding and engram gathers use this operation. */
  row(matrix: CqMatrix, row: number, output?: Float32Array): Float32Array;

  dispose(): void | Promise<void>;
}

export interface BackendFactoryOptions {
  readonly device?: GPUDevice;
}
