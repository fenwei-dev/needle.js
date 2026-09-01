/// <reference types="@webgpu/types" preserve="true" />
import type { CactLayer, CactWeights, CqMatrix } from "../model/cact.js";

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

export interface FusedAttentionResetOptions {
  readonly maximumLength: number;
  readonly sinkLength: number;
  readonly kvCache: "int8" | "float32";
  readonly collectConfidence: boolean;
}

export interface FusedAttentionResult {
  readonly kind: "projected" | "delta" | "nextX";
  readonly values: Float32Array;
}

export interface FusedAttentionRequest {
  readonly layer: CactLayer;
  readonly layerIndex: number;
  readonly position: number;
  readonly input: Float32Array;
  readonly blockInput: Float32Array;
  readonly updateInput: Float32Array;
  readonly x: Float32Array;
  readonly phiPost: Float32Array;
  readonly phiResidual: Float32Array;
}

export interface ResidentTokenSelection {
  readonly id: number;
  readonly logProbability: number;
}

export interface ResidentLayerRequest {
  readonly layerIndex: number;
  readonly position: number;
  readonly engramKey?: Float32Array;
  readonly engramValue?: Float32Array;
}

/** Stateful accelerator owned and disposed by its backend. */
export interface FusedAttentionSession {
  reset(options: FusedAttentionResetOptions): boolean;
  forward(request: FusedAttentionRequest): Promise<FusedAttentionResult>;
  beginResidentToken?(lanes: Float32Array): Promise<boolean>;
  forwardResidentLayer?(request: ResidentLayerRequest): Promise<void>;
  finishResidentToken?(wantLogits: boolean): Promise<Float32Array | null>;
  finishResidentTokenForSelection?(): Promise<void>;
  selectResidentToken?(allowedTokenIds?: Uint32Array): Promise<ResidentTokenSelection>;
  residentLayersEnabled?(): boolean;
  residentConfidence?(): Promise<number | undefined>;
  readResidentLanes?(): Promise<Float32Array>;
  dispose(): void;
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

  /** Optional stateful Q/K/V, KV-cache, attention, gate, and output fusion. */
  createFusedAttentionSession?(): FusedAttentionSession | undefined;

  /** Dequantizes one row. Embedding and engram gathers use this operation. */
  row(matrix: CqMatrix, row: number, output?: Float32Array): Float32Array;

  dispose(): void | Promise<void>;
}

export interface BackendFactoryOptions {
  readonly device?: GPUDevice;
}
