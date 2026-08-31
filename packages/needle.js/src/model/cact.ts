import { NeedleError, invariant } from "../errors.js";
import { decodeFloat16Array } from "./fp16.js";

export const CACT_TAG = 0x05e12a83;
export const CACT_HEADER_BYTES = 120;
export const CACT_RECORD_BYTES = 44;

export const enum CactDType {
  Float16 = 1,
  Float32 = 2,
  CactusQuant = 3,
  Raw = 4,
}

export interface CactGeometry {
  readonly vocabularySize: number;
  readonly modelDimension: number;
  readonly numberOfHeads: number;
  readonly numberOfKVHeads: number;
  readonly numberOfLayers: number;
  readonly headDimension: number;
  readonly maximumSequenceLength: number;
  readonly hadamardDimension: number;
  readonly mhcLanes: number;
  readonly engramSlots: number;
  readonly engramSubDimension: number;
  readonly numberOfEngramTables: number;
  readonly engramConvolutionTaps: number;
  readonly engramConvolutionDilation: number;
  readonly engramOrders: readonly number[];
  readonly engramLayers: readonly number[];
  readonly ropeTheta: number;
  readonly kvWindow: number;
  readonly kvBits: number;
}

export interface CactTensorRecord {
  readonly index: number;
  readonly dtype: CactDType;
  readonly shape: readonly number[];
  readonly offset: number;
  readonly byteLength: number;
  readonly groupSize: number;
  readonly bits: number;
}

export interface DenseTensor {
  readonly kind: "dense";
  readonly shape: readonly number[];
  readonly data: Float32Array;
  readonly record: CactTensorRecord;
}

export interface RawTensor {
  readonly kind: "raw";
  readonly shape: readonly number[];
  readonly data: Uint8Array;
  readonly record: CactTensorRecord;
}

export interface CqMatrix {
  readonly kind: "cq";
  readonly shape: readonly [number, number];
  readonly outputSize: number;
  readonly inputSize: number;
  readonly inputSizePadded: number;
  readonly groupSize: number;
  /** 2, 3, 4, or 5 (5 is the ternary record marker). */
  readonly bits: number;
  readonly rowByteLength: number;
  readonly packed: Uint8Array;
  readonly norms: DataView;
  readonly codebooks: ReadonlyMap<number, Float32Array>;
  readonly record: CactTensorRecord;
}

export type CactTensor = DenseTensor | RawTensor | CqMatrix;

export interface CactLayer {
  readonly normInput: Float32Array;
  readonly queryProjection: CqMatrix;
  readonly keyProjection: CqMatrix;
  readonly valueProjection: CqMatrix;
  readonly queryNorm: Float32Array;
  readonly keyNorm: Float32Array;
  readonly gateProjection: CqMatrix;
  readonly outputProjection: CqMatrix;
  readonly postAttentionNorm: Float32Array;
  readonly attentionGate: number;
  readonly preHadamardNorm: Float32Array;
  readonly hadamardD1: Float32Array;
  readonly hadamardD2: Float32Array;
  readonly hadamardD3: Float32Array;
}

export interface CactEngramSite {
  readonly tables: CqMatrix;
  readonly keyProjection: CqMatrix;
  readonly valueProjection: CqMatrix;
  readonly taps: Float32Array;
}

export type ProbeHeadName = "contrastive" | "confidence";

export interface CactProbeHead {
  readonly name: ProbeHeadName;
  readonly probes: Float32Array;
  readonly probeCount: number;
  readonly projection: Float32Array;
  readonly outputSize: number;
  readonly bias: Float32Array;
}

export interface CactWeights {
  readonly bytes: Uint8Array;
  readonly geometry: CactGeometry;
  readonly numberOfTensors: number;
  readonly codebook: Float32Array;
  readonly codebooks: ReadonlyMap<number, Float32Array>;
  readonly records: readonly CactTensorRecord[];
  readonly tensors: readonly CactTensor[];
  readonly embedding: CqMatrix;
  readonly layers: readonly CactLayer[];
  readonly mhcAPre: Float32Array;
  readonly mhcAPost: Float32Array;
  readonly mhcAResidual: Float32Array;
  readonly mhcBPre: Float32Array;
  readonly mhcBPost: Float32Array;
  readonly mhcBResidual: Float32Array;
  readonly mhcPhiPre: CqMatrix;
  readonly mhcPhiPost: CqMatrix;
  readonly mhcPhiResidual: CqMatrix;
  readonly engrams: readonly CactEngramSite[];
  readonly finalNorm: Float32Array;
  readonly heads: ReadonlyMap<ProbeHeadName, CactProbeHead>;
  readonly tokenizerBlob: Uint8Array;
}

function product(shape: readonly number[]): number {
  let result = 1;
  for (const value of shape) {
    invariant(Number.isSafeInteger(value) && value >= 0, "INVALID_CACT", `Invalid tensor dimension ${value}`);
    result *= value;
    invariant(Number.isSafeInteger(result), "INVALID_CACT", "Tensor element count exceeds JavaScript's safe integer range");
  }
  return result;
}

function asBytes(source: ArrayBuffer | ArrayBufferView): Uint8Array {
  if (source instanceof ArrayBuffer) return new Uint8Array(source);
  return new Uint8Array(source.buffer, source.byteOffset, source.byteLength);
}

function assertRange(bytes: Uint8Array, offset: number, length: number, label: string): void {
  invariant(
    Number.isSafeInteger(offset) && Number.isSafeInteger(length) && offset >= 0 && length >= 0 && offset + length <= bytes.byteLength,
    "INVALID_CACT",
    `${label} points outside the ${bytes.byteLength}-byte archive (offset ${offset}, length ${length})`,
  );
}

function dense(tensor: CactTensor, name: string): Float32Array {
  invariant(tensor.kind === "dense", "INVALID_CACT", `${name} must be an FP16/FP32 tensor`);
  return tensor.data;
}

function cq(tensor: CactTensor, name: string): CqMatrix {
  invariant(tensor.kind === "cq", "INVALID_CACT", `${name} must be a Cactus-Quant matrix`);
  return tensor;
}

/**
 * Parses the official little-endian `.cact` deployment archive without
 * expanding quantized matrices. Dense control tensors are converted to f32;
 * packed model weights remain zero-copy views of the supplied byte array.
 */
export function parseCact(source: ArrayBuffer | ArrayBufferView): CactWeights {
  const bytes = asBytes(source);
  assertRange(bytes, 0, CACT_HEADER_BYTES, "CACT header");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

  const u32 = (index: number): number => view.getUint32(index * 4, true);
  const tag = u32(0);
  invariant(tag === CACT_TAG, "INVALID_CACT", `Invalid .cact tag 0x${tag.toString(16)} (expected 0x${CACT_TAG.toString(16)})`);

  const numberOfTensors = u32(1);
  const codebookLength = u32(2);
  invariant(numberOfTensors > 0, "INVALID_CACT", "The .cact archive has no tensors");
  invariant(codebookLength >= 28, "UNSUPPORTED_CACT", `Needle 2 requires the CQ2/CQ3/CQ4 codebook (28 entries); archive has ${codebookLength}`);

  const numberOfOrders = u32(19);
  const numberOfSites = u32(24);
  invariant(numberOfOrders <= 4 && numberOfSites <= 4, "UNSUPPORTED_CACT", "The current .cact format supports at most four engram orders and sites");

  const orders = Array.from({ length: numberOfOrders }, (_, index) => u32(20 + index));
  const sites = Array.from({ length: numberOfSites }, (_, index) => u32(25 + index));
  const geometry: CactGeometry = {
    kvWindow: u32(3),
    kvBits: u32(4),
    vocabularySize: u32(5),
    modelDimension: u32(6),
    numberOfHeads: u32(7),
    numberOfKVHeads: u32(8),
    numberOfLayers: u32(9),
    headDimension: u32(10),
    maximumSequenceLength: u32(11),
    hadamardDimension: u32(12),
    mhcLanes: u32(13),
    engramSlots: u32(14),
    engramSubDimension: u32(15),
    numberOfEngramTables: u32(16),
    engramConvolutionTaps: u32(17),
    engramConvolutionDilation: u32(18),
    engramOrders: orders,
    engramLayers: sites,
    ropeTheta: view.getFloat32(29 * 4, true),
  };

  invariant(geometry.modelDimension > 0 && geometry.vocabularySize > 0, "INVALID_CACT", "Model geometry contains a zero-sized vocabulary or hidden dimension");
  invariant(geometry.numberOfHeads > 0 && geometry.numberOfKVHeads > 0 && geometry.numberOfHeads % geometry.numberOfKVHeads === 0, "UNSUPPORTED_CACT", "Attention heads must be a positive multiple of KV heads");
  invariant(geometry.numberOfHeads * geometry.headDimension > 0, "INVALID_CACT", "Invalid attention dimensions");
  invariant((geometry.hadamardDimension & (geometry.hadamardDimension - 1)) === 0, "UNSUPPORTED_CACT", "Hadamard dimension must be a power of two");

  const codebookOffset = CACT_HEADER_BYTES;
  assertRange(bytes, codebookOffset, codebookLength * 4, "CQ codebook");
  const codebook = new Float32Array(codebookLength);
  for (let index = 0; index < codebookLength; index++) {
    codebook[index] = view.getFloat32(codebookOffset + index * 4, true);
  }
  const codebooks = new Map<number, Float32Array>([
    [2, codebook.slice(0, 4)],
    [3, codebook.slice(4, 12)],
    [4, codebook.slice(12, 28)],
  ]);

  const directoryOffset = codebookOffset + codebookLength * 4;
  assertRange(bytes, directoryOffset, numberOfTensors * CACT_RECORD_BYTES, "Tensor directory");
  const records: CactTensorRecord[] = [];
  for (let index = 0; index < numberOfTensors; index++) {
    const offset = directoryOffset + index * CACT_RECORD_BYTES;
    const dtype = view.getUint8(offset) as CactDType;
    const dimensions = view.getUint8(offset + 1);
    invariant(dimensions <= 4, "UNSUPPORTED_CACT", `Tensor ${index} has ${dimensions} dimensions; the format supports four`);
    const shape = Array.from({ length: dimensions }, (_, dimension) => view.getUint32(offset + 4 + dimension * 4, true));
    const blobOffset = Number(view.getBigUint64(offset + 20, true));
    const byteLength = Number(view.getBigUint64(offset + 28, true));
    invariant(Number.isSafeInteger(blobOffset) && Number.isSafeInteger(byteLength), "UNSUPPORTED_CACT", `Tensor ${index} offset does not fit safely in JavaScript`);
    assertRange(bytes, blobOffset, byteLength, `Tensor ${index}`);
    records.push({
      index,
      dtype,
      shape,
      offset: blobOffset,
      byteLength,
      groupSize: view.getUint32(offset + 36, true),
      bits: view.getUint32(offset + 40, true),
    });
  }

  const tensors: CactTensor[] = records.map((record): CactTensor => {
    const count = product(record.shape);
    if (record.dtype === CactDType.Float16) {
      invariant(record.byteLength === count * 2, "INVALID_CACT", `FP16 tensor ${record.index} has an inconsistent byte length`);
      return {
        kind: "dense",
        shape: record.shape,
        data: decodeFloat16Array(view, record.offset, count),
        record,
      };
    }
    if (record.dtype === CactDType.Float32) {
      invariant(record.byteLength === count * 4, "INVALID_CACT", `FP32 tensor ${record.index} has an inconsistent byte length`);
      const data = new Float32Array(count);
      for (let index = 0; index < count; index++) data[index] = view.getFloat32(record.offset + index * 4, true);
      return { kind: "dense", shape: record.shape, data, record };
    }
    if (record.dtype === CactDType.Raw) {
      return {
        kind: "raw",
        shape: record.shape,
        data: bytes.subarray(record.offset, record.offset + record.byteLength),
        record,
      };
    }
    if (record.dtype === CactDType.CactusQuant) {
      invariant(record.shape.length === 2, "INVALID_CACT", `CQ tensor ${record.index} is not a matrix`);
      invariant(record.groupSize > 0 && (record.groupSize & (record.groupSize - 1)) === 0, "UNSUPPORTED_CACT", `CQ tensor ${record.index} has non-power-of-two group ${record.groupSize}`);
      invariant(record.bits === 2 || record.bits === 3 || record.bits === 4 || record.bits === 5, "UNSUPPORTED_CACT", `CQ tensor ${record.index} uses unsupported record width ${record.bits}`);
      const outputSize = record.shape[0] ?? 0;
      const inputSize = record.shape[1] ?? 0;
      const inputSizePadded = Math.ceil(inputSize / record.groupSize) * record.groupSize;
      const packedBits = record.bits === 5 ? 2 : record.bits;
      const rowByteLength = (inputSizePadded * packedBits) / 8;
      invariant(Number.isInteger(rowByteLength), "INVALID_CACT", `CQ tensor ${record.index} has a fractional packed row size`);
      const packedByteLength = outputSize * rowByteLength;
      const normCount = outputSize * (inputSizePadded / record.groupSize);
      invariant(record.byteLength === packedByteLength + normCount * 2, "INVALID_CACT", `CQ tensor ${record.index} has an inconsistent packed/norm length`);
      const packed = bytes.subarray(record.offset, record.offset + packedByteLength);
      const normOffset = record.offset + packedByteLength;
      const norms = new DataView(bytes.buffer, bytes.byteOffset + normOffset, normCount * 2);
      return {
        kind: "cq",
        shape: [outputSize, inputSize],
        outputSize,
        inputSize,
        inputSizePadded,
        groupSize: record.groupSize,
        bits: record.bits,
        rowByteLength,
        packed,
        norms,
        codebooks,
        record,
      };
    }
    throw new NeedleError("UNSUPPORTED_CACT", `Tensor ${record.index} uses unknown dtype ${dtypeName(record.dtype)}`);
  });

  let cursor = 0;
  const take = (name: string): CactTensor => {
    const tensor = tensors[cursor++];
    invariant(tensor !== undefined, "INVALID_CACT", `Archive ended before ${name}`);
    return tensor;
  };

  const embedding = cq(take("embedding"), "embedding");
  invariant(embedding.outputSize === geometry.vocabularySize && embedding.inputSize === geometry.modelDimension, "INVALID_CACT", "Embedding shape does not match model geometry");

  const layers: CactLayer[] = [];
  for (let layer = 0; layer < geometry.numberOfLayers; layer++) {
    const prefix = `layer ${layer}`;
    const normInput = dense(take(`${prefix} input norm`), `${prefix} input norm`);
    const queryProjection = cq(take(`${prefix} query projection`), `${prefix} query projection`);
    const keyProjection = cq(take(`${prefix} key projection`), `${prefix} key projection`);
    const valueProjection = cq(take(`${prefix} value projection`), `${prefix} value projection`);
    const queryNorm = dense(take(`${prefix} query norm`), `${prefix} query norm`);
    const keyNorm = dense(take(`${prefix} key norm`), `${prefix} key norm`);
    const gateProjection = cq(take(`${prefix} gate projection`), `${prefix} gate projection`);
    const outputProjection = cq(take(`${prefix} output projection`), `${prefix} output projection`);
    const postAttentionNorm = dense(take(`${prefix} post-attention norm`), `${prefix} post-attention norm`);
    const attentionGateTensor = dense(take(`${prefix} attention gate`), `${prefix} attention gate`);
    const preHadamardNorm = dense(take(`${prefix} pre-Hadamard norm`), `${prefix} pre-Hadamard norm`);
    const hadamardD1 = dense(take(`${prefix} Hadamard d1`), `${prefix} Hadamard d1`);
    const hadamardD2 = dense(take(`${prefix} Hadamard d2`), `${prefix} Hadamard d2`);
    const hadamardD3 = dense(take(`${prefix} Hadamard d3`), `${prefix} Hadamard d3`);
    layers.push({
      normInput,
      queryProjection,
      keyProjection,
      valueProjection,
      queryNorm,
      keyNorm,
      gateProjection,
      outputProjection,
      postAttentionNorm,
      attentionGate: attentionGateTensor[0] ?? 0,
      preHadamardNorm,
      hadamardD1,
      hadamardD2,
      hadamardD3,
    });
  }

  const mhcAPre = dense(take("mHC a_pre"), "mHC a_pre");
  const mhcAPost = dense(take("mHC a_post"), "mHC a_post");
  const mhcAResidual = dense(take("mHC a_res"), "mHC a_res");
  const mhcBPre = dense(take("mHC b_pre"), "mHC b_pre");
  const mhcBPost = dense(take("mHC b_post"), "mHC b_post");
  const mhcBResidual = dense(take("mHC b_res"), "mHC b_res");
  const mhcPhiPre = cq(take("mHC phi_pre"), "mHC phi_pre");
  const mhcPhiPost = cq(take("mHC phi_post"), "mHC phi_post");
  const mhcPhiResidual = cq(take("mHC phi_res"), "mHC phi_res");

  const engrams: CactEngramSite[] = [];
  for (let site = 0; site < geometry.engramLayers.length; site++) {
    engrams.push({
      tables: cq(take(`engram ${site} tables`), `engram ${site} tables`),
      keyProjection: cq(take(`engram ${site} key projection`), `engram ${site} key projection`),
      valueProjection: cq(take(`engram ${site} value projection`), `engram ${site} value projection`),
      taps: dense(take(`engram ${site} taps`), `engram ${site} taps`),
    });
  }
  const finalNorm = dense(take("final norm"), "final norm");

  const heads = new Map<ProbeHeadName, CactProbeHead>();
  const next = tensors[cursor];
  if (next?.kind === "dense") {
    const manifest = next.data;
    cursor++;
    for (const rawCode of manifest) {
      const code = Math.round(rawCode);
      const name: ProbeHeadName | undefined = code === 1 ? "contrastive" : code === 2 ? "confidence" : undefined;
      invariant(name !== undefined, "UNSUPPORTED_CACT", `Unknown probe-head manifest code ${code}`);
      const probesTensor = take(`${name} probes`);
      const projectionTensor = take(`${name} projection`);
      const biasTensor = take(`${name} bias`);
      const probes = dense(probesTensor, `${name} probes`);
      const projection = dense(projectionTensor, `${name} projection`);
      const bias = dense(biasTensor, `${name} bias`);
      const probeCount = probesTensor.shape[0] ?? 0;
      const outputSize = projectionTensor.shape[0] ?? bias.length;
      invariant(probeCount > 0 && probes.length === probeCount * geometry.modelDimension, "INVALID_CACT", `${name} probe shape is inconsistent`);
      invariant(projection.length === outputSize * probeCount * geometry.modelDimension, "INVALID_CACT", `${name} projection shape is inconsistent`);
      heads.set(name, { name, probes, probeCount, projection, outputSize, bias });
    }
  }

  let tokenizerBlob: Uint8Array | undefined;
  while (cursor < tensors.length) {
    const tensor = take("trailing tensor");
    if (tensor.kind === "raw") {
      invariant(tokenizerBlob === undefined, "INVALID_CACT", "Archive contains more than one RAW tokenizer tensor");
      tokenizerBlob = tensor.data;
    } else {
      throw new NeedleError("UNSUPPORTED_CACT", `Unexpected trailing tensor ${tensor.record.index}`);
    }
  }
  invariant(tokenizerBlob !== undefined, "INVALID_CACT", "Archive does not contain its RAW SentencePiece tokenizer");

  return {
    bytes,
    geometry,
    numberOfTensors,
    codebook,
    codebooks,
    records,
    tensors,
    embedding,
    layers,
    mhcAPre,
    mhcAPost,
    mhcAResidual,
    mhcBPre,
    mhcBPost,
    mhcBResidual,
    mhcPhiPre,
    mhcPhiPost,
    mhcPhiResidual,
    engrams,
    finalNorm,
    heads,
    tokenizerBlob,
  };
}

function dtypeName(dtype: number): string {
  return ({ 1: "FP16", 2: "FP32", 3: "CQ", 4: "RAW" } as Record<number, string>)[dtype] ?? String(dtype);
}
