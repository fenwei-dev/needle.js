import { invariant } from "../errors.js";

export const PAD_TOKEN_ID = 0;
export const EOS_TOKEN_ID = 1;
export const BOS_TOKEN_ID = 2;
export const UNKNOWN_TOKEN_ID = 3;

export const CHAT_MARKERS = {
  imStart: "<|im_start|>",
  imEnd: "<|im_end|>",
  thinkStart: "<think>",
  thinkEnd: "</think>",
  toolsStart: "<tools>",
  toolsEnd: "</tools>",
  toolCallStart: "<tool_call>",
  toolCallEnd: "</tool_call>",
  toolResultStart: "<tool_result>",
  toolResultEnd: "</tool_result>",
} as const;

export enum TokenPieceType {
  Normal = 0,
  Unknown = 1,
  Control = 2,
  UserDefined = 3,
  Byte = 4,
}

const META_SPACE = "▁";
const encoder = new TextEncoder();
const decoder = new TextDecoder("utf-8", { fatal: false });

export interface TokenizerMetadata {
  readonly pieces: readonly string[];
  readonly scores: Float32Array;
  readonly types: Uint8Array;
  readonly padTokenId: number;
  readonly eosTokenId: number;
  readonly bosTokenId: number;
  readonly unknownTokenId: number;
  readonly addDummyPrefix: boolean;
  readonly byteFallback: boolean;
}

function readMetadata(blob: Uint8Array): TokenizerMetadata {
  invariant(
    blob.byteLength >= 24,
    "INVALID_CACT",
    "Tokenizer blob is shorter than its 24-byte header",
  );
  const view = new DataView(blob.buffer, blob.byteOffset, blob.byteLength);
  const count = view.getUint32(0, true);
  const padTokenId = view.getUint32(4, true);
  const eosTokenId = view.getUint32(8, true);
  const bosTokenId = view.getUint32(12, true);
  const unknownTokenId = view.getUint32(16, true);
  const addDummyPrefix = view.getUint8(20) !== 0;
  const byteFallback = view.getUint8(21) !== 0;
  const pieces: string[] = [];
  const scores = new Float32Array(count);
  const types = new Uint8Array(count);
  let offset = 24;
  for (let id = 0; id < count; id++) {
    invariant(offset + 7 <= blob.byteLength, "INVALID_CACT", `Tokenizer record ${id} is truncated`);
    scores[id] = view.getFloat32(offset, true);
    types[id] = view.getUint8(offset + 4);
    const byteLength = view.getUint16(offset + 5, true);
    offset += 7;
    invariant(
      offset + byteLength <= blob.byteLength,
      "INVALID_CACT",
      `Tokenizer piece ${id} is truncated`,
    );
    pieces.push(decoder.decode(blob.subarray(offset, offset + byteLength)));
    offset += byteLength;
  }
  invariant(
    offset === blob.byteLength,
    "INVALID_CACT",
    `Tokenizer blob has ${blob.byteLength - offset} unexplained trailing bytes`,
  );
  return {
    pieces,
    scores,
    types,
    padTokenId,
    eosTokenId,
    bosTokenId,
    unknownTokenId,
    addDummyPrefix,
    byteFallback,
  };
}

/** Reference-compatible tokenizer embedded in a Needle 2 `.cact` archive. */
export class NeedleTokenizer {
  readonly pieces: readonly string[];
  readonly scores: Float32Array;
  readonly types: Uint8Array;
  readonly padTokenId: number;
  readonly eosTokenId: number;
  readonly bosTokenId: number;
  readonly unknownTokenId: number;
  readonly addDummyPrefix: boolean;
  readonly byteFallback: boolean;
  readonly vocabularySize: number;

  readonly #pieceToId: Map<string, number>;
  readonly #byteToId: Int32Array;
  readonly #markers: readonly string[];
  readonly #pieceBytes: (Uint8Array | undefined)[];

  constructor(blobOrMetadata: Uint8Array | TokenizerMetadata) {
    const metadata =
      blobOrMetadata instanceof Uint8Array ? readMetadata(blobOrMetadata) : blobOrMetadata;
    this.pieces = metadata.pieces;
    this.scores = metadata.scores;
    this.types = metadata.types;
    this.padTokenId = metadata.padTokenId;
    this.eosTokenId = metadata.eosTokenId;
    this.bosTokenId = metadata.bosTokenId;
    this.unknownTokenId = metadata.unknownTokenId;
    this.addDummyPrefix = metadata.addDummyPrefix;
    this.byteFallback = metadata.byteFallback;
    this.vocabularySize = this.pieces.length;
    invariant(
      this.scores.length === this.vocabularySize && this.types.length === this.vocabularySize,
      "INVALID_CACT",
      "Tokenizer piece, score, and type counts differ",
    );

    this.#pieceToId = new Map(this.pieces.map((piece, id) => [piece, id]));
    this.#byteToId = new Int32Array(256);
    this.#byteToId.fill(-1);
    const markers: string[] = [];
    for (let id = 0; id < this.vocabularySize; id++) {
      const piece = this.pieces[id] ?? "";
      const type = this.types[id];
      if (type === TokenPieceType.Byte) {
        const match = /^<0x([0-9A-Fa-f]{2})>$/.exec(piece);
        if (match) this.#byteToId[Number.parseInt(match[1] ?? "00", 16)] = id;
      } else if (type === TokenPieceType.UserDefined) {
        markers.push(piece);
      }
    }
    markers.sort((left, right) => right.length - left.length);
    this.#markers = markers;
    this.#pieceBytes = new Array(this.vocabularySize);
  }

  /** SentencePiece BPE encoding, including the model's dummy-prefix behavior. */
  encode(text: string): number[] {
    return this.#encode(text, this.addDummyPrefix);
  }

  /** Encodes a fragment that occurs in the middle of an existing sequence. */
  encodeRaw(text: string): number[] {
    return this.#encode(text, false);
  }

  #encode(text: string, addDummyPrefix: boolean): number[] {
    if (text.length === 0) return [];
    let escaped = text.replaceAll(" ", META_SPACE);
    if (addDummyPrefix) escaped = META_SPACE + escaped;

    const ids: number[] = [];
    let segmentStart = 0;
    let offset = 0;
    while (offset < escaped.length) {
      let marker: string | undefined;
      for (const candidate of this.#markers) {
        if (escaped.startsWith(candidate, offset)) {
          marker = candidate;
          break;
        }
      }
      if (marker === undefined) {
        const point = escaped.codePointAt(offset) ?? 0;
        offset += point > 0xffff ? 2 : 1;
        continue;
      }
      ids.push(...this.#bpe(escaped.slice(segmentStart, offset)));
      const id = this.#pieceToId.get(marker);
      if (id !== undefined) ids.push(id);
      offset += marker.length;
      segmentStart = offset;
    }
    ids.push(...this.#bpe(escaped.slice(segmentStart)));
    return ids;
  }

  #bpe(segment: string): number[] {
    if (segment.length === 0) return [];
    const symbols = Array.from(segment);
    while (symbols.length > 1) {
      let bestScore = Number.NEGATIVE_INFINITY;
      let bestIndex = -1;
      for (let index = 0; index + 1 < symbols.length; index++) {
        const merged = (symbols[index] ?? "") + (symbols[index + 1] ?? "");
        const id = this.#pieceToId.get(merged);
        if (id !== undefined) {
          const score = this.scores[id] ?? Number.NEGATIVE_INFINITY;
          // SentencePiece's reference walk keeps the first merge on score ties.
          if (bestIndex < 0 || score > bestScore) {
            bestScore = score;
            bestIndex = index;
          }
        }
      }
      if (bestIndex < 0) break;
      symbols.splice(bestIndex, 2, (symbols[bestIndex] ?? "") + (symbols[bestIndex + 1] ?? ""));
    }

    const ids: number[] = [];
    for (const symbol of symbols) {
      const id = this.#pieceToId.get(symbol);
      if (id !== undefined) {
        ids.push(id);
      } else if (this.byteFallback) {
        for (const byte of encoder.encode(symbol)) {
          const byteId = this.#byteToId[byte] ?? -1;
          ids.push(byteId >= 0 ? byteId : this.unknownTokenId);
        }
      } else {
        ids.push(this.unknownTokenId);
      }
    }
    return ids;
  }

  /** Decodes one or more token IDs into UTF-8 text. */
  decode(ids: Iterable<number>): string {
    const chunks: Uint8Array[] = [];
    let total = 0;
    for (const id of ids) {
      const bytes = this.pieceBytes(id);
      if (bytes.byteLength > 0) {
        chunks.push(bytes);
        total += bytes.byteLength;
      }
    }
    const joined = new Uint8Array(total);
    let offset = 0;
    for (const chunk of chunks) {
      joined.set(chunk, offset);
      offset += chunk.byteLength;
    }
    let text = decoder.decode(joined);
    if (this.addDummyPrefix && text.startsWith(" ")) text = text.slice(1);
    return text;
  }

  /**
   * Returns the exact decoded bytes represented by one token. Byte-fallback
   * pieces may intentionally be invalid UTF-8 until adjacent pieces join them.
   */
  pieceBytes(id: number): Uint8Array {
    invariant(
      Number.isInteger(id) && id >= 0 && id < this.vocabularySize,
      "INVALID_CACT",
      `Token ID ${id} is outside the vocabulary`,
    );
    const cached = this.#pieceBytes[id];
    if (cached !== undefined) return cached;
    const type = this.types[id];
    const piece = this.pieces[id] ?? "";
    let result: Uint8Array;
    if (type === TokenPieceType.Control || type === TokenPieceType.Unknown) {
      result = new Uint8Array(0);
    } else if (type === TokenPieceType.Byte) {
      const match = /^<0x([0-9A-Fa-f]{2})>$/.exec(piece);
      result = match ? Uint8Array.of(Number.parseInt(match[1] ?? "00", 16)) : new Uint8Array(0);
    } else {
      result = encoder.encode(piece.replaceAll(META_SPACE, " "));
    }
    this.#pieceBytes[id] = result;
    return result;
  }

  pieceText(id: number): string {
    return decoder.decode(this.pieceBytes(id));
  }

  idForPiece(piece: string): number | undefined {
    return this.#pieceToId.get(piece);
  }

  isNormalOrByte(id: number): boolean {
    const type = this.types[id];
    return type === TokenPieceType.Normal || type === TokenPieceType.Byte;
  }
}

export { readMetadata as parseTokenizerMetadata };
