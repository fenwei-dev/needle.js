import { describe, expect, test } from "bun:test";
import {
  NeedleTokenizer,
  TokenPieceType,
  type TokenizerMetadata,
} from "../src/model/tokenizer.js";

function tokenizer(): NeedleTokenizer {
  const pieces = [
    "<pad>", "</s>", "<s>", "<unk>", "<tool_call>", "▁", "h", "e", "l", "o",
    "he", "ll", "hell", "hello", "▁hello",
  ];
  const types = [
    TokenPieceType.Control,
    TokenPieceType.Control,
    TokenPieceType.Control,
    TokenPieceType.Unknown,
    TokenPieceType.UserDefined,
    ...Array.from({ length: 10 }, () => TokenPieceType.Normal),
  ];
  for (let byte = 0; byte < 256; byte++) {
    pieces.push(`<0x${byte.toString(16).toUpperCase().padStart(2, "0")}>`);
    types.push(TokenPieceType.Byte);
  }
  const scores = new Float32Array(pieces.length);
  scores[10] = 10;
  scores[11] = 9;
  scores[12] = 8;
  scores[13] = 7;
  scores[14] = 6;
  const metadata: TokenizerMetadata = {
    pieces,
    scores,
    types: Uint8Array.from(types),
    padTokenId: 0,
    eosTokenId: 1,
    bosTokenId: 2,
    unknownTokenId: 3,
    addDummyPrefix: true,
    byteFallback: true,
  };
  return new NeedleTokenizer(metadata);
}

describe("Needle tokenizer", () => {
  test("performs score-ordered BPE and strips the dummy prefix on decode", () => {
    const current = tokenizer();
    expect(current.encode("hello")).toEqual([14]);
    expect(current.decode(current.encode("hello hello"))).toBe("hello hello");
  });

  test("keeps user-defined chat markers atomic", () => {
    const current = tokenizer();
    const ids = current.encode("hello<tool_call>hello");
    expect(ids).toContain(4);
    expect(current.decode(ids)).toBe("hello<tool_call>hello");
  });

  test("round-trips unknown Unicode through byte fallback", () => {
    const current = tokenizer();
    const text = "hello ☃";
    expect(current.decode(current.encode(text))).toBe(text);
  });

  test("encodeRaw does not inject a mid-sequence space", () => {
    const current = tokenizer();
    expect(current.decode(current.encodeRaw("hello"))).toBe("hello");
    expect(current.encodeRaw("hello")).not.toEqual(current.encode("hello"));
  });
});
