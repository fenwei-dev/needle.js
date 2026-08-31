import { describe, expect, test } from "bun:test";
import { CACT_TAG, parseCact } from "../src/model/cact.js";

describe(".cact parser", () => {
  test("rejects a truncated header", () => {
    expect(() => parseCact(new Uint8Array(12))).toThrow(/header/i);
  });

  test("rejects bad magic", () => {
    const bytes = new Uint8Array(120);
    expect(() => parseCact(bytes)).toThrow(/tag/i);
  });

  test("validates the shared codebook before reading the directory", () => {
    const bytes = new Uint8Array(120);
    const view = new DataView(bytes.buffer);
    view.setUint32(0, CACT_TAG, true);
    view.setUint32(4, 1, true);
    view.setUint32(8, 4, true);
    expect(() => parseCact(bytes)).toThrow(/codebook/i);
  });
});
