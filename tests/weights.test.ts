import { afterEach, describe, expect, test } from "bun:test";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import {
  clearMemoryWeightCache,
  loadWeights,
  sha256,
} from "../src/weights/source.js";

let temporary: string | undefined;
afterEach(async () => {
  clearMemoryWeightCache();
  if (temporary) await rm(temporary, { recursive: true, force: true });
  temporary = undefined;
});

describe("weight sources", () => {
  test("loads explicit embedded bytes and base64", async () => {
    const expected = new TextEncoder().encode("needle");
    expect(await loadWeights({ kind: "embedded", data: expected })).toEqual(expected);
    expect(await loadWeights({ kind: "embedded", data: Buffer.from(expected).toString("base64") })).toEqual(expected);
  });

  test("loads a local file", async () => {
    temporary = await mkdtemp(join(tmpdir(), "needle-js-"));
    const path = join(temporary, "tiny.cact");
    await writeFile(path, "weights");
    expect(new TextDecoder().decode(await loadWeights(path))).toBe("weights");
  });

  test("downloads, reports progress, and verifies SHA-256", async () => {
    const bytes = new TextEncoder().encode("downloaded model");
    const hash = await sha256(bytes);
    const progress: number[] = [];
    const loaded = await loadWeights({
      kind: "url",
      url: `data:application/octet-stream;base64,${Buffer.from(bytes).toString("base64")}`,
      sha256: hash,
      byteLength: bytes.length,
      cache: false,
    }, {
      onProgress: (event) => progress.push(event.loaded),
    });
    expect(loaded).toEqual(bytes);
    expect(progress.at(-1)).toBe(bytes.length);

    await expect(loadWeights({
      kind: "url",
      url: "data:application/octet-stream;base64,AA==",
      sha256: "0".repeat(64),
      cache: false,
    })).rejects.toMatchObject({ code: "WEIGHTS_INTEGRITY" });
  });
});
