import { invariant, NeedleError } from "../errors.js";
import { embeddedModelBase64Chunks, embeddedModelSha256 } from "./embedded-model.js";

export const NEEDLE_2_REPOSITORY = "Cactus-Compute/needle2";
export const NEEDLE_2_REVISION = "98fbd955b0347e78059be0c253cc1ffa09b87bc7";
export const NEEDLE_2_SHA256 = "b43aabfcaf1a6db6acf488076eab71d823c08697c7af4521fc1d174b60ede5ba";
export const NEEDLE_2_BYTES = 13_737_807;
export const NEEDLE_2_URL = `https://huggingface.co/${NEEDLE_2_REPOSITORY}/resolve/${NEEDLE_2_REVISION}/needle2.cact`;

export interface WeightProgress {
  readonly loaded: number;
  readonly total?: number;
  readonly source: string;
  readonly cached: boolean;
}

export interface UrlWeightSource {
  readonly kind: "url";
  readonly url: string | URL;
  readonly sha256?: string;
  readonly byteLength?: number;
  readonly headers?: Readonly<Record<string, string>>;
  readonly cache?: boolean;
  readonly cacheDir?: string;
}

export interface FileWeightSource {
  readonly kind: "file";
  readonly path: string | URL;
  readonly sha256?: string;
}

export interface EmbeddedWeightSource {
  readonly kind: "embedded";
  readonly data?: ArrayBuffer | ArrayBufferView | string;
  readonly sha256?: string;
}

export type WeightSource =
  | "download"
  | "embedded"
  | string
  | URL
  | ArrayBuffer
  | ArrayBufferView
  | UrlWeightSource
  | FileWeightSource
  | EmbeddedWeightSource;

export interface LoadWeightsOptions {
  readonly onProgress?: (progress: WeightProgress) => void;
  readonly signal?: AbortSignal;
  readonly fetch?: typeof globalThis.fetch;
  readonly cache?: boolean;
  readonly cacheDir?: string;
}

const memoryCache = new Map<string, Uint8Array>();

function isView(value: unknown): value is ArrayBufferView {
  return ArrayBuffer.isView(value);
}

function copyView(value: ArrayBuffer | ArrayBufferView): Uint8Array {
  if (value instanceof ArrayBuffer) return new Uint8Array(value);
  return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
}

function isNodeRuntime(): boolean {
  return typeof process !== "undefined" && typeof process.versions?.node === "string";
}

function looksLikeUrl(value: string): boolean {
  return /^(?:https?:|data:|blob:|file:)/i.test(value);
}

function normalizeHash(hash: string | undefined): string | undefined {
  if (!hash) return undefined;
  const value = hash.toLowerCase().replace(/^sha256[:-]?/, "");
  invariant(
    /^[0-9a-f]{64}$/.test(value),
    "WEIGHTS_INTEGRITY",
    `Invalid SHA-256 value ${JSON.stringify(hash)}`,
  );
  return value;
}

export async function sha256(bytes: Uint8Array): Promise<string> {
  if (globalThis.crypto?.subtle) {
    // Copying avoids runtimes whose BufferSource typing rejects SharedArrayBuffer-backed views.
    const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes.slice().buffer);
    return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join(
      "",
    );
  }
  if (isNodeRuntime()) {
    const { createHash } = await import("node:crypto");
    return createHash("sha256").update(bytes).digest("hex");
  }
  throw new NeedleError("WEIGHTS_INTEGRITY", "This runtime has no SHA-256 implementation");
}

async function verify(
  bytes: Uint8Array,
  expectedHash?: string,
  expectedBytes?: number,
): Promise<Uint8Array> {
  if (expectedBytes !== undefined && bytes.byteLength !== expectedBytes) {
    throw new NeedleError(
      "WEIGHTS_INTEGRITY",
      `Model has ${bytes.byteLength} bytes; expected ${expectedBytes}`,
    );
  }
  const normalized = normalizeHash(expectedHash);
  if (normalized) {
    const actual = await sha256(bytes);
    if (actual !== normalized) {
      throw new NeedleError(
        "WEIGHTS_INTEGRITY",
        `Model SHA-256 is ${actual}; expected ${normalized}`,
      );
    }
  }
  return bytes;
}

function decodeBase64Chunks(chunks: readonly string[]): Uint8Array {
  if (typeof Buffer !== "undefined") {
    const parts = chunks.map((chunk) => Buffer.from(chunk, "base64"));
    const total = parts.reduce((sum, part) => sum + part.byteLength, 0);
    const result = new Uint8Array(total);
    let offset = 0;
    for (const part of parts) {
      result.set(part, offset);
      offset += part.byteLength;
    }
    return result;
  }
  const decoded = chunks.map((chunk) => globalThis.atob(chunk));
  const total = decoded.reduce((sum, part) => sum + part.length, 0);
  const result = new Uint8Array(total);
  let offset = 0;
  for (const part of decoded) {
    for (let index = 0; index < part.length; index++) result[offset++] = part.charCodeAt(index);
  }
  return result;
}

export function hasEmbeddedWeights(): boolean {
  return embeddedModelBase64Chunks !== null && embeddedModelBase64Chunks.length > 0;
}

export function getEmbeddedWeights(): Uint8Array {
  if (!embeddedModelBase64Chunks) {
    throw new NeedleError(
      "WEIGHTS_NOT_FOUND",
      'This build does not contain embedded weights. Run `bun run embed:model /path/to/needle2.cact` before building, pass bytes explicitly, or use source: "download".',
    );
  }
  const key = `embedded:${embeddedModelSha256 ?? "unversioned"}`;
  const cached = memoryCache.get(key);
  if (cached) return cached;
  const bytes = decodeBase64Chunks(embeddedModelBase64Chunks);
  memoryCache.set(key, bytes);
  return bytes;
}

async function readFileSource(
  source: FileWeightSource,
  options: LoadWeightsOptions,
): Promise<Uint8Array> {
  if (!isNodeRuntime())
    throw new NeedleError(
      "WEIGHTS_NOT_FOUND",
      "File weight sources are only available in Node.js/Bun",
    );
  const { readFile } = await import("node:fs/promises");
  try {
    const data = await readFile(source.path);
    const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    options.onProgress?.({
      loaded: bytes.byteLength,
      total: bytes.byteLength,
      source: String(source.path),
      cached: true,
    });
    return verify(bytes, source.sha256);
  } catch (cause) {
    if (cause instanceof NeedleError) throw cause;
    throw new NeedleError(
      "WEIGHTS_NOT_FOUND",
      `Unable to read model weights from ${String(source.path)}`,
      { cause },
    );
  }
}

async function defaultNodeCachePath(
  source: UrlWeightSource,
  options: LoadWeightsOptions,
): Promise<string> {
  const [{ homedir }, path] = await Promise.all([import("node:os"), import("node:path")]);
  const root =
    source.cacheDir ?? options.cacheDir ?? path.join(homedir(), ".cache", "needle.js", "models");
  const key =
    normalizeHash(source.sha256) ?? (await sha256(new TextEncoder().encode(String(source.url))));
  return path.join(root, `${key}.cact`);
}

async function readNodeCache(
  path: string,
  source: UrlWeightSource,
  options: LoadWeightsOptions,
): Promise<Uint8Array | undefined> {
  try {
    const { readFile } = await import("node:fs/promises");
    const data = await readFile(path);
    const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    await verify(bytes, source.sha256, source.byteLength);
    options.onProgress?.({
      loaded: bytes.byteLength,
      total: bytes.byteLength,
      source: String(source.url),
      cached: true,
    });
    return bytes;
  } catch {
    // Remove stale/truncated entries so a subsequent atomic rename can repair
    // the cache (notably on Windows, where rename does not replace a target).
    try {
      const { unlink } = await import("node:fs/promises");
      await unlink(path);
    } catch {
      // A missing or read-only cache is simply a miss.
    }
    return undefined;
  }
}

async function writeNodeCache(path: string, bytes: Uint8Array): Promise<void> {
  const [{ mkdir, rename, writeFile }, pathModule] = await Promise.all([
    import("node:fs/promises"),
    import("node:path"),
  ]);
  await mkdir(pathModule.dirname(path), { recursive: true });
  const temporary = `${path}.${process.pid}.${Math.random().toString(16).slice(2)}.tmp`;
  try {
    await writeFile(temporary, bytes);
    await rename(temporary, path);
  } catch (cause) {
    // Another process may have won the same atomic cache race.
    try {
      const { unlink } = await import("node:fs/promises");
      await unlink(temporary);
    } catch {
      // Ignore cleanup failure.
    }
    const { stat } = await import("node:fs/promises");
    try {
      await stat(path);
    } catch {
      throw cause;
    }
  }
}

async function fetchBytes(
  source: UrlWeightSource,
  options: LoadWeightsOptions,
): Promise<Uint8Array> {
  const fetchImplementation = options.fetch ?? globalThis.fetch;
  if (!fetchImplementation)
    throw new NeedleError("WEIGHTS_NOT_FOUND", "This runtime has no fetch implementation");
  let response: Response;
  try {
    response = await fetchImplementation(source.url, {
      ...(source.headers === undefined ? {} : { headers: source.headers }),
      ...(options.signal === undefined ? {} : { signal: options.signal }),
    });
  } catch (cause) {
    throw new NeedleError(
      "WEIGHTS_NOT_FOUND",
      `Unable to download model from ${String(source.url)}`,
      { cause },
    );
  }
  if (!response.ok)
    throw new NeedleError(
      "WEIGHTS_NOT_FOUND",
      `Model download failed: HTTP ${response.status} ${response.statusText}`,
    );
  const totalHeader = Number(response.headers.get("content-length"));
  const total =
    source.byteLength ??
    (Number.isFinite(totalHeader) && totalHeader > 0 ? totalHeader : undefined);
  if (!response.body) {
    const bytes = new Uint8Array(await response.arrayBuffer());
    options.onProgress?.({
      loaded: bytes.byteLength,
      ...(total === undefined ? {} : { total }),
      source: String(source.url),
      cached: false,
    });
    return bytes;
  }

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value) {
      chunks.push(value);
      loaded += value.byteLength;
      options.onProgress?.({
        loaded,
        ...(total === undefined ? {} : { total }),
        source: String(source.url),
        cached: false,
      });
    }
  }
  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
}

async function readBrowserCache(source: UrlWeightSource): Promise<Uint8Array | undefined> {
  if (typeof caches === "undefined") return undefined;
  const cache = await caches.open("needle.js-models-v1");
  const response = await cache.match(String(source.url));
  return response ? new Uint8Array(await response.arrayBuffer()) : undefined;
}

async function writeBrowserCache(source: UrlWeightSource, bytes: Uint8Array): Promise<void> {
  if (typeof caches === "undefined") return;
  const cache = await caches.open("needle.js-models-v1");
  await cache.put(
    String(source.url),
    new Response(bytes.slice().buffer, {
      headers: { "content-type": "application/octet-stream" },
    }),
  );
}

async function loadUrlSource(
  source: UrlWeightSource,
  options: LoadWeightsOptions,
): Promise<Uint8Array> {
  const cacheEnabled = source.cache ?? options.cache ?? true;
  const cacheKey = `${String(source.url)}#${source.sha256 ?? ""}`;
  const memory = memoryCache.get(cacheKey);
  if (memory) {
    options.onProgress?.({
      loaded: memory.byteLength,
      total: source.byteLength ?? memory.byteLength,
      source: String(source.url),
      cached: true,
    });
    return memory;
  }

  let nodePath: string | undefined;
  if (cacheEnabled && isNodeRuntime() && !String(source.url).startsWith("data:")) {
    nodePath = await defaultNodeCachePath(source, options);
    const cached = await readNodeCache(nodePath, source, options);
    if (cached) {
      memoryCache.set(cacheKey, cached);
      return cached;
    }
  } else if (cacheEnabled && !isNodeRuntime()) {
    try {
      const cached = await readBrowserCache(source);
      if (cached) {
        await verify(cached, source.sha256, source.byteLength);
        options.onProgress?.({
          loaded: cached.byteLength,
          total: source.byteLength ?? cached.byteLength,
          source: String(source.url),
          cached: true,
        });
        memoryCache.set(cacheKey, cached);
        return cached;
      }
    } catch {
      // A corrupt or unavailable browser cache is treated as a miss.
    }
  }

  const downloaded = await fetchBytes(source, options);
  await verify(downloaded, source.sha256, source.byteLength);
  memoryCache.set(cacheKey, downloaded);
  if (cacheEnabled) {
    if (nodePath) await writeNodeCache(nodePath, downloaded);
    else if (!isNodeRuntime()) await writeBrowserCache(source, downloaded);
  }
  return downloaded;
}

function normalizeSource(
  source: WeightSource,
): UrlWeightSource | FileWeightSource | EmbeddedWeightSource | ArrayBuffer | ArrayBufferView {
  if (source === "download") {
    return {
      kind: "url",
      url: NEEDLE_2_URL,
      sha256: NEEDLE_2_SHA256,
      byteLength: NEEDLE_2_BYTES,
    };
  }
  if (source === "embedded") return { kind: "embedded" };
  if (source instanceof ArrayBuffer || isView(source)) return source;
  if (source instanceof URL) {
    return source.protocol === "file:"
      ? { kind: "file", path: source }
      : { kind: "url", url: source };
  }
  if (typeof source === "string") {
    return looksLikeUrl(source)
      ? source.startsWith("file:")
        ? { kind: "file", path: new URL(source) }
        : { kind: "url", url: source }
      : { kind: "file", path: source };
  }
  return source;
}

/** Resolves in-memory, embedded, file, or cached HTTP model weights. */
export async function loadWeights(
  source: WeightSource = hasEmbeddedWeights() ? "embedded" : "download",
  options: LoadWeightsOptions = {},
): Promise<Uint8Array> {
  const normalized = normalizeSource(source);
  if (normalized instanceof ArrayBuffer || isView(normalized)) return copyView(normalized);
  if (normalized.kind === "file") return readFileSource(normalized, options);
  if (normalized.kind === "url") return loadUrlSource(normalized, options);

  let bytes: Uint8Array;
  if (normalized.data === undefined) {
    bytes = getEmbeddedWeights();
  } else if (typeof normalized.data === "string") {
    bytes = decodeBase64Chunks([normalized.data]);
  } else {
    bytes = copyView(normalized.data);
  }
  await verify(bytes, normalized.sha256 ?? embeddedModelSha256 ?? undefined);
  options.onProgress?.({
    loaded: bytes.byteLength,
    total: bytes.byteLength,
    source: "embedded",
    cached: true,
  });
  return bytes;
}

/** Clears only this process's in-memory model cache (disk/CacheStorage remains). */
export function clearMemoryWeightCache(): void {
  memoryCache.clear();
}
