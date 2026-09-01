/// <reference types="@webgpu/types" preserve="true" />
import type { Compute, Gpu, StorageBuffer } from "vgpu";
import { invariant, NeedleError } from "../errors.js";
import type { CactWeights, CqMatrix } from "../model/cact.js";
import type { InferenceBackend, MatrixRowRange, MatvecRequest } from "./backend.js";
import {
  cqMatvecPrepared,
  dequantizeCqRow,
  prepareCqActivation,
  prepareCqActivationCached,
} from "./cq.js";
import {
  CQ_MATVEC_WGSL,
  matrixParameters,
  normalizeMinimumGpuRows,
  webGpuMatrixData,
} from "./webgpu-kernel.js";

interface VgpuModule {
  readonly init: (options?: unknown) => Promise<Gpu>;
  readonly initFromDevice: (device: GPUDevice) => Promise<Gpu>;
  readonly compute: (gpu: Gpu, source: string, options?: Record<string, unknown>) => Compute;
  readonly storage: (gpu: Gpu, bytes: number, access?: "read" | "read-write") => StorageBuffer;
}

interface VgpuMatrix {
  readonly packed: StorageBuffer;
  readonly norms: StorageBuffer;
}

interface BatchRead {
  readonly index: number;
  readonly rowCount: number;
  readonly bytes: Promise<ArrayBuffer>;
}

export interface VGPUBackendOptions {
  /** Reuse a caller-owned vgpu context. */
  readonly gpu?: Gpu;
  /** Adopt an existing WebGPU device using vgpu's supported adoption route. */
  readonly device?: GPUDevice;
  /** Options passed to vgpu's browser or Node `init`. */
  readonly init?: unknown;
  /** Force the `vgpu/node` entrypoint (auto-detected under Node by default). */
  readonly node?: boolean;
  /**
   * Small matvecs run on CPU because submission/readback costs dominate.
   * Defaults to 1024 output rows. Set to 0 to force every matvec onto WebGPU.
   */
  readonly minimumGpuRows?: number;
}

/** CQ matvec acceleration implemented with vgpu storage and compute APIs. */
export class VGPUBackend implements InferenceBackend {
  readonly kind = "vgpu" as const;
  readonly name = "vgpu / WebGPU";
  readonly gpu: Gpu;

  readonly #module: VgpuModule;
  readonly #ownsGpu: boolean;
  readonly #compute: Compute;
  readonly #input: StorageBuffer;
  readonly #outputs = new Map<number, StorageBuffer>();
  readonly #parameters: StorageBuffer;
  readonly #codebook: StorageBuffer;
  readonly #matrixCache = new WeakMap<CqMatrix, VgpuMatrix>();
  readonly #minimumGpuRows: number;
  #disposed = false;

  private constructor(
    readonly weights: CactWeights,
    module: VgpuModule,
    gpu: Gpu,
    ownsGpu: boolean,
    minimumGpuRows: number,
  ) {
    this.#module = module;
    this.gpu = gpu;
    this.#ownsGpu = ownsGpu;
    this.#minimumGpuRows = minimumGpuRows;
    const quantized = weights.tensors.filter((tensor): tensor is CqMatrix => tensor.kind === "cq");
    const maximumInput = Math.max(1, ...quantized.map((matrix) => matrix.inputSizePadded));
    this.#input = module.storage(gpu, align4(maximumInput * 4), "read");
    this.#parameters = module.storage(gpu, 32, "read");
    this.#codebook = module.storage(gpu, align4(weights.codebook.byteLength), "read");
    this.#codebook.write(bufferSource(weights.codebook));
    this.#compute = module.compute(gpu, CQ_MATVEC_WGSL, { label: "needle.vgpu.cq-matvec" });
  }

  static async create(
    weights: CactWeights,
    options: VGPUBackendOptions = {},
  ): Promise<VGPUBackend> {
    try {
      const useNode =
        options.node ??
        (typeof process !== "undefined" && typeof process.versions?.node === "string");
      const imported = useNode ? await import("vgpu/node") : await import("vgpu");
      const module = imported as unknown as VgpuModule;
      let gpu: Gpu;
      let ownsGpu = false;
      if (options.gpu) {
        gpu = options.gpu;
      } else if (options.device) {
        gpu = await module.initFromDevice(options.device);
        ownsGpu = true;
      } else {
        gpu = await module.init(options.init);
        ownsGpu = true;
      }
      return new VGPUBackend(
        weights,
        module,
        gpu,
        ownsGpu,
        normalizeMinimumGpuRows(options.minimumGpuRows),
      );
    } catch (cause) {
      if (cause instanceof NeedleError) throw cause;
      throw new NeedleError("WEBGPU_UNAVAILABLE", "Unable to initialize the vgpu backend", {
        cause,
      });
    }
  }

  async matvec(
    matrix: CqMatrix,
    input: Float32Array,
    range: MatrixRowRange = {},
  ): Promise<Float32Array> {
    invariant(!this.#disposed, "BACKEND_UNAVAILABLE", "vgpu backend has been disposed");
    const rowStart = range.rowStart ?? 0;
    const rowCount = range.rowCount ?? matrix.outputSize - rowStart;
    invariant(
      rowStart >= 0 && rowCount >= 0 && rowStart + rowCount <= matrix.outputSize,
      "INVALID_CACT",
      "Invalid vgpu matvec row range",
    );
    if (rowCount === 0) return new Float32Array();
    const prepared = prepareCqActivation(matrix, input);
    // Needle's 4–512-row projections cannot amortize a WebGPU submission and
    // synchronous readback. The 8,192-row vocabulary projection can.
    if (rowCount < this.#minimumGpuRows)
      return cqMatvecPrepared(matrix, prepared, rowStart, rowCount);
    const gpuMatrix = this.#gpuMatrix(matrix);
    const output = this.#output(rowCount);
    this.#input.write(bufferSource(prepared));
    this.#parameters.write(bufferSource(matrixParameters(matrix, rowStart, rowCount)));
    this.#compute.set({
      packed: gpuMatrix.packed,
      norms: gpuMatrix.norms,
      input: this.#input,
      codebook: this.#codebook,
      params: this.#parameters,
      output,
    });
    this.#compute.dispatch(rowCount);
    const bytes = await output.read();
    return new Float32Array(bytes, 0, rowCount);
  }

  async matvecBatch(requests: readonly MatvecRequest[]): Promise<readonly Float32Array[]> {
    invariant(!this.#disposed, "BACKEND_UNAVAILABLE", "vgpu backend has been disposed");
    if (requests.length === 0) return [];

    const results = new Array<Float32Array>(requests.length);
    const preparedCache = new Map<Float32Array, Map<string, Float32Array>>();
    const reads: BatchRead[] = [];
    let uploadedInput: Float32Array | undefined;

    for (let index = 0; index < requests.length; index++) {
      const request = requests[index];
      invariant(request, "INVALID_CACT", `Missing vgpu batch request ${index}`);
      const { matrix, input, range = {} } = request;
      const rowStart = range.rowStart ?? 0;
      const rowCount = range.rowCount ?? matrix.outputSize - rowStart;
      invariant(
        rowStart >= 0 && rowCount >= 0 && rowStart + rowCount <= matrix.outputSize,
        "INVALID_CACT",
        "Invalid vgpu batched matvec row range",
      );
      if (rowCount === 0) {
        results[index] = new Float32Array();
        continue;
      }

      const prepared = prepareCqActivationCached(preparedCache, matrix, input);
      if (rowCount < this.#minimumGpuRows) {
        results[index] = cqMatvecPrepared(matrix, prepared, rowStart, rowCount);
        continue;
      }

      const gpuMatrix = this.#gpuMatrix(matrix);
      const output = this.#output(rowCount);
      if (uploadedInput !== prepared) {
        this.#input.write(bufferSource(prepared));
        uploadedInput = prepared;
      }
      this.#parameters.write(bufferSource(matrixParameters(matrix, rowStart, rowCount)));
      this.#compute.set({
        packed: gpuMatrix.packed,
        norms: gpuMatrix.norms,
        input: this.#input,
        codebook: this.#codebook,
        params: this.#parameters,
        output,
      });
      this.#compute.dispatch(rowCount);
      // read() submits its copy before yielding. Queue ordering preserves this
      // result even when the next request reuses the same output-size buffer.
      reads.push({ index, rowCount, bytes: output.read() });
    }

    const buffers = await Promise.all(reads.map((read) => read.bytes));
    for (let index = 0; index < reads.length; index++) {
      const read = reads[index];
      const bytes = buffers[index];
      invariant(read && bytes, "BACKEND_UNAVAILABLE", "vgpu batch readback is missing");
      results[read.index] = new Float32Array(bytes, 0, read.rowCount);
    }
    return results;
  }

  row(matrix: CqMatrix, row: number, output?: Float32Array): Float32Array {
    return dequantizeCqRow(matrix, row, output);
  }

  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    // vgpu resources are registered with their Gpu lifetime. Dispose only a
    // context created by this backend; a supplied context remains caller-owned.
    if (this.#ownsGpu) this.gpu.dispose();
  }

  #output(rowCount: number): StorageBuffer {
    let output = this.#outputs.get(rowCount);
    if (!output) {
      output = this.#module.storage(this.gpu, align4(rowCount * 4), "read-write");
      this.#outputs.set(rowCount, output);
    }
    return output;
  }

  #gpuMatrix(matrix: CqMatrix): VgpuMatrix {
    const cached = this.#matrixCache.get(matrix);
    if (cached) return cached;
    const data = webGpuMatrixData(matrix);
    const packed = this.#module.storage(this.gpu, align4(data.packedWords.byteLength), "read");
    const norms = this.#module.storage(this.gpu, align4(data.norms.byteLength), "read");
    packed.write(bufferSource(data.packedWords));
    norms.write(bufferSource(data.norms));
    const result = { packed, norms };
    this.#matrixCache.set(matrix, result);
    return result;
  }
}

function align4(value: number): number {
  return Math.max(4, (value + 3) & ~3);
}

function bufferSource(data: ArrayBufferView): BufferSource {
  if (data.buffer instanceof ArrayBuffer) return data as ArrayBufferView<ArrayBuffer>;
  return new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice();
}

export async function createVGPUBackend(
  weights: CactWeights,
  options: VGPUBackendOptions = {},
): Promise<VGPUBackend> {
  return VGPUBackend.create(weights, options);
}
