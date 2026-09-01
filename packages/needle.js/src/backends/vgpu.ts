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
  batchMatrixParameters,
  CQ_MATVEC_WGSL,
  matrixParameters,
  normalizeMinimumGpuRows,
  webGpuBatchMatrixData,
  webGpuMatrixData,
} from "./webgpu-kernel.js";

const MAX_BATCH_JOBS = 4;

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

interface VgpuBatchMatrix extends VgpuMatrix {
  readonly entries: readonly { readonly packedWordOffset: number; readonly normOffset: number }[];
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
  readonly #maximumInputBytes: number;
  readonly #maximumOutputBytes: number;
  readonly #matrixCache = new WeakMap<CqMatrix, VgpuMatrix>();
  readonly #batchMatrixCache = new Map<string, VgpuBatchMatrix>();
  readonly #minimumGpuRows: number;
  #batchCompute: Compute | undefined;
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
    const maximumOutput = Math.max(1, ...quantized.map((matrix) => matrix.outputSize));
    this.#maximumInputBytes = align4(maximumInput * 4 * MAX_BATCH_JOBS);
    this.#maximumOutputBytes = align4(maximumOutput * 4 * MAX_BATCH_JOBS);
    this.#input = module.storage(gpu, this.#maximumInputBytes, "read");
    this.#parameters = module.storage(gpu, 16_384, "read");
    this.#codebook = module.storage(gpu, align4(weights.codebook.byteLength), "read");
    this.#codebook.write(bufferSource(weights.codebook));
    this.#compute = module.compute(gpu, CQ_MATVEC_WGSL, {
      label: "needle.vgpu.cq-matvec",
      entry: "main",
    });
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
    const jobs: Array<{
      index: number;
      matrix: CqMatrix;
      prepared: Float32Array;
      rowStart: number;
      rowCount: number;
      outputStart: number;
    }> = [];
    let outputCount = 0;

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
      jobs.push({ index, matrix, prepared, rowStart, rowCount, outputStart: outputCount });
      outputCount += rowCount;
    }

    if (jobs.length === 0) return results;
    const inputOffsets = new Map<Float32Array, number>();
    let inputCount = 0;
    for (const job of jobs) {
      if (!inputOffsets.has(job.prepared)) {
        inputOffsets.set(job.prepared, inputCount);
        inputCount += job.prepared.length;
      }
    }
    invariant(
      inputCount * 4 <= this.#maximumInputBytes && outputCount * 4 <= this.#maximumOutputBytes,
      "BACKEND_UNAVAILABLE",
      "vgpu matvec batch exceeds its input or output arena",
    );

    const batchInput = new Float32Array(inputCount);
    for (const [prepared, offset] of inputOffsets) batchInput.set(prepared, offset);
    const batchMatrix = this.#gpuBatchMatrix(jobs.map((job) => job.matrix));
    const parameters = batchMatrixParameters(
      jobs.map((job, index) => {
        const entry = batchMatrix.entries[index];
        const inputOffset = inputOffsets.get(job.prepared);
        invariant(entry && inputOffset !== undefined, "BACKEND_UNAVAILABLE", "Invalid GPU batch");
        return {
          matrix: job.matrix,
          rowStart: job.rowStart,
          rowCount: job.rowCount,
          outputStart: job.outputStart,
          inputOffset,
          packedWordOffset: entry.packedWordOffset,
          normOffset: entry.normOffset,
        };
      }),
    );
    invariant(parameters.byteLength <= 16_384, "BACKEND_UNAVAILABLE", "GPU batch is too large");
    const output = this.#output(outputCount);
    this.#input.write(bufferSource(batchInput));
    this.#parameters.write(bufferSource(parameters));
    const batchCompute = this.#getBatchCompute();
    batchCompute.set({
      packed: batchMatrix.packed,
      norms: batchMatrix.norms,
      input: this.#input,
      codebook: this.#codebook,
      params: this.#parameters,
      output,
    });
    batchCompute.dispatch(outputCount);
    const bytes = await output.read();
    const combined = new Float32Array(bytes, 0, outputCount);
    for (const job of jobs) {
      results[job.index] = combined.slice(job.outputStart, job.outputStart + job.rowCount);
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

  #getBatchCompute(): Compute {
    if (!this.#batchCompute) {
      this.#batchCompute = this.#module.compute(this.gpu, CQ_MATVEC_WGSL, {
        label: "needle.vgpu.cq-matvec-batch",
        entry: "batch_main",
      });
    }
    return this.#batchCompute;
  }

  #output(rowCount: number): StorageBuffer {
    let output = this.#outputs.get(rowCount);
    if (!output) {
      output = this.#module.storage(this.gpu, align4(rowCount * 4), "read-write");
      this.#outputs.set(rowCount, output);
    }
    return output;
  }

  #gpuBatchMatrix(matrices: readonly CqMatrix[]): VgpuBatchMatrix {
    const key = matrices.map((matrix) => matrix.record.index).join(":");
    const cached = this.#batchMatrixCache.get(key);
    if (cached) return cached;
    const data = webGpuBatchMatrixData(matrices);
    const packed = this.#module.storage(this.gpu, align4(data.packedWords.byteLength), "read");
    const norms = this.#module.storage(this.gpu, align4(data.norms.byteLength), "read");
    packed.write(bufferSource(data.packedWords));
    norms.write(bufferSource(data.norms));
    const result = { packed, norms, entries: data.entries };
    this.#batchMatrixCache.set(key, result);
    return result;
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
