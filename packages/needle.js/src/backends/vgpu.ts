/// <reference types="@webgpu/types" preserve="true" />
import type { Compute, Gpu, StorageBuffer } from "vgpu";
import { invariant, NeedleError } from "../errors.js";
import type { CactWeights, CqMatrix } from "../model/cact.js";
import type { InferenceBackend, MatrixRowRange } from "./backend.js";
import { dequantizeCqRow, prepareCqActivation } from "./cq.js";
import { CQ_MATVEC_WGSL, matrixParameters, webGpuMatrixData } from "./webgpu-kernel.js";

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

export interface VGPUBackendOptions {
  /** Reuse a caller-owned vgpu context. */
  readonly gpu?: Gpu;
  /** Adopt an existing WebGPU device using vgpu's supported adoption route. */
  readonly device?: GPUDevice;
  /** Options passed to vgpu's browser or Node `init`. */
  readonly init?: unknown;
  /** Force the `vgpu/node` entrypoint (auto-detected under Node by default). */
  readonly node?: boolean;
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
  readonly #output: StorageBuffer;
  readonly #parameters: StorageBuffer;
  readonly #codebook: StorageBuffer;
  readonly #matrixCache = new WeakMap<CqMatrix, VgpuMatrix>();
  #disposed = false;

  private constructor(
    readonly weights: CactWeights,
    module: VgpuModule,
    gpu: Gpu,
    ownsGpu: boolean,
  ) {
    this.#module = module;
    this.gpu = gpu;
    this.#ownsGpu = ownsGpu;
    const quantized = weights.tensors.filter((tensor): tensor is CqMatrix => tensor.kind === "cq");
    const maximumInput = Math.max(1, ...quantized.map((matrix) => matrix.inputSizePadded));
    const maximumOutput = Math.max(1, ...quantized.map((matrix) => matrix.outputSize));
    this.#input = module.storage(gpu, align4(maximumInput * 4), "read");
    this.#output = module.storage(gpu, align4(maximumOutput * 4), "read-write");
    this.#parameters = module.storage(gpu, 32, "read");
    this.#codebook = module.storage(gpu, align4(weights.codebook.byteLength), "read");
    this.#codebook.write(ownedBuffer(weights.codebook));
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
      return new VGPUBackend(weights, module, gpu, ownsGpu);
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
    const prepared = prepareCqActivation(matrix, input);
    const gpuMatrix = this.#gpuMatrix(matrix);
    this.#input.write(ownedBuffer(prepared));
    this.#parameters.write(ownedBuffer(matrixParameters(matrix, rowStart, rowCount)));
    this.#compute.set({
      packed: gpuMatrix.packed,
      norms: gpuMatrix.norms,
      input: this.#input,
      codebook: this.#codebook,
      params: this.#parameters,
      output: this.#output,
    });
    this.#compute.dispatch(Math.ceil(rowCount / 64));
    const bytes = await this.#output.read();
    return new Float32Array(bytes, 0, rowCount).slice();
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

  #gpuMatrix(matrix: CqMatrix): VgpuMatrix {
    const cached = this.#matrixCache.get(matrix);
    if (cached) return cached;
    const data = webGpuMatrixData(matrix);
    const packed = this.#module.storage(this.gpu, align4(data.packedWords.byteLength), "read");
    const norms = this.#module.storage(this.gpu, align4(data.norms.byteLength), "read");
    packed.write(ownedBuffer(data.packedWords));
    norms.write(ownedBuffer(data.norms));
    const result = { packed, norms };
    this.#matrixCache.set(matrix, result);
    return result;
  }
}

function align4(value: number): number {
  return Math.max(4, (value + 3) & ~3);
}

function ownedBuffer(data: ArrayBufferView): ArrayBuffer {
  return new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice().buffer;
}

export async function createVGPUBackend(
  weights: CactWeights,
  options: VGPUBackendOptions = {},
): Promise<VGPUBackend> {
  return VGPUBackend.create(weights, options);
}
