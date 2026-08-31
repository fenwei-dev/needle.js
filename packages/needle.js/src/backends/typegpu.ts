/// <reference types="@webgpu/types" preserve="true" />
import { type TgpuRoot, tgpu } from "typegpu";
import { invariant, NeedleError } from "../errors.js";
import type { CactWeights, CqMatrix } from "../model/cact.js";
import type { InferenceBackend, MatrixRowRange } from "./backend.js";
import { cqMatvecPrepared, dequantizeCqRow, prepareCqActivation } from "./cq.js";
import {
  CQ_MATVEC_WGSL,
  matrixParameters,
  normalizeMinimumGpuRows,
  webGpuMatrixData,
} from "./webgpu-kernel.js";

const MAP_READ = 0x0001;
const COPY_SRC = 0x0004;
const COPY_DST = 0x0008;
const STORAGE = 0x0080;

interface GpuMatrix {
  readonly packed: GPUBuffer;
  readonly norms: GPUBuffer;
  readonly bindGroup: GPUBindGroup;
}

export interface TypeGPUBackendOptions {
  /** Reuse an existing TypeGPU root. Its lifetime remains caller-owned. */
  readonly root?: TgpuRoot;
  /** Adopt a raw device through `tgpu.initFromDevice`. */
  readonly device?: GPUDevice;
  /** Forwarded to `tgpu.init` when neither root nor device is supplied. */
  readonly init?: Parameters<typeof tgpu.init>[0];
  /**
   * Small matvecs run on CPU because submission/readback costs dominate.
   * Defaults to 1024 output rows. Set to 0 to force every matvec onto WebGPU.
   */
  readonly minimumGpuRows?: number;
}

/** CQ matvec acceleration using a device initialized and owned through TypeGPU. */
export class TypeGPUBackend implements InferenceBackend {
  readonly kind = "typegpu" as const;
  readonly name = "TypeGPU / WebGPU";
  readonly root: TgpuRoot;
  readonly device: GPUDevice;

  readonly #ownsRoot: boolean;
  readonly #pipeline: GPUComputePipeline;
  readonly #input: GPUBuffer;
  readonly #output: GPUBuffer;
  readonly #parameters: GPUBuffer;
  readonly #codebook: GPUBuffer;
  readonly #staging: GPUBuffer;
  readonly #matrixCache = new WeakMap<CqMatrix, GpuMatrix>();
  readonly #allocatedMatrices: GpuMatrix[] = [];
  readonly #minimumGpuRows: number;
  #disposed = false;

  private constructor(
    readonly weights: CactWeights,
    root: TgpuRoot,
    ownsRoot: boolean,
    pipeline: GPUComputePipeline,
    minimumGpuRows: number,
  ) {
    this.root = root;
    this.device = root.device;
    this.#ownsRoot = ownsRoot;
    this.#pipeline = pipeline;
    this.#minimumGpuRows = minimumGpuRows;
    const quantized = weights.tensors.filter((tensor): tensor is CqMatrix => tensor.kind === "cq");
    const maximumInput = Math.max(1, ...quantized.map((matrix) => matrix.inputSizePadded));
    const maximumOutput = Math.max(1, ...quantized.map((matrix) => matrix.outputSize));
    this.#input = this.device.createBuffer({
      label: "needle.typegpu.input",
      size: align4(maximumInput * 4),
      usage: STORAGE | COPY_DST,
    });
    this.#output = this.device.createBuffer({
      label: "needle.typegpu.output",
      size: align4(maximumOutput * 4),
      usage: STORAGE | COPY_SRC,
    });
    this.#parameters = this.device.createBuffer({
      label: "needle.typegpu.parameters",
      size: 32,
      usage: STORAGE | COPY_DST,
    });
    this.#codebook = bufferWithData(
      this.device,
      "needle.typegpu.codebook",
      weights.codebook,
      STORAGE | COPY_DST,
    );
    this.#staging = this.device.createBuffer({
      label: "needle.typegpu.readback",
      size: align4(maximumOutput * 4),
      usage: MAP_READ | COPY_DST,
    });
  }

  static async create(
    weights: CactWeights,
    options: TypeGPUBackendOptions = {},
  ): Promise<TypeGPUBackend> {
    let root: TgpuRoot;
    let ownsRoot = false;
    try {
      if (options.root) {
        root = options.root;
      } else if (options.device) {
        root = tgpu.initFromDevice({ device: options.device });
        ownsRoot = true;
      } else {
        root = await tgpu.init(options.init);
        ownsRoot = true;
      }
      const shader = tgpu.resolve({ template: CQ_MATVEC_WGSL, externals: {} });
      const module = root.device.createShaderModule({
        label: "needle.typegpu.cq-matvec.shader",
        code: shader,
      });
      const descriptor: GPUComputePipelineDescriptor = {
        label: "needle.typegpu.cq-matvec.pipeline",
        layout: "auto",
        compute: { module, entryPoint: "main" },
      };
      const pipeline = root.device.createComputePipelineAsync
        ? await root.device.createComputePipelineAsync(descriptor)
        : root.device.createComputePipeline(descriptor);
      return new TypeGPUBackend(
        weights,
        root,
        ownsRoot,
        pipeline,
        normalizeMinimumGpuRows(options.minimumGpuRows),
      );
    } catch (cause) {
      if (cause instanceof NeedleError) throw cause;
      throw new NeedleError("WEBGPU_UNAVAILABLE", "Unable to initialize the TypeGPU backend", {
        cause,
      });
    }
  }

  async matvec(
    matrix: CqMatrix,
    input: Float32Array,
    range: MatrixRowRange = {},
  ): Promise<Float32Array> {
    invariant(!this.#disposed, "BACKEND_UNAVAILABLE", "TypeGPU backend has been disposed");
    const rowStart = range.rowStart ?? 0;
    const rowCount = range.rowCount ?? matrix.outputSize - rowStart;
    invariant(
      rowStart >= 0 && rowCount >= 0 && rowStart + rowCount <= matrix.outputSize,
      "INVALID_CACT",
      "Invalid TypeGPU matvec row range",
    );
    const prepared = prepareCqActivation(matrix, input);
    // Needle's 4–512-row projections cannot amortize a WebGPU submission and
    // synchronous readback. The 8,192-row vocabulary projection can.
    if (rowCount < this.#minimumGpuRows)
      return cqMatvecPrepared(matrix, prepared, rowStart, rowCount);
    const gpuMatrix = this.#gpuMatrix(matrix);
    const parameters = matrixParameters(matrix, rowStart, rowCount);
    this.device.queue.writeBuffer(this.#input, 0, bufferSource(prepared));
    this.device.queue.writeBuffer(this.#parameters, 0, bufferSource(parameters));

    const encoder = this.device.createCommandEncoder({ label: "needle.typegpu.cq-matvec.encoder" });
    const pass = encoder.beginComputePass({ label: "needle.typegpu.cq-matvec.pass" });
    pass.setPipeline(this.#pipeline);
    pass.setBindGroup(0, gpuMatrix.bindGroup);
    pass.dispatchWorkgroups(rowCount);
    pass.end();
    encoder.copyBufferToBuffer(this.#output, 0, this.#staging, 0, align4(rowCount * 4));
    this.device.queue.submit([encoder.finish()]);
    await this.#staging.mapAsync(MAP_READ, 0, align4(rowCount * 4));
    const mapped = this.#staging.getMappedRange(0, align4(rowCount * 4));
    const result = new Float32Array(rowCount);
    result.set(new Float32Array(mapped, 0, rowCount));
    this.#staging.unmap();
    return result;
  }

  row(matrix: CqMatrix, row: number, output?: Float32Array): Float32Array {
    return dequantizeCqRow(matrix, row, output);
  }

  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    this.#input.destroy();
    this.#output.destroy();
    this.#parameters.destroy();
    this.#codebook.destroy();
    this.#staging.destroy();
    for (const matrix of this.#allocatedMatrices) {
      matrix.packed.destroy();
      matrix.norms.destroy();
    }
    if (this.#ownsRoot) this.root.destroy();
  }

  #gpuMatrix(matrix: CqMatrix): GpuMatrix {
    const cached = this.#matrixCache.get(matrix);
    if (cached) return cached;
    const data = webGpuMatrixData(matrix);
    const packed = bufferWithData(
      this.device,
      `needle.typegpu.matrix.${matrix.record.index}.packed`,
      data.packedWords,
      STORAGE | COPY_DST,
    );
    const norms = bufferWithData(
      this.device,
      `needle.typegpu.matrix.${matrix.record.index}.norms`,
      data.norms,
      STORAGE | COPY_DST,
    );
    const bindGroup = this.device.createBindGroup({
      label: `needle.typegpu.matrix.${matrix.record.index}.bind-group`,
      layout: this.#pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: packed } },
        { binding: 1, resource: { buffer: norms } },
        { binding: 2, resource: { buffer: this.#input } },
        { binding: 3, resource: { buffer: this.#codebook } },
        { binding: 4, resource: { buffer: this.#parameters } },
        { binding: 5, resource: { buffer: this.#output } },
      ],
    });
    const result = { packed, norms, bindGroup };
    this.#matrixCache.set(matrix, result);
    this.#allocatedMatrices.push(result);
    return result;
  }
}

function align4(value: number): number {
  return Math.max(4, (value + 3) & ~3);
}

function bufferWithData(
  device: GPUDevice,
  label: string,
  data: ArrayBufferView,
  usage: GPUBufferUsageFlags,
): GPUBuffer {
  const buffer = device.createBuffer({ label, size: align4(data.byteLength), usage });
  device.queue.writeBuffer(buffer, 0, bufferSource(data));
  return buffer;
}

function bufferSource(data: ArrayBufferView): GPUAllowSharedBufferSource {
  if (data.buffer instanceof ArrayBuffer) return data as ArrayBufferView<ArrayBuffer>;
  return new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice();
}

export async function createTypeGPUBackend(
  weights: CactWeights,
  options: TypeGPUBackendOptions = {},
): Promise<TypeGPUBackend> {
  return TypeGPUBackend.create(weights, options);
}
