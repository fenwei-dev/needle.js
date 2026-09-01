/// <reference types="@webgpu/types" preserve="true" />
import { type TgpuRoot, tgpu } from "typegpu";
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

const MAP_READ = 0x0001;
const COPY_SRC = 0x0004;
const COPY_DST = 0x0008;
const STORAGE = 0x0080;

interface GpuMatrix {
  readonly packed: GPUBuffer;
  readonly norms: GPUBuffer;
}

interface GpuSlot {
  readonly input: GPUBuffer;
  readonly output: GPUBuffer;
  readonly parameters: GPUBuffer;
  readonly bindGroups: WeakMap<CqMatrix, GPUBindGroup>;
}

interface BatchJob {
  readonly index: number;
  readonly rowCount: number;
  readonly stagingOffset: number;
  readonly slot: GpuSlot;
  readonly matrix: GpuMatrix;
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
  readonly #codebook: GPUBuffer;
  readonly #staging: GPUBuffer;
  readonly #maximumInputBytes: number;
  readonly #maximumOutputBytes: number;
  readonly #slots: GpuSlot[] = [];
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
    this.#maximumInputBytes = align4(maximumInput * 4);
    this.#maximumOutputBytes = align4(maximumOutput * 4);
    this.#codebook = bufferWithData(
      this.device,
      "needle.typegpu.codebook",
      weights.codebook,
      STORAGE | COPY_DST,
    );
    this.#staging = this.device.createBuffer({
      label: "needle.typegpu.readback",
      size: this.#maximumOutputBytes,
      usage: MAP_READ | COPY_DST,
    });
    this.#slots.push(this.#createSlot(0));
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
    if (rowCount === 0) return new Float32Array();
    const prepared = prepareCqActivation(matrix, input);
    // Needle's 4–512-row projections cannot amortize a WebGPU submission and
    // synchronous readback. The 8,192-row vocabulary projection can.
    if (rowCount < this.#minimumGpuRows)
      return cqMatvecPrepared(matrix, prepared, rowStart, rowCount);
    const gpuMatrix = this.#gpuMatrix(matrix);
    const parameters = matrixParameters(matrix, rowStart, rowCount);
    const slot = this.#slot(0);
    this.device.queue.writeBuffer(slot.input, 0, bufferSource(prepared));
    this.device.queue.writeBuffer(slot.parameters, 0, bufferSource(parameters));

    const encoder = this.device.createCommandEncoder({ label: "needle.typegpu.cq-matvec.encoder" });
    const pass = encoder.beginComputePass({ label: "needle.typegpu.cq-matvec.pass" });
    pass.setPipeline(this.#pipeline);
    pass.setBindGroup(0, this.#bindGroup(matrix, gpuMatrix, slot));
    pass.dispatchWorkgroups(rowCount);
    pass.end();
    encoder.copyBufferToBuffer(slot.output, 0, this.#staging, 0, align4(rowCount * 4));
    this.device.queue.submit([encoder.finish()]);
    await this.#staging.mapAsync(MAP_READ, 0, align4(rowCount * 4));
    const mapped = this.#staging.getMappedRange(0, align4(rowCount * 4));
    const result = new Float32Array(rowCount);
    result.set(new Float32Array(mapped, 0, rowCount));
    this.#staging.unmap();
    return result;
  }

  async matvecBatch(requests: readonly MatvecRequest[]): Promise<readonly Float32Array[]> {
    invariant(!this.#disposed, "BACKEND_UNAVAILABLE", "TypeGPU backend has been disposed");
    if (requests.length === 0) return [];

    const results = new Array<Float32Array>(requests.length);
    const preparedCache = new Map<Float32Array, Map<string, Float32Array>>();
    const jobs: BatchJob[] = [];
    let stagingBytes = 0;

    for (let index = 0; index < requests.length; index++) {
      const request = requests[index];
      invariant(request, "INVALID_CACT", `Missing TypeGPU batch request ${index}`);
      const { matrix, input, range = {} } = request;
      const rowStart = range.rowStart ?? 0;
      const rowCount = range.rowCount ?? matrix.outputSize - rowStart;
      invariant(
        rowStart >= 0 && rowCount >= 0 && rowStart + rowCount <= matrix.outputSize,
        "INVALID_CACT",
        "Invalid TypeGPU batched matvec row range",
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

      const slot = this.#slot(jobs.length);
      this.device.queue.writeBuffer(slot.input, 0, bufferSource(prepared));
      this.device.queue.writeBuffer(
        slot.parameters,
        0,
        bufferSource(matrixParameters(matrix, rowStart, rowCount)),
      );
      jobs.push({
        index,
        rowCount,
        stagingOffset: stagingBytes,
        slot,
        matrix: this.#gpuMatrix(matrix),
      });
      stagingBytes += align4(rowCount * 4);
    }

    if (jobs.length === 0) return results;
    const temporaryStaging = stagingBytes > this.#maximumOutputBytes;
    const staging = temporaryStaging
      ? this.device.createBuffer({
          label: "needle.typegpu.batch-readback",
          size: stagingBytes,
          usage: MAP_READ | COPY_DST,
        })
      : this.#staging;

    try {
      const encoder = this.device.createCommandEncoder({
        label: "needle.typegpu.cq-matvec-batch.encoder",
      });
      const pass = encoder.beginComputePass({
        label: "needle.typegpu.cq-matvec-batch.pass",
      });
      pass.setPipeline(this.#pipeline);
      for (const job of jobs) {
        const request = requests[job.index];
        invariant(request, "INVALID_CACT", `Missing TypeGPU batch request ${job.index}`);
        pass.setBindGroup(0, this.#bindGroup(request.matrix, job.matrix, job.slot));
        pass.dispatchWorkgroups(job.rowCount);
      }
      pass.end();
      for (const job of jobs) {
        encoder.copyBufferToBuffer(
          job.slot.output,
          0,
          staging,
          job.stagingOffset,
          align4(job.rowCount * 4),
        );
      }
      this.device.queue.submit([encoder.finish()]);
      await staging.mapAsync(MAP_READ, 0, stagingBytes);
      try {
        const mapped = staging.getMappedRange(0, stagingBytes);
        for (const job of jobs) {
          const result = new Float32Array(job.rowCount);
          result.set(new Float32Array(mapped, job.stagingOffset, job.rowCount));
          results[job.index] = result;
        }
      } finally {
        staging.unmap();
      }
      return results;
    } finally {
      if (temporaryStaging) staging.destroy();
    }
  }

  row(matrix: CqMatrix, row: number, output?: Float32Array): Float32Array {
    return dequantizeCqRow(matrix, row, output);
  }

  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    for (const slot of this.#slots) {
      slot.input.destroy();
      slot.output.destroy();
      slot.parameters.destroy();
    }
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
    const result = { packed, norms };
    this.#matrixCache.set(matrix, result);
    this.#allocatedMatrices.push(result);
    return result;
  }

  #slot(index: number): GpuSlot {
    let slot = this.#slots[index];
    if (!slot) {
      slot = this.#createSlot(index);
      this.#slots[index] = slot;
    }
    return slot;
  }

  #createSlot(index: number): GpuSlot {
    return {
      input: this.device.createBuffer({
        label: `needle.typegpu.input.${index}`,
        size: this.#maximumInputBytes,
        usage: STORAGE | COPY_DST,
      }),
      output: this.device.createBuffer({
        label: `needle.typegpu.output.${index}`,
        size: this.#maximumOutputBytes,
        usage: STORAGE | COPY_SRC,
      }),
      parameters: this.device.createBuffer({
        label: `needle.typegpu.parameters.${index}`,
        size: 32,
        usage: STORAGE | COPY_DST,
      }),
      bindGroups: new WeakMap(),
    };
  }

  #bindGroup(matrix: CqMatrix, gpuMatrix: GpuMatrix, slot: GpuSlot): GPUBindGroup {
    const cached = slot.bindGroups.get(matrix);
    if (cached) return cached;
    const bindGroup = this.device.createBindGroup({
      label: `needle.typegpu.matrix.${matrix.record.index}.bind-group`,
      layout: this.#pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: gpuMatrix.packed } },
        { binding: 1, resource: { buffer: gpuMatrix.norms } },
        { binding: 2, resource: { buffer: slot.input } },
        { binding: 3, resource: { buffer: this.#codebook } },
        { binding: 4, resource: { buffer: slot.parameters } },
        { binding: 5, resource: { buffer: slot.output } },
      ],
    });
    slot.bindGroups.set(matrix, bindGroup);
    return bindGroup;
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
