---
title: Inference backends
description: Choose pure TypeScript, TypeGPU, vgpu, or a custom Needle operator backend.
---

The model architecture and decoding loop are backend-independent. A backend supplies two packed-weight operations: selected-row matrix–vector multiplication and row gather.

## Pure TypeScript

```ts
import { createNeedleCPU } from "needle.js/cpu";

const agent = await createNeedleCPU({
  weights: "download",
  tools,
});
```

- Works in Bun, Node, workers, and browsers
- No native addon, WASM, BLAS, or GPU
- Reads CQ weights directly instead of expanding the model
- Useful as the reference implementation and universal fallback

The CQ kernel applies a normalized fast Walsh–Hadamard transform to each activation group. It then decodes packed codebook indices while taking the dot product. The inverse transform never has to materialize a dense matrix.

## TypeGPU

```ts
import { createNeedleTypeGPU } from "needle.js/typegpu";

const agent = await createNeedleTypeGPU({
  weights: "download",
  tools,
  backendOptions: {
    // root: existingTgpuRoot,
    // device: existingGPUDevice,
    // minimumGpuRows: 0, // force all matvecs onto WebGPU
  },
});
```

Install the optional peer:

```bash
bun add typegpu
```

TypeGPU initializes or adopts the WebGPU device and resolves the CQ matvec WGSL. Packed matrices are uploaded lazily and cached on first use. Its row-parallel kernel uses one 32-lane workgroup per output row.

## vgpu

```ts
import { createNeedleVGPU } from "needle.js/vgpu";

const agent = await createNeedleVGPU({
  weights: "download",
  tools,
  backendOptions: {
    // gpu: existingVgpuContext,
    // device: existingGPUDevice,
    // node: true,
    // minimumGpuRows: 0, // force all matvecs onto WebGPU
  },
});
```

Install the optional peer:

```bash
bun add vgpu
```

The browser route uses `vgpu`; Node uses `vgpu/node`. A supplied context remains caller-owned.

## Automatic selection

```ts
const model = await NeedleModel.load({
  weights: "download",
  backend: "auto",
});
```

In a browser with WebGPU, `auto` tries TypeGPU and then vgpu. Otherwise it uses CPU. Explicit GPU selection throws when initialization fails, which is preferable when GPU execution is a deployment requirement.

## Hybrid GPU execution

The GPU backends accelerate packed CQ matrix–vector operators. Independent mHC projections, Q/K/V/gate projections, and engram key/value projections use `matvecBatch()` so each group has one host synchronization. Small recurrent operations, attention bookkeeping, grammar state, and sampling remain in TypeScript, with readback between dependent groups.

A matvec with only 4–512 output rows is too small to amortize submission and synchronous readback in this architecture. Both backends therefore default `minimumGpuRows` to `1024`; smaller projections use the optimized TypeScript CQ kernel and the official model's 8,192-row vocabulary projection uses WebGPU. Set `minimumGpuRows: 0` to force every CQ matvec onto the GPU for diagnostics.

This adaptive split slightly beats pure TypeScript on the measured Apple WebKit workload. A future fully resident engine could keep hidden state and KV cache on GPU and make the smaller projections worthwhile without intermediate readback. See [backend benchmarks](/needle.js/reference/benchmarks/).

## Custom backend

Implement `InferenceBackend`:

```ts
interface MatvecRequest {
  matrix: CqMatrix;
  input: Float32Array;
  range?: { rowStart?: number; rowCount?: number };
}

interface InferenceBackend {
  kind: "cpu" | "typegpu" | "vgpu";
  weights: CactWeights;
  matvec(
    matrix: CqMatrix,
    input: Float32Array,
    range?: { rowStart?: number; rowCount?: number },
  ): Float32Array | Promise<Float32Array>;
  matvecBatch?(
    requests: readonly MatvecRequest[],
  ): readonly Float32Array[] | Promise<readonly Float32Array[]>;
  row(matrix: CqMatrix, row: number, output?: Float32Array): Float32Array;
  dispose(): void | Promise<void>;
}
```

The interface is the seam used by all bundled implementations and a starting point for experimenting with WASM, WebNN, SIMD, or a fused WebGPU runtime.
