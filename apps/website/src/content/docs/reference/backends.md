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
  },
});
```

Install the optional peer:

```bash
bun add typegpu
```

TypeGPU initializes or adopts the WebGPU device and resolves the CQ matvec WGSL. Packed matrices are uploaded lazily and cached on first use.

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

The GPU backends accelerate packed CQ matrix–vector operators. Small recurrent operations, attention bookkeeping, grammar state, and sampling remain in TypeScript, with readback between dependent operators. This favors a compact, auditable implementation over maximum throughput.

A future fully fused engine could keep hidden state and KV cache resident on GPU and dispatch layer pipelines without intermediate CPU readback. Measured greedy generation on Apple WebKit WebGPU is therefore slower than the pure TypeScript kernel today; see [backend benchmarks](/needle.js/reference/benchmarks/).

## Custom backend

Implement `InferenceBackend`:

```ts
interface InferenceBackend {
  kind: "cpu" | "typegpu" | "vgpu";
  weights: CactWeights;
  matvec(
    matrix: CqMatrix,
    input: Float32Array,
    range?: { rowStart?: number; rowCount?: number },
  ): Float32Array | Promise<Float32Array>;
  row(matrix: CqMatrix, row: number, output?: Float32Array): Float32Array;
  dispose(): void | Promise<void>;
}
```

The interface is the seam used by all bundled implementations and a starting point for experimenting with WASM, WebNN, SIMD, or a fused WebGPU runtime.
