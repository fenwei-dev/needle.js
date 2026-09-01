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
    // fusedAttention: true, // experimental resident int8 KV + attention
    // fusedMlp: true, // also fuse sandwich residuals and Hadamard MLP
    // fusedRouting: true, // also fuse post-mHC Sinkhorn and nextX lanes
    // residentLayers: true, // retain nextX and run complete layers on GPU
    // singleTokenSubmission: true, // diagnostic: one submit for all layers
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

The GPU backends accelerate packed CQ matrix–vector operators. Independent mHC projections, Q/K/V/gate projections, and engram key/value projections use `matvecBatch()`. Their packed weights and norms are combined into a cached arena, and one dispatch writes a contiguous result, so each group has one queue submission and one host synchronization. Small recurrent operations, attention bookkeeping, grammar state, and sampling remain in TypeScript, with readback between dependent groups.

A matvec with only 4–512 output rows is too small to amortize submission and synchronous readback in this architecture. Both backends therefore default `minimumGpuRows` to `1024`; smaller projections use the optimized TypeScript CQ kernel and the official model's 8,192-row vocabulary projection uses WebGPU. Set `minimumGpuRows: 0` to force every CQ matvec onto the GPU for diagnostics.

This adaptive split slightly beats pure TypeScript on the measured Apple WebKit workload. See [backend benchmarks](/needle.js/reference/benchmarks/).

## Experimental resident attention

TypeGPU accepts `fusedAttention: true`. A backend-owned runtime session keeps Q/K/V, the quantized int8 KV cache and scales, attention scores, gating, and the output projection on GPU. Query/key normalization, RoPE, cache updates, score computation, softmax, value mixing, activation preparation, and CQ output projection are encoded before one 512-value readback per layer.

The session supports the published 512-dimensional, 8-query-head/4-KV-head geometry, int8 KV mode, and a combined sink/window span up to 512. Unsupported configurations automatically use the reference runtime.

This cuts the forced-all-GPU path from about 83 host synchronization groups to about 56. `fusedMlp: true` extends the same command buffer through post-attention RMS normalization, the gated residual, pre-Hadamard normalization, two 512-point Walsh–Hadamard transforms, SiLU, and layer-delta construction.

`fusedRouting: true` continues through h-post gating, the 20-iteration 4×4 log-space Sinkhorn router, four-lane mixing, and `nextX` construction. It returns all 2,048 lane values to the reference loop.

The individual fusion options are experimental. Small-context WebGPU attention is slower than CPU attention despite removing readbacks; the MLP extension adds work without removing another boundary; and standalone routing uploads and reads lane state every layer.

### Resident layers

`residentLayers: true` implies all three fusion stages and eliminates that lane-state round trip. At the start of a token it uploads the four replicated embedding lanes. For each layer it then queues, without a host readback:

1. global lane RMS and sixteen groupwise Hadamard preparations;
2. selected mHC phi projections and h-pre lane reduction;
3. optional engram injection and input RMSNorm;
4. resident attention/KV, output projection, sandwich residual, and MLP;
5. h-post, Sinkhorn routing, and `nextX`, copied into the next layer's lane buffer.

By default each encoded layer is submitted immediately. This allows WebKit to execute GPU work while JavaScript encodes the following layer. `singleTokenSubmission: true` instead gives every layer distinct parameter slots and submits all 27 command sequences together. It is correct but diagnostic: the standard benchmark fell from 52.5 to 38.0 tok/s because delaying submission removed CPU/GPU overlap.

After layer 27, final lane averaging, RMSNorm, activation preparation, and vocabulary projection remain on GPU. Non-logit prefill steps do not read layer/final state at all. Raw callers may read all 8,192 logits once; high-level tool calling builds an ascending list of grammar-allowed token IDs in JavaScript, uploads that list, and receives only the selected ID and full-vocabulary log probability.

Confidence collection is also resident for the official eight-probe, one-output head. A GPU online pool consumes the replicated embedding and every layer's four-lane mean, then returns one sigmoid score. The flashlight integration produced the same constrained call on CPU and resident TypeGPU in 1,150 ms and 1,012 ms respectively, with final confidence 0.7526 vs 0.7563. Raw generation measured 51.4 tok/s on the standard workload and 5.40 tok/s at a 98-token prompt, ahead of CPU in both paired runs.

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
  createFusedAttentionSession?(): FusedAttentionSession | undefined;
  row(matrix: CqMatrix, row: number, output?: Float32Array): Float32Array;
  dispose(): void | Promise<void>;
}
```

The interface is the seam used by all bundled implementations and a starting point for experimenting with WASM, WebNN, SIMD, or a fused WebGPU runtime.
