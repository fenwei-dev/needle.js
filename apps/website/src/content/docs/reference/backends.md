---
title: Inference backends
description: Choose pure TypeScript, TypeGPU, or a custom Needle operator backend.
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
    execution: "resident", // or "adaptive" (default)
  },
});
```

Install the optional peer:

```bash
bun add typegpu
```

TypeGPU initializes or adopts the WebGPU device and resolves the CQ matvec WGSL. Packed matrices are uploaded lazily and cached on first use. Its row-parallel kernel uses one 32-lane workgroup per output row.

`execution: "adaptive"` uses the CPU/GPU row cutoff and is the portable default. `execution: "resident"` enables complete resident execution when the published geometry, int8 KV mode, and device limits are compatible; otherwise it automatically falls back. The granular fusion, row-cutoff, and submission options remain diagnostic controls.

## Automatic selection

```ts
const model = await NeedleModel.load({
  weights: "download",
  backend: "auto",
});
```

In a browser with WebGPU, `auto` tries TypeGPU. Otherwise it uses CPU. Explicit GPU selection throws when initialization fails, which is preferable when GPU execution is a deployment requirement.

## Hybrid GPU execution

The TypeGPU backend accelerates packed CQ matrix–vector operators. Independent mHC projections, Q/K/V/gate projections, and engram key/value projections use `matvecBatch()`. Their packed weights and norms are combined into a cached arena, and one dispatch writes a contiguous result, so each group has one queue submission and one host synchronization. Small recurrent operations, attention bookkeeping, grammar state, and sampling remain in TypeScript, with readback between dependent groups.

A matvec with only 4–512 output rows is too small to amortize submission and synchronous readback in this architecture. The adaptive backend therefore defaults `minimumGpuRows` to `1024`; smaller projections use the optimized TypeScript CQ kernel and the official model's 8,192-row vocabulary projection uses WebGPU. Set `minimumGpuRows: 0` to force every CQ matvec onto the GPU for diagnostics.

This adaptive split slightly beats pure TypeScript on the measured Apple WebKit workload. See [backend benchmarks](/needle.js/reference/benchmarks/).

## Experimental resident attention

TypeGPU accepts `fusedAttention: true`. A backend-owned runtime session keeps Q/K/V, the quantized int8 KV cache and scales, attention scores, gating, and the output projection on GPU. Query/key normalization, RoPE, cache updates, score computation, softmax, value mixing, activation preparation, and CQ output projection are encoded before one 512-value readback per layer.

The session supports the published 512-dimensional, 8-query-head/4-KV-head geometry, int8 KV mode, and a combined sink/window span up to 512. Unsupported configurations automatically use the reference runtime.

This cuts the forced-all-GPU path from about 83 host synchronization groups to about 56. `fusedMlp: true` extends the same command buffer through post-attention RMS normalization, the gated residual, pre-Hadamard normalization, two 512-point Walsh–Hadamard transforms, SiLU, and layer-delta construction.

`fusedRouting: true` continues through h-post gating, the 20-iteration 4×4 log-space Sinkhorn router, four-lane mixing, and `nextX` construction. It returns all 2,048 lane values to the reference loop.

The individual fusion options are experimental. Small-context WebGPU attention is slower than CPU attention despite removing readbacks; the MLP extension adds work without removing another boundary; and standalone routing uploads and reads lane state every layer.

### Resident layers

`execution: "resident"` enables all compatible fusion stages and eliminates that lane-state round trip. At the start of a token it uploads the four replicated embedding lanes. For each layer it then queues, without a host readback:

1. global lane RMS and sixteen groupwise Hadamard preparations;
2. selected mHC phi projections and h-pre lane reduction;
3. resident engram row gather/dequantization, key/value projection, convolution-ring update, injection, and input RMSNorm;
4. resident attention/KV, output projection, sandwich residual, and MLP;
5. h-post, Sinkhorn routing, and `nextX`, copied into the next layer's lane buffer.

By default each encoded layer is submitted immediately. This allows WebKit to execute GPU work while JavaScript encodes the following layer. `singleTokenSubmission: true` instead gives every layer distinct parameter slots and submits all 27 command sequences together. It is correct but diagnostic: the standard benchmark fell from 52.5 to 38.0 tok/s because delaying submission removed CPU/GPU overlap.

After layer 27, final lane averaging, RMSNorm, activation preparation, and vocabulary projection remain on GPU. Non-logit prefill steps do not read layer/final state at all. Raw callers may read all 8,192 logits once; high-level tool calling builds an ascending list of grammar-allowed token IDs in JavaScript, uploads that list, and receives only the selected ID and full-vocabulary log probability.

JavaScript still computes the four n-gram hashes, but uploads only row IDs and validity flags. Both CQ engram tables, projected keys/values, and the dilated convolution ring remain GPU-resident.

Confidence collection is also resident for the official eight-probe, one-output head. A GPU online pool consumes the replicated embedding and every layer's four-lane mean, then returns one sigmoid score. The flashlight integration produced the same constrained call on CPU and resident TypeGPU in 1,153 ms and 1,016 ms respectively, with final confidence 0.7526 vs 0.7481. Raw generation measured 52.0 tok/s on the standard workload and 5.40 tok/s at a 98-token prompt, ahead of CPU in both paired runs.

## Custom backend

Implement `InferenceBackend`:

```ts
interface MatvecRequest {
  matrix: CqMatrix;
  input: Float32Array;
  range?: { rowStart?: number; rowCount?: number };
}

interface InferenceBackend {
  kind: "cpu" | "typegpu";
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

## Resident engine architecture

The resident implementation no longer lives inside the TypeGPU adapter. It is organized as a raw-WebGPU core:

```text
src/resident/
  session.ts        # model state and execution lifecycle
  pipelines.ts      # shader-module and pipeline compilation
  kernels.ts        # WGSL kernels
  confidence.ts     # online confidence pooling/head
  selection.ts      # allowed-token argmax and log-softmax
  compatibility.ts  # geometry/KV feature gates
  webgpu.ts         # raw buffer and upload helpers
  parameters.ts     # neutral typed parameter contracts/raw factory
  resources.ts      # neutral f32/u32/i32 allocation contracts
  bindings.ts       # neutral typed binding contracts

src/backends/typegpu.ts
  # TypeGPU root/device acquisition, lifecycle, adaptive operators
src/backends/typegpu-parameters.ts
  # d.struct schemas and root-owned typed parameter buffers
src/backends/typegpu-resources.ts
  # d.arrayOf schemas and root-owned resident state
src/backends/typegpu-bindings.ts
  # typed query, KV-store, score, and softmax bind-group layouts
src/backends/typegpu-kernels.ts
  # TypeGPU compute functions and shader dependency composition
```

`WebGpuResidentSession` depends only on `GPUDevice`, parsed model weights, and neutral resident options. TypeGPU is the supported public integration.

Dynamic query, KV, and attention parameter blocks use TypeGPU `d.struct` schemas and root-owned typed storage buffers. Primary resident state—including lanes, QKV, attention/MLP intermediates, engram rings, logits, and int8-as-i32 KV storage—uses TypeGPU `d.arrayOf` schemas through a neutral resource factory. The adapter supplies them through `ResidentParameterFactory`; the resident core sees named values and raw buffers, not TypeGPU implementation types. A raw factory remains available as the portability/reference implementation. The resource contract also owns disposal: TypeGPU-created raw handles resolve back to their typed wrappers for destruction, while the raw factory destroys its own `GPUBuffer` objects. The session no longer double-manages TypeGPU resources. The query normalization/RoPE, KV-store, attention-score, and softmax/value-mixing passes use TypeGPU bind-group layouts and explicit pipeline layouts. Resident query normalization/RoPE and packed CQ matvec are compiled as TypeGPU `computeFn` pipelines; TypeGPU owns its workgroup variables, binding dependencies, generated declarations, and pipeline. Their function bodies remain inline WGSL so the published library does not require consumers to configure `unplugin-typegpu`. Raw command encoding preserves the measured scheduling behavior. The adaptive operator path retains its faster measured raw CQ pipeline; resident execution uses the TypeGPU CQ pipeline. This recovers TypeGPU's compile-time layouts, typed writes, root ownership, and binding validation without coupling the execution graph or regressing throughput.

## Browser correctness suite

Run `bun run --cwd packages/needle.js test:browser` to validate the resident TypeGPU path with real weights in Bun.WebView. It covers greedy and constrained decoding, confidence, long sliding-window attention with prefix sinks, resets, fallback, and cancellation. Set `RESIDENT_WEBVIEW_BACKEND=chrome` to exercise Bun's Chrome backend on a machine where headless WebGPU is enabled.
