# needle.js GPU architecture

## Goals

The GPU implementation should combine TypeGPU's compile-time layout and
resource safety with the explicit scheduling required by Needle's recurrent,
autoregressive graph. CPU remains the numerical reference and universal
fallback.

## Boundaries

### TypeGPU owns

- browser device acquisition and adopted-device integration;
- root and typed-resource lifecycle;
- `d.struct` dynamic parameter layouts and object writes;
- `d.arrayOf` resident f32/u32/i32 state allocations;
- CQ, query, KV, attention, confidence, and selection bind-group layouts;
- explicit pipeline layouts;
- resident CQ and query compute-function dependency graphs.

### Raw WebGPU owns

- command encoder creation and pass ordering;
- immediate per-layer submission, which preserves measured CPU/GPU overlap;
- copies into staging buffers and mapped readback;
- the adaptive operator CQ path, where the raw pipeline benchmarks faster;
- fallback resources when the resident core is used without a TypeGPU factory.

Raw use is an explicit scheduling/performance boundary, not the model's data
or binding architecture.

## Resident modules

```text
src/resident/
  session.ts        token/layer orchestration
  engram.ts         table gather, projection, convolution state, injection
  confidence.ts     online probe pooling and confidence head
  selection.ts      constrained argmax and log-softmax
  pipelines.ts      pipeline compilation and raw fallbacks
  kernels.ts        reference WGSL kernels
  parameters.ts     neutral named parameter contracts
  resources.ts      neutral typed-array allocation contracts
  bindings.ts       neutral binding contracts
  compatibility.ts  model and runtime feature gates
  webgpu.ts         raw WebGPU utilities

src/backends/
  typegpu.ts             public adapter and adaptive operators
  typegpu-parameters.ts  d.struct schemas and typed writers
  typegpu-resources.ts   d.arrayOf state allocation
  typegpu-layouts.ts     typed bind-group layouts
  typegpu-bindings.ts    root-created groups and pipelines
  typegpu-kernels.ts     TypeGPU compute functions
```

## Shader body decision

TypeGPU TypeScript function bodies require `unplugin-typegpu` metadata
transformation. needle.js currently publishes direct `tsc` output, and package
consumers must not need a special application bundler plugin to import the
prebuilt library.

For now, TypeGPU compute functions use inline WGSL bodies with TypeGPU-owned
schemas, layouts, workgroup variables, dependencies, generated declarations,
bind groups, and pipelines. This captures the safety and composition benefits
without changing the publication toolchain.

A future `unplugin-typegpu` experiment must:

1. transform library sources during package build, not in consumer projects;
2. preserve declaration output, ESM exports, source maps, and tree shaking;
3. produce inspectable WGSL equivalent to the committed reference;
4. pass the real-browser suite and confidence corpus;
5. remain within 3% of the canonical resident benchmark.

Until all five conditions pass, inline WGSL compute-function bodies are the
accepted production architecture.

## Performance rules

- Do not reduce queue submissions without measuring overlap. A single delayed
  token submission measured slower than immediate layer submissions on WebKit.
- Keep intermediate state resident; read only logits, selected tokens, or final
  scalar confidence values.
- Preserve a raw implementation for parity and performance comparison.
- Reject migrations with call divergence, threshold-decision divergence, or a
  repeatable throughput regression above 3%.

## Verification

```bash
bun run --cwd packages/needle.js test
bun run --cwd packages/needle.js test:browser
bun run --cwd packages/needle.js test:corpus
bun run --cwd packages/needle.js test:package
```

`test:browser` requires a real WebGPU adapter. GitHub-hosted WKWebView currently
reports a capability warning and skips; local or self-hosted compatible runners
execute the full suite.
