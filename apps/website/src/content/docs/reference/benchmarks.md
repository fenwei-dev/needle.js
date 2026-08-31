---
title: Backend benchmarks
description: Measured Needle 2 generation throughput for the CPU, TypeGPU, and vgpu backends.
---

Needle 2 generation was timed in **Bun.WebView** (system WKWebView on macOS) so TypeGPU and vgpu use a real `navigator.gpu` device. All three backends loaded the same official 13.7 MB `.cact` archive and ran greedy generation from a shared prompt.

## Latest numbers

The default GPU configuration is adaptive: CQ projections with fewer than 1,024 output rows stay on CPU, while the official model's 8,192-row vocabulary projection runs on WebGPU.

| Backend | Status | Load (ms) | Median tok/s | Median ms / 32 tokens |
| --- | --- | ---: | ---: | ---: |
| Pure TypeScript (`cpu`) | ok | 18 | 44.2 | 725 |
| TypeGPU + WebGPU | ok | 17 | **50.8** | 630 |
| vgpu + WebGPU | ok | 10 | 50.5 | 634 |

- **Host:** macOS `darwin arm64`, Bun 1.4.0 WebView
- **User agent:** `Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko)`
- **WebGPU:** available
- **Workload:** warmup 1 + 2 timed runs, 32 new tokens, temperature 0
- **Prompt:** `The most surprising thing about local inference is` (13 prompt tokens)
- **GPU cutoff:** `minimumGpuRows: 1024` (the default)

Both GPU backends produced the same greedy text as CPU on this prompt.

## Why the adaptive split wins

Needle 2's hidden dimension is only 512. Most selected-row CQ operations produce 4, 16, 256, or 512 values, and each GPU result is needed by TypeScript before the next dependent operator. For these matrices, WebGPU submission and synchronous readback cost more than the arithmetic.

The 8,192 × 512 vocabulary projection is large enough to benefit. Its CQ shader assigns one output row to a 32-lane workgroup, computes partial dot products in parallel, and reduces them in workgroup memory. Keeping the smaller projections in the optimized TypeScript CQ kernel removes almost all GPU synchronization while retaining acceleration for the large output head.

## Diagnostic all-GPU matvec result

Setting `minimumGpuRows: 0` forces all packed matvecs through WebGPU. This is useful for backend validation, but it creates roughly 220 submission/readback boundaries per model step.

| Backend | Median tok/s | Median ms / 32 tokens |
| --- | ---: | ---: |
| TypeGPU, every matvec on WebGPU | 6.68 | 4,794 |
| vgpu, every matvec on WebGPU | 11.33 | 2,825 |

A future fully resident engine could keep hidden state, intermediate activations, and KV cache on GPU; encode dependent kernels into a small number of command buffers; and read back only selected tokens or final logits. That is the path to making the whole layer stack faster rather than merely moving each individual matvec to a GPU.

## Reproduce

From the monorepo, with Needle 2 weights on disk (`NEEDLE_MODEL_PATH` or `~/.cache/needle.js/models/<sha256>.cact`):

```bash
# Default adaptive backend
bun run --cwd packages/needle.js bench
bun run --cwd packages/needle.js bench -- --tokens 32 --warmup 1 --runs 2 --json

# Diagnostic: force every matvec onto WebGPU
bun run --cwd packages/needle.js bench -- --minimum-gpu-rows 0
```

The harness bundles `scripts/bench-browser.ts`, serves it locally, and drives `Bun.WebView` with the `webkit` backend so Chrome is not launched with `--disable-gpu`.
