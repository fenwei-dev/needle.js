---
title: Backend benchmarks
description: Measured Needle 2 generation throughput for the CPU and TypeGPU backends.
---

Needle 2 generation was timed in **Bun.WebView** (system WKWebView on macOS) so TypeGPU uses a real `navigator.gpu` device. Both backends loaded the same official 13.7 MB `.cact` archive and ran greedy generation from a shared prompt.

## Latest numbers

The canonical GPU result uses `execution: "resident"`; the complete layer graph, engrams, confidence state, and vocabulary projection remain on WebGPU.

| Backend | Status | Load (ms) | Median tok/s | Median ms / 32 tokens |
| --- | --- | ---: | ---: | ---: |
| Pure TypeScript (`cpu`) | ok | 24 | 45.3 | 707 |
| TypeGPU resident WebGPU | ok | 19 | **54.7** | 585 |

- **Host:** Apple M4 (`Mac16,12`), macOS `darwin arm64`, Bun 1.4.0 WebView
- **User agent:** `Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko)`
- **WebGPU:** available
- **Workload:** 3 warmups + 10 timed runs, 32 new tokens, temperature 0
- **Prompt:** `The most surprising thing about local inference is` (13 prompt tokens)
- **Execution:** `resident`

TypeGPU produced the same greedy text as CPU on this prompt.

## Why the adaptive split wins

Needle 2's hidden dimension is only 512. Most selected-row CQ operations produce 4, 16, 256, or 512 values, and each GPU result is needed by TypeScript before the next dependent operator. For these matrices, WebGPU submission and synchronous readback cost more than the arithmetic.

The 8,192 × 512 vocabulary projection is large enough to benefit. Its CQ shader assigns one output row to a 32-lane workgroup, computes partial dot products in parallel, and reduces them in workgroup memory. Keeping the smaller projections in the optimized TypeScript CQ kernel removes almost all GPU synchronization while retaining acceleration for the large output head.

## Diagnostic all-GPU matvec result

Setting `diagnostics: { minimumGpuRows: 0 }` forces all packed matvecs through WebGPU. This is useful for backend validation. The runtime batches projections that are independent at that point in the graph:

- the three mHC routing projections;
- query, key, value, and gate;
- both engram sites' key and value projections.

This reduces host synchronization from roughly 220 individual matvec boundaries to about 83 groups per model step.

| Backend | Median tok/s | Median ms / 32 tokens |
| --- | ---: | ---: |
| TypeGPU, every matvec on WebGPU | 21.0 | 1,524 |

These diagnostic medians use two warmups and three timed runs. Before batching, the all-WebGPU path measured 6.68 tok/s. A batch now uses one dispatch over cached, combined matrix arenas rather than one dispatch per projection.

### Resident-attention experiment

TypeGPU's `diagnostics: { residentStage: "attention" }` keeps Q/K/V, int8 KV state, attention, gating, and output projection resident through one layer command buffer. It reduces the forced-all-GPU path from about 83 to about 56 host groups per model step and matched CPU greedy output on the benchmark prompts.

For this short 13-token prompt, it measured about **17 tok/s**, below the 21 tok/s batched-matvec path: GPU softmax and cache kernels cost more than the readback they replace at this sequence length. It is therefore experimental rather than enabled by default. At a 98-token prompt it was approximately even with, and slightly ahead of, the non-resident all-GPU path.

`residentStage: "mlp"` continues through sandwich normalization and the fixed Hadamard MLP, returning the layer delta. It measured about **15 tok/s** and matched the same greedy output.

`residentStage: "routing"` additionally performs h-post gating, 20 log-space Sinkhorn iterations, four-lane routing, and `nextX` construction on GPU. In isolation it measured about **10.6 tok/s**, because it transfers 2,048 lane values in both directions each layer.

`execution: "resident"` retains that `nextX` buffer for the next layer and adds GPU mHC pre-routing, h-pre reduction, engram injection, and input normalization. All 27 layers run without host readback. Final lane averaging, RMSNorm, activation preparation, and vocabulary projection also execute on GPU; non-logit prefill steps now read nothing after their layers.

| Resident-layer workload | CPU | TypeGPU resident layers |
| --- | ---: | ---: |
| Standard prompt, 32 generated tokens | 45.3 tok/s | **54.7 tok/s** |
| 98-token prompt, 8 generated tokens | 4.68 tok/s | **5.40 tok/s** |

The resident path matched CPU greedy text in both workloads. The canonical ten-run JSON report is [stored in the repository](https://github.com/fenwei-dev/needle.js/blob/main/packages/needle.js/benchmarks/2026-09-02-macos-m4-webkit.json).

### One submission per token experiment

A token-level command builder assigns independent parameter buffers and bind groups to all 27 layers, allowing one `queue.submit()` after every layer has been encoded. Output remained identical, but WKWebView throughput fell from **52.5 tok/s** with immediate per-layer submissions to about **38.0 tok/s**. Immediate submission overlaps JavaScript command encoding with GPU execution; one delayed submission loses that overlap. The implementation remains available as `diagnostics: { submission: "single" }` for cross-device diagnostics, but is not the default.

### Resident tool selection and confidence

For `Needle.complete()`, JavaScript still advances the byte grammar, but it sends only the currently allowed token IDs to WebGPU. A 256-lane reduction selects the best allowed token while computing log-sum-exp over the full vocabulary, and reads back an 8-byte token/log-probability pair instead of 32 KB of logits.

The resident confidence implementation maintains eight online probe pools from the token embedding and all layer means, then evaluates the one-output confidence projection on GPU. On the constrained flashlight integration:

| Backend | Raw call | Final confidence | Elapsed |
| --- | --- | ---: | ---: |
| CPU | `[{"name":"turn_on_flashlight","arguments":{}}]` | 0.7526 | 1,153 ms |
| TypeGPU resident | same | 0.7481 | **1,016 ms** |

A prompt-only probe check differed by about `1.1e-9`; the final turn's small confidence difference includes accumulated f32 model, resident engram, and token-log-probability differences. Both paths selected the exact same schema-constrained token sequence.

A broader committed corpus covers 20 commands across flashlight, weather, timers, volume, brightness, messaging, calendar, conversion, lights, and reminders. CPU and resident TypeGPU produced **zero call/argument mismatches** and **zero decision mismatches** at confidence thresholds 0.7, 0.8, and 0.9. Median confidence delta was `0.00023`; mean was `0.00170`; maximum was `0.01122`. See the [raw corpus report](https://github.com/fenwei-dev/needle.js/blob/main/packages/needle.js/benchmarks/2026-09-02-tool-confidence-corpus.json).

### Resident engrams

The resident path now uploads four hashed row IDs rather than CPU-materialized engram keys and values. WebGPU gathers and dequantizes four 128-value W2 rows per site, performs key/value CQ projections, updates the dilated convolution ring, and injects the result at layers 2 and 15. Standard throughput increased from 51.4 to **52.0 tok/s** while preserving greedy text and the constrained flashlight call.

## Reproduce

From the monorepo, with Needle 2 weights on disk (`NEEDLE_MODEL_PATH` or `~/.cache/needle.js/models/<sha256>.cact`):

```bash
# Default adaptive backend
bun run --cwd packages/needle.js bench
bun run --cwd packages/needle.js bench -- --tokens 32 --warmup 1 --runs 2 --json

# Diagnostic: force every matvec onto WebGPU
bun run --cwd packages/needle.js bench -- --minimum-gpu-rows 0

# TypeGPU resident-attention experiment
bun run --cwd packages/needle.js bench -- --minimum-gpu-rows 0 --fused-attention

# Continue through sandwich residuals and Hadamard MLP
bun run --cwd packages/needle.js bench -- --minimum-gpu-rows 0 --fused-mlp

# Continue through post-mHC routing and four-lane nextX
bun run --cwd packages/needle.js bench -- --minimum-gpu-rows 0 --fused-routing

# Retain lanes and run all 27 layers without host readback
bun run --cwd packages/needle.js bench -- --execution resident

# Compare resident confidence and constrained tool selection
bun run --cwd packages/needle.js bench -- --execution resident --confidence --tool-parity

# Run the reusable 20-command confidence/tool corpus
bun run --cwd packages/needle.js test:corpus

# Diagnostic one-submission token builder
bun run --cwd packages/needle.js bench -- --execution resident --single-submission
```

The harness bundles `scripts/bench-browser.ts`, serves it locally, and drives `Bun.WebView` with the `webkit` backend so Chrome is not launched with `--disable-gpu`.

## Automated resident correctness

```bash
bun run --cwd packages/needle.js test:browser
```

The separate resident suite runs real model weights through WKWebView/WebGPU and checks greedy parity, prompt confidence tolerance, the exact constrained flashlight call, 270-token KV-window/prefix-sink behavior, deterministic resets, float32-KV fallback, and cancellation. The `Resident browser correctness` GitHub workflow runs a capability-gated WKWebView check on macOS for relevant pushes and pull requests, caching the pinned model archive. GitHub-hosted WKWebView currently exposes no WebGPU adapter, so that runner reports a warning and skips the GPU assertions; local or self-hosted Apple runners with WebGPU execute the full suite. Chrome SwiftShader is not used as a parity substitute because its resident output currently differs from the hardware path.
