---
title: Backend benchmarks
description: Measured Needle 2 generation throughput for the CPU, TypeGPU, and vgpu backends.
---

Needle 2 generation was timed in **Bun.WebView** (system WKWebView on macOS) so TypeGPU and vgpu use a real `navigator.gpu` device. All three backends loaded the same official 13.7 MB `.cact` archive and ran greedy generation from a shared prompt.

## Latest numbers

| Backend | Status | Load (ms) | Median tok/s | Median ms / 32 tokens |
| --- | --- | ---: | ---: | ---: |
| Pure TypeScript (`cpu`) | ok | 18 | **44.0** | 727 |
| TypeGPU + WebGPU | ok | 18 | 1.98 | 16,160 |
| vgpu + WebGPU | ok | 13 | 2.00 | 15,995 |

- **Host:** macOS `darwin arm64`, Bun 1.4.0 WebView
- **User agent:** `Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko)`
- **WebGPU:** available
- **Workload:** warmup 1 + 2 timed runs, 32 new tokens, temperature 0
- **Prompt:** `The most surprising thing about local inference is` (13 prompt tokens)

Both GPU backends produced the same greedy text as CPU on this prompt.

## How to read this

The GPU backends accelerate packed CQ **matrix–vector** products on WebGPU. Attention, RMSNorm, mHC routing, grammar, and sampling stay on the CPU, and each matvec currently **reads results back** before the next TypeScript operator. That hybrid loop is compact and easy to audit, but it is not a fused GPU engine: PCIe/IPC-style round trips dominate, so WebGPU is slower than the in-process CPU kernel on this machine.

Use **CPU** for throughput and portability. Use **TypeGPU** or **vgpu** when you want the same operators on a `GPUDevice`, or as a starting point for a future fully resident GPU runtime.

## Reproduce

From the monorepo, with Needle 2 weights on disk (`NEEDLE_MODEL_PATH` or `~/.cache/needle.js/models/<sha256>.cact`):

```bash
bun run --cwd packages/needle.js bench
bun run --cwd packages/needle.js bench -- --tokens 32 --warmup 1 --runs 2 --json
```

The harness bundles `scripts/bench-browser.ts`, serves it locally, and drives `Bun.WebView` with the `webkit` backend so Chrome is not launched with `--disable-gpu`.
