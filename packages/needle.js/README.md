# needle.js

A TypeScript inference library for **Needle 2**, Cactus Compute's 45M-parameter tool-calling model. It reads the official self-contained `.cact` file directly and runs in Node.js, Bun, and modern browsers.

> **Unofficial implementation:** needle.js is an independent project and is not affiliated with or endorsed by Cactus Compute.

The package provides three interchangeable implementations:

| Entry point | Backend | Extra requirement |
| --- | --- | --- |
| `needle.js/cpu` | Pure TypeScript | None |
| `needle.js/typegpu` | TypeGPU + WebGPU | `typegpu` and a WebGPU device |
| `needle.js/vgpu` | vgpu + WebGPU | `vgpu` and a WebGPU device |

All three use the same tokenizer, incremental decoder, conversation API, and continuous byte-level JSON Schema grammar. The GPU implementations offload sufficiently large packed Cactus-Quant matrix-vector products to WebGPU while TypeScript handles small projections, recurrent operations, attention, and grammar.

## Implemented

- Official 120-byte `.cact` v2 header and nameless tensor directory
- Mixed CQ2/CQ3/CQ4/ternary weights, without expanding the model in memory
- Embedded SentencePiece BPE tokenizer and byte fallback
- Needle 2 SAN architecture:
  - grouped-query attention and RoPE
  - zero-centered RMS normalization and gated sandwich residuals
  - four-lane mHC routing with log-space Sinkhorn normalization
  - Hadamard MLP
  - hashed n-gram engram memory and causal convolution taps
  - int8 KV cache, 256-token sliding window, and pinned prefix sinks
- Greedy and temperature/top-k text generation
- Schema-constrained tool calling for strings, numbers, integers, booleans, nulls, arrays, nested objects, enums, constants, required/optional fields, and common range/length constraints
- Agentic tool execution loop, multi-call turns, follow-up tool results, extraction, BM25 tool retrieval, and the exported confidence head
- In-memory weights, generated embedded weights, local files, HTTP download, SHA-256 verification, Node disk cache, and browser Cache Storage

The independent TypeScript implementation was developed using the public architecture, deployment-format, tokenizer, quantization, and decoding specifications in the [official Needle repository](https://github.com/cactus-compute/needle). [MimiModel](https://github.com/memovai/mimimodel) was also consulted as a secondary validation reference. See [NOTICE](NOTICE) for attribution and revision details.

## Installation

```bash
npm install needle.js
# or
bun add needle.js
```

For a GPU backend, install its optional peer:

```bash
npm install typegpu       # TypeGPU version
# or
npm install vgpu          # vgpu version
```

Node-side WebGPU initialization depends on the selected package and platform. A browser can use `navigator.gpu`; both backends also accept an existing `GPUDevice`.

## Quick start: tool calling

```ts
import { createNeedle, defineTool } from "needle.js";

const getWeather = defineTool<
  { city: string; units?: "celsius" | "fahrenheit" },
  { city: string; temperature: number; units: string }
>({
  name: "get_weather",
  description: "Get the current weather for a city",
  parameters: {
    type: "object",
    properties: {
      city: {
        type: "string",
        description: "City whose current weather is requested",
      },
      units: {
        type: "string",
        enum: ["celsius", "fahrenheit"],
      },
    },
    required: ["city"],
  },
  execute: async ({ city, units = "celsius" }) => ({
    city,
    temperature: 27,
    units,
  }),
});

const agent = await createNeedle({
  // Downloads the pinned 13.7 MB archive once and caches it.
  weights: "download",
  backend: "cpu",
  tools: [getWeather],
  system: "date: 2026-08-31; locale: en-US; device: laptop",
  onProgress: ({ loaded, total }) => {
    console.log(`model: ${loaded}/${total ?? "?"} bytes`);
  },
});

const response = await agent.run("What's the weather in Lagos right now?");
console.log(response.functionCalls);
console.log(response.results);
console.log(response.confidence);

await agent.dispose();
```

`complete()` performs one model turn and leaves execution to the caller. `run()` executes registered implementations and feeds their results back until the model responds or `maxSteps` is reached.

```ts
const turn = await agent.complete("Set the study lights to 30 percent");
if (turn.type === "call") {
  console.log(turn.functionCalls[0]);
}

agent.reset(); // clear conversation state, retain model and tools
```

The response includes camel-case `functionCalls` and an official-envelope-compatible `function_calls` alias.

## Pure TypeScript, TypeGPU, and vgpu

### Pure TypeScript

```ts
import { createNeedleCPU } from "needle.js/cpu";

const agent = await createNeedleCPU({
  weights: "download",
  tools: [getWeather],
});
```

No native addon, WASM module, BLAS library, or WebGPU implementation is required.

### TypeGPU

```ts
import { createNeedleTypeGPU } from "needle.js/typegpu";

const agent = await createNeedleTypeGPU({
  weights: "download",
  tools: [getWeather],
  backendOptions: {
    // Optional: reuse a TgpuRoot or adopt a GPUDevice.
    // root,
    // device,
    init: { device: { optionalFeatures: ["shader-f16"] } },
    // Matvecs below this row count stay on CPU (default: 1024).
    // minimumGpuRows: 0, // force every matvec onto WebGPU
    // fusedAttention: true, // experimental resident int8 KV + attention
    // fusedMlp: true, // also fuse sandwich residuals and Hadamard MLP
    // fusedRouting: true, // also fuse post-mHC Sinkhorn and nextX lanes
    // residentLayers: true, // retain nextX and execute complete layers on GPU
  },
});
```

The backend acquires its device through `tgpu.init()` or adopts one through `tgpu.initFromDevice()`.

### vgpu

```ts
import { createNeedleVGPU } from "needle.js/vgpu";

const agent = await createNeedleVGPU({
  weights: "download",
  tools: [getWeather],
  backendOptions: {
    // Optional: gpu, device, init, or node: true
    // minimumGpuRows: 0, // force every matvec onto WebGPU
  },
});
```

In Node, the vgpu backend uses `vgpu/node`; in a browser it uses `vgpu`. Passing `gpu` keeps its lifetime caller-owned. Passing `device` adopts the device via vgpu's `initFromDevice()` API.

You can also use the generic factory:

```ts
const model = await NeedleModel.load({ backend: "auto", weights: "download" });
```

`auto` tries TypeGPU and then vgpu when browser WebGPU is visible, and otherwise falls back to the pure TypeScript backend. Explicit GPU selection throws a useful error rather than silently changing backend.

## Benchmarks

Greedy generation of 32 tokens in Bun.WebView (WKWebView + WebGPU, macOS arm64, Bun 1.4.0), same `.cact` weights and prompt for every backend:

| Backend | Median tok/s | Median ms / 32 tokens |
| --- | ---: | ---: |
| Pure TypeScript | 44.2 | 725 |
| TypeGPU | **50.8** | 630 |
| vgpu | 50.5 | 634 |

The WebGPU shader reduces each CQ row across a 32-lane workgroup. Independent mHC, Q/K/V/gate, and engram projections are packed into one matrix arena and executed with one dispatch and one host synchronization per group. By default, projections below 1,024 output rows still stay on CPU because their readback cost exceeds their compute time; for the official model, WebGPU therefore handles the 8,192-row vocabulary projection. Forcing every matvec onto WebGPU now measures 21.0 tok/s with TypeGPU and 23.3 tok/s with vgpu, up from 6.68 and 11.33 tok/s before batching. Reproduce with:

```bash
bun run bench
bun run bench -- --tokens 32 --warmup 1 --runs 2 --json
bun run bench -- --minimum-gpu-rows 0 # diagnostic all-WebGPU matvec path
```

Methodology and notes: [backend benchmarks](https://fenwei-dev.github.io/needle.js/reference/benchmarks/).

TypeGPU also has opt-in residency experiments. `fusedAttention` keeps Q/K/V, int8 KV state, attention, gating, and output projection on GPU; `fusedMlp` adds sandwich residuals and both 512-point Hadamard transforms; `fusedRouting` adds h-post gates, 20-step 4×4 Sinkhorn routing, and four-lane `nextX` construction. The individual stages preserve greedy output but cost more than CPU while lane state still round-trips.

`residentLayers: true` removes that round trip. It retains all 2,048 lane values and performs mHC pre-projections/gating, optional engram injection, attention, MLP, post-routing, final RMSNorm, and vocabulary projection on GPU. Non-logit prefill steps have no layer/final readback. Raw generation can read the vocabulary once; high-level tool calling instead uploads the grammar's allowed token IDs and reads only the selected ID plus log probability.

The confidence path is resident too: eight online probe pools are updated from the embedding and every layer mean, and the final confidence projection returns one score. On the flashlight integration turn, CPU and resident TypeGPU produced the same call in 1,150 ms and 1,012 ms respectively; final confidence was 0.7526 vs 0.7563. Raw forced-all-GPU throughput measured **51.4 tok/s** on the standard prompt and 5.40 tok/s for a 98-token prompt, versus CPU's 41.6 and 4.68 tok/s in paired runs.

## Weight sources

### Download and cache

The default URL is revision-pinned on Hugging Face, and its expected byte count and SHA-256 are built in. Inference performs no network access after loading.

```ts
await NeedleModel.load({ weights: "download" });
```

Node/Bun cache path:

```text
~/.cache/needle.js/models/<sha256>.cact
```

Browsers use a `needle.js-models-v1` Cache Storage cache when available.

A custom URL can carry its own integrity metadata:

```ts
await NeedleModel.load({
  weights: {
    kind: "url",
    url: "https://example.test/my-needle.cact",
    sha256: "...64 hex characters...",
    cache: true,
  },
});
```

### Local file or in-memory bytes

```ts
await NeedleModel.load({ weights: "./models/needle2.cact" });
await NeedleModel.load({ weights: new Uint8Array(modelBytes) });
await NeedleModel.load({
  weights: { kind: "embedded", data: modelBytes, sha256: "..." },
});
```

File paths are available in Node/Bun. `Uint8Array`, typed-array views, and `ArrayBuffer` work everywhere and are convenient with bundler asset plugins.

### Generate a model-embedded build

Weights are not committed to this repository. To produce a build whose JavaScript contains the model:

```bash
bun run embed:model /path/to/needle2.cact
bun run build
```

This replaces `src/weights/embedded-model.ts` with chunked base64 and an integrity hash. Then:

```ts
const model = await NeedleModel.load({ weights: "embedded" });
```

Restore the lightweight placeholder with:

```bash
bun run embed:model --clear
```

This mechanism makes embedding explicit: applications that download or ship the `.cact` as a separate asset do not pay a 13.7 MB package/bundle cost. Any redistributed embedded build must retain the model's [Apache-2.0 license](LICENSES/Apache-2.0.txt), this package's [NOTICE](NOTICE), and applicable model attribution.

## Structured extraction

Extraction is a one-tool call:

```ts
import { extract } from "needle.js";

const invoice = await extract(
  "Invoice from Acme Corp, total $1,200, due 2026-09-01",
  {
    title: "invoice",
    description: "Invoice fields found in supplied text",
    type: "object",
    properties: {
      vendor: { type: "string" },
      total: { type: "number", minimum: 0 },
      due_date: { type: "string" },
    },
    required: ["vendor", "total"],
  },
  { weights: "download" },
);
```

## Raw generation

```ts
import { NeedleModel } from "needle.js";

const model = await NeedleModel.load({
  weights: "download",
  backend: "cpu",
});

const output = await model.generate("The most surprising thing about", {
  maxNewTokens: 64,
  temperature: 0,
  onToken: ({ piece }) => process.stdout.write(piece),
});

console.log(output.text, output.tokensPerSecond);
await model.dispose();
```

Needle 2 is trained for structured tool calls rather than general chat, so `Needle`/`run()` is normally the appropriate interface.

## Confidence gating

The base archive includes a calibrated confidence head. This library performs its probe pooling online, avoiding retention of all layer cells, and reports the minimum of:

1. the post-hoc confidence-head score; and
2. the least likely token selected for the constrained call.

Use a product-specific threshold before actions with side effects:

```ts
const response = await agent.complete(command);
if (response.confidence !== null && response.confidence >= 0.8) {
  // execute, otherwise ask again or escalate
}
```

As in the official package, a base-model confidence calibration should not be assumed valid after fine-tuning.

## Lower-level API

The package exports the format parser, tokenizer, runtime, CPU CQ kernels, and grammar components:

```ts
import {
  parseCact,
  NeedleTokenizer,
  NeedleRuntime,
  JsonSchemaGrammar,
  ToolCallGrammar,
} from "needle.js";
```

A custom backend implements `InferenceBackend`: selected-row CQ matvec, optional independent-matvec batching, row gather, and disposal.

## Development

```bash
bun install
bun run typecheck
bun test
bun run build
bun run bench
```

Run the parity integration test against an official archive:

```bash
NEEDLE_MODEL_PATH=/path/to/needle2.cact bun test tests/model.integration.test.ts
```

The fixture checks the published 8,192-token/27-layer geometry and an exact constrained flashlight call.

## Attribution and licenses

- All `needle.js` TypeScript source code is MIT licensed; see [LICENSE](LICENSE).
- The consulted upstream Needle source and separately distributed model weights are Apache-2.0.
- Needle 2 weights are **not included in this repository**. Redistributed embedded builds must retain the model's Apache-2.0 license and attribution.

See [NOTICE](NOTICE) for source references and revision details.
