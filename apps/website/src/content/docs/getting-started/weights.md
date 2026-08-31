---
title: Weights and deployment
description: Download, cache, embed, and ship Needle 2 model archives.
---

A Needle deployment is one self-contained `.cact` archive: geometry, mixed-bit matrices, control tensors, probe heads, and tokenizer. The official base model is 13,737,807 bytes.

## Download once

```ts
const model = await NeedleModel.load({ weights: "download" });
```

The built-in URL is revision-pinned and verified against an expected SHA-256. Node and Bun cache under:

```text
~/.cache/needle.js/models/<sha256>.cact
```

Browsers use Cache Storage when available. Inference itself does not contact the network after loading.

## Local and tuned archives

```ts
await NeedleModel.load({ weights: "./models/needle2.cact" });
await NeedleModel.load({ weights: new Uint8Array(modelBytes) });
await NeedleModel.load({
  weights: {
    kind: "url",
    url: "https://example.test/tuned.cact",
    sha256: "…64 hexadecimal characters…",
    cache: true,
  },
});
```

File paths are Node/Bun-only. URLs, buffers, and typed-array views work in browsers.

## Embed weights in a build

From the monorepo or a source checkout:

```bash
bun run --cwd packages/needle.js embed:model /path/to/needle2.cact
bun run --cwd packages/needle.js build
```

The generator writes chunked base64 and its integrity hash to the embedded-model module. Load it with:

```ts
await NeedleModel.load({ weights: "embedded" });
```

Restore the lightweight placeholder before publishing a download-first build:

```bash
bun run --cwd packages/needle.js embed:model --clear
```

:::caution
Embedding increases the JavaScript payload by roughly the base64 size of the model. A separate cacheable `.cact` asset is usually better for websites; embedding is useful for single-artifact or offline deployments.
:::

## Air-gapped applications

Resolve weights before disconnecting, copy the verified archive with your application, and pass either its file path or bytes. No tokenizer or configuration file is fetched separately.

The Needle 2 weights are Apache-2.0 and are not committed to this repository. Keep their license with redistributed or embedded builds.
