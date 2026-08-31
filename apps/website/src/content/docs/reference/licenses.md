---
title: Licenses and attribution
description: Project licensing, upstream references, model redistribution requirements, and third-party notices.
---

needle.js is an independent implementation and is **not affiliated with or endorsed by Cactus Compute**.

## Project license

All TypeScript source code and original documentation in the needle.js monorepo are licensed under the MIT License:

| Component | License |
| --- | --- |
| `needle.js` core runtime | MIT |
| `needle-ai-provider` | MIT |
| `needle-pi-ai-provider` | MIT |
| Website application and documentation | MIT |

## Upstream Needle reference

The public Needle 2 repository and model specification were consulted for interoperability, architecture details, model geometry, tokenizer behavior, quantization layout, and the `.cact` deployment format.

- Repository: [Cactus Compute/needle](https://github.com/cactus-compute/needle)
- Inspected revision: `ee221ce7c13579d9809209b979a9b7a50936614c`
- Upstream license: [Apache License 2.0](/needle.js/licenses/Apache-2.0.txt)

The upstream license does not replace the MIT license used for the independent TypeScript implementation.

## Model weights

The official archive is downloaded separately from [Cactus-Compute/needle2](https://huggingface.co/Cactus-Compute/needle2). It is not committed to this source repository.

The model weights are Apache-2.0 licensed. Applications that redistribute an embedded or copied model must retain the model's Apache-2.0 license and applicable attribution. The model revision pinned by the library is:

```text
98fbd955b0347e78059be0c253cc1ffa09b87bc7
```

## Complete notices

- [Monorepo NOTICE](/needle.js/licenses/NOTICE.txt)
- [Apache License 2.0](/needle.js/licenses/Apache-2.0.txt)
