# needle.js monorepo

[![Deploy website](https://github.com/fenwei-dev/needle.js/actions/workflows/deploy-website.yml/badge.svg)](https://github.com/fenwei-dev/needle.js/actions/workflows/deploy-website.yml)

Needle 2 inference for TypeScript with a pure TypeScript reference backend, a TypeGPU-resident WebGPU engine, agent-framework integrations, and an educational implementation guide.

> **Unofficial implementation:** This project is independent and is not affiliated with or endorsed by Cactus Compute.

**Documentation:** <https://fenwei-dev.github.io/needle.js/>

## Workspaces

- [`packages/needle.js`](packages/needle.js) — Needle 2 inference, constrained tool calling, and CPU and TypeGPU backends.
- [`packages/needle-ai-provider`](packages/needle-ai-provider) — Vercel AI SDK v7 provider for local Needle inference.
- [`packages/needle-pi-ai-provider`](packages/needle-pi-ai-provider) — native pi-ai provider and installable pi extension.
- [`apps/website`](apps/website) — Astro Starlight documentation and three local browser demos.

## Development

```bash
bun install
bun run typecheck
bun test
bun run build
bun run --cwd packages/needle.js bench
```

Commands run across all workspaces that expose the corresponding script.

Start the documentation site with:

```bash
bun run --cwd apps/website dev
```

## Licensing

All source code and original documentation in this monorepo are MIT licensed:

| Component | License |
| --- | --- |
| `packages/needle.js` | MIT |
| `packages/needle-ai-provider` | MIT |
| `packages/needle-pi-ai-provider` | MIT |
| Website application and documentation | MIT |

The consulted upstream Needle source and separately distributed Needle 2 model weights are Apache-2.0. Embedded or redistributed model weights must retain that license. The Apache-2.0 text is in [LICENSES](LICENSES). See [NOTICE](NOTICE) for attribution and upstream revisions.
