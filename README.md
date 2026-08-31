# needle.js monorepo

[![Deploy website](https://github.com/fenwei-dev/needle.js/actions/workflows/deploy-website.yml/badge.svg)](https://github.com/fenwei-dev/needle.js/actions/workflows/deploy-website.yml)

TypeScript tooling for running [Needle 2](https://github.com/cactus-compute/needle), integrating it with agent frameworks, and learning how its inference engine works.

**Documentation:** <https://fenwei-dev.github.io/needle.js/>

## Workspaces

- [`packages/needle.js`](packages/needle.js) — Needle 2 inference, constrained tool calling, and CPU/TypeGPU/vgpu backends.
- [`packages/needle-ai-provider`](packages/needle-ai-provider) — Vercel AI SDK v7 provider for local Needle inference.
- [`packages/needle-pi-ai-provider`](packages/needle-pi-ai-provider) — native pi-ai provider and installable pi extension.
- [`apps/website`](apps/website) — Astro Starlight documentation and three local browser demos.

## Development

```bash
bun install
bun run typecheck
bun test
bun run build
```

Commands run across all workspaces that expose the corresponding script.

Start the documentation site with:

```bash
bun run --cwd apps/website dev
```
