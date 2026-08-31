# needle.js monorepo

TypeScript tooling for running [Needle 2](https://github.com/cactus-compute/needle), integrating it with agent frameworks, and learning how its inference engine works.

## Workspaces

- [`packages/needle.js`](packages/needle.js) — Needle 2 inference, constrained tool calling, and CPU/TypeGPU/vgpu backends.
- `packages/needle-ai-provider` — Vercel AI SDK provider (planned).
- `packages/needle-pi-ai-provider` — pi-ai provider (planned).
- `apps/website` — Starlight documentation and browser demos (planned).

## Development

```bash
bun install
bun run typecheck
bun test
bun run build
```

Commands run across all workspaces that expose the corresponding script.
