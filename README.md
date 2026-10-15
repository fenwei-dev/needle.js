# needle.js monorepo

[![Deploy website](https://github.com/fenwei-dev/needle.js/actions/workflows/deploy-website.yml/badge.svg)](https://github.com/fenwei-dev/needle.js/actions/workflows/deploy-website.yml)

Needle 2 inference for TypeScript with a pure TypeScript reference backend, a TypeGPU-resident WebGPU engine, agent-framework integrations, and an educational implementation guide.

> **Unofficial implementation:** This project is independent and is not affiliated with or endorsed by Cactus Compute.

**Documentation:** <https://needle.fenwei.dev/>

## Workspaces

- [`@fenwei/needle.js`](packages/needle.js) — Needle 2 inference, constrained tool calling, and CPU and TypeGPU backends.
- [`@fenwei/needle-ai-provider`](packages/needle-ai-provider) — Vercel AI SDK v7 provider for local Needle inference.
- [`@fenwei/pi-provider-needle`](packages/pi-provider-needle) — native pi-ai provider and installable pi extension.
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

## Publishing

The [Publish npm packages](.github/workflows/publish-npm.yml) workflow publishes `@fenwei/needle.js`, `@fenwei/needle-ai-provider`, and `@fenwei/pi-provider-needle` together when a matching `vX.Y.Z` tag is pushed. It validates all workspaces and smoke-tests the exact tarballs before publishing with provenance. Manual runs default to a dry run.

See [Publishing to npm](docs/publishing.md) for the one-time bootstrap, npm trusted-publisher setup, and release commands.

## Licensing

All source code and original documentation in this monorepo are MIT licensed:

| Component | License |
| --- | --- |
| `@fenwei/needle.js` | MIT |
| `@fenwei/needle-ai-provider` | MIT |
| `@fenwei/pi-provider-needle` | MIT |
| Website application and documentation | MIT |

The consulted upstream Needle source and separately distributed Needle 2 model weights are Apache-2.0. Embedded or redistributed model weights must retain that license. The Apache-2.0 text is in [LICENSES](LICENSES). See [NOTICE](NOTICE) for attribution and upstream revisions.
