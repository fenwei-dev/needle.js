# Benchmark artifacts

Committed reports are immutable snapshots produced by the Bun.WebView harness.
They include every timed sample, environment metadata, options, and derived
summary statistics.

Reproduce the canonical workload with:

```bash
NEEDLE_MODEL_PATH=/path/to/needle2.cact \
  bun run --cwd packages/needle.js bench -- \
  --backends cpu,typegpu \
  --tokens 32 \
  --warmup 3 \
  --runs 10 \
  --execution resident \
  --json
```

Run the tool/confidence parity corpus with:

```bash
NEEDLE_MODEL_PATH=/path/to/needle2.cact \
  bun run --cwd packages/needle.js test:corpus
```

The corpus compares normalized calls and arguments, CPU/GPU confidence deltas,
and threshold decisions at 0.7, 0.8, and 0.9.

Results are machine-specific and should not be compared across different
hardware, browser engines, power states, or model revisions.
