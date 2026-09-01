#!/usr/bin/env bun
/**
 * Compare Needle 2 inference backends in a real browser WebGPU context.
 *
 * GPU kernels need WebGPU, so this drives Bun.WebView (WKWebView on macOS)
 * instead of Bun's process-local JS. CPU is measured in the same page so the
 * three backends share tokenizer, weights, and host.
 *
 *   bun run --cwd packages/needle.js bench
 */
import { mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import type { BackendResult, BrowserBenchOptions } from "./bench-browser.ts";

type BackendName = "cpu" | "typegpu";

interface CliOptions {
  backends: BackendName[];
  tokens: number;
  warmup: number;
  runs: number;
  prompt: string;
  json: boolean;
  minimumGpuRows: number | undefined;
  execution: "adaptive" | "resident" | undefined;
  fusedAttention: boolean;
  fusedMlp: boolean;
  fusedRouting: boolean;
  residentLayers: boolean;
  singleTokenSubmission: boolean;
  collectConfidence: boolean;
  toolParity: boolean;
}

const DEFAULT_PROMPT = "The most surprising thing about local inference is";
const ROOT = dirname(fileURLToPath(import.meta.url));
const PACKAGE_ROOT = join(ROOT, "..");

function parseArgs(argv: string[]): CliOptions {
  const options: CliOptions = {
    backends: ["cpu", "typegpu"],
    tokens: 32,
    warmup: 1,
    runs: 2,
    prompt: DEFAULT_PROMPT,
    json: false,
    minimumGpuRows: undefined,
    execution: undefined,
    fusedAttention: false,
    fusedMlp: false,
    fusedRouting: false,
    residentLayers: false,
    singleTokenSubmission: false,
    collectConfidence: false,
    toolParity: false,
  };
  for (let index = 0; index < argv.length; index++) {
    const argument = argv[index];
    const next = argv[index + 1];
    if (argument === "--json") options.json = true;
    else if (argument === "--confidence") options.collectConfidence = true;
    else if (argument === "--tool-parity") options.toolParity = true;
    else if (argument === "--fused-attention") options.fusedAttention = true;
    else if (argument === "--execution" && next) {
      if (next !== "adaptive" && next !== "resident")
        throw new Error(`Invalid execution mode ${next}`);
      options.execution = next;
      index++;
    } else if (argument === "--fused-mlp") {
      options.fusedAttention = true;
      options.fusedMlp = true;
    } else if (argument === "--fused-routing") {
      options.fusedAttention = true;
      options.fusedMlp = true;
      options.fusedRouting = true;
    } else if (argument === "--resident-layers") {
      options.fusedAttention = true;
      options.fusedMlp = true;
      options.fusedRouting = true;
      options.residentLayers = true;
    } else if (argument === "--single-submission") {
      options.fusedAttention = true;
      options.fusedMlp = true;
      options.fusedRouting = true;
      options.residentLayers = true;
      options.singleTokenSubmission = true;
    } else if (argument === "--backends" && next) {
      options.backends = next.split(",").map((name) => name.trim()) as BackendName[];
      index++;
    } else if (argument === "--tokens" && next) {
      options.tokens = Number(next);
      index++;
    } else if (argument === "--warmup" && next) {
      options.warmup = Number(next);
      index++;
    } else if (argument === "--runs" && next) {
      options.runs = Number(next);
      index++;
    } else if (argument === "--prompt" && next) {
      options.prompt = next;
      index++;
    } else if (argument === "--minimum-gpu-rows" && next) {
      options.minimumGpuRows = Number(next);
      index++;
    } else if (argument === "--help" || argument === "-h") {
      printHelp();
      process.exit(0);
    }
  }
  return options;
}

function printHelp(): void {
  console.log(`Usage: bun run scripts/bench.ts [options]

Runs inside Bun.WebView so TypeGPU can use browser WebGPU.

Options:
  --backends cpu,typegpu        Backends to measure
  --tokens <n>                  Generated tokens per run (default 32)
  --warmup <n>                  Warmup generations (default 1)
  --runs <n>                    Timed generations (default 2)
  --prompt <text>               Generation prompt
  --minimum-gpu-rows <n>        CPU fallback cutoff (default 1024; 0 = all GPU)
  --execution adaptive|resident Stable TypeGPU execution policy
  --fused-attention             TypeGPU GPU-resident attention/KV experiment
  --fused-mlp                   Also fuse sandwich residuals and Hadamard MLP
  --fused-routing               Also fuse post-mHC routing and nextX lanes
  --resident-layers             Retain lanes and run full layers without readback
  --single-submission           Submit all 27 resident layers together (diagnostic)
  --confidence                  Probe CPU/GPU confidence pooling on the prompt
  --tool-parity                 Run the constrained flashlight integration turn
  --json                        Print machine-readable JSON
`);
}

function median(values: number[]): number {
  if (values.length === 0) return Number.NaN;
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) {
    return ((sorted[middle - 1] ?? 0) + (sorted[middle] ?? 0)) / 2;
  }
  return sorted[middle] ?? Number.NaN;
}

function formatTable(results: BackendResult[]): string {
  const header = ["Backend", "Status", "Load (ms)", "Median tok/s", "Median ms", "Tokens"];
  const rows = results.map((result) => {
    const speeds = result.runs.map((run) => run.tokensPerSecond);
    const times = result.runs.map((run) => run.elapsedMs);
    const tokens = result.runs[0]?.generatedTokens ?? "";
    return [
      result.backend,
      result.status,
      result.loadMs?.toFixed(0) ?? "—",
      Number.isFinite(median(speeds)) ? median(speeds).toFixed(2) : "—",
      Number.isFinite(median(times)) ? median(times).toFixed(0) : "—",
      String(tokens),
    ];
  });
  const widths = header.map((cell, column) =>
    Math.max(cell.length, ...rows.map((row) => row[column]?.length ?? 0)),
  );
  const line = (cells: string[]) =>
    `| ${cells.map((cell, column) => cell.padEnd(widths[column] ?? 0)).join(" | ")} |`;
  const rule = `| ${widths.map((width) => "-".repeat(width)).join(" | ")} |`;
  return [line(header), rule, ...rows.map(line)].join("\n");
}

function defaultModelPath(): string {
  if (process.env.NEEDLE_MODEL_PATH) return process.env.NEEDLE_MODEL_PATH;
  return join(
    homedir(),
    ".cache/needle.js/models/b43aabfcaf1a6db6acf488076eab71d823c08697c7af4521fc1d174b60ede5ba.cact",
  );
}

const cli = parseArgs(process.argv.slice(2));
const modelPath = defaultModelPath();
const modelFile = Bun.file(modelPath);
if (!(await modelFile.exists())) {
  throw new Error(
    `Model not found at ${modelPath}. Set NEEDLE_MODEL_PATH or download weights first.`,
  );
}

const outdir = join(PACKAGE_ROOT, ".bench");
mkdirSync(outdir, { recursive: true });
const bundle = await Bun.build({
  entrypoints: [join(ROOT, "bench-browser.ts")],
  outdir,
  target: "browser",
  format: "esm",
  minify: false,
  sourcemap: "none",
});
if (!bundle.success) {
  console.error(bundle.logs);
  throw new Error("Failed to bundle browser benchmark");
}

const html = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>needle.js backend bench</title>
  </head>
  <body>
    <p>needle.js backend bench</p>
    <script type="module" src="/bench-browser.js"></script>
  </body>
</html>
`;

const server = Bun.serve({
  port: 0,
  async fetch(request) {
    const url = new URL(request.url);
    if (url.pathname === "/" || url.pathname === "/index.html") {
      return new Response(html, { headers: { "content-type": "text/html; charset=utf-8" } });
    }
    if (url.pathname === "/bench-browser.js") {
      return new Response(Bun.file(join(outdir, "bench-browser.js")), {
        headers: { "content-type": "text/javascript; charset=utf-8" },
      });
    }
    if (url.pathname === "/model.cact") {
      return new Response(modelFile, {
        headers: { "content-type": "application/octet-stream" },
      });
    }
    return new Response("not found", { status: 404 });
  },
});

const origin = `http://127.0.0.1:${server.port}`;
process.stderr.write(`serving ${origin} (WebView / WebGPU)\n`);

const view = new Bun.WebView({
  width: 800,
  height: 600,
  backend: "webkit",
});

let page: {
  webgpu: boolean;
  userAgent: string;
  results: BackendResult[];
};
try {
  await view.navigate(`${origin}/`);
  await view.evaluate(`(() => new Promise((resolve, reject) => {
    const started = Date.now();
    const tick = () => {
      if (typeof window.runNeedleBench === "function") return resolve(true);
      if (Date.now() - started > 30_000) return reject(new Error("bench module did not load"));
      setTimeout(tick, 50);
    };
    tick();
  }))()`);

  const benchOptions: BrowserBenchOptions = {
    backends: cli.backends,
    tokens: cli.tokens,
    warmup: cli.warmup,
    runs: cli.runs,
    prompt: cli.prompt,
    modelUrl: `${origin}/model.cact`,
    ...(cli.minimumGpuRows === undefined ? {} : { minimumGpuRows: cli.minimumGpuRows }),
    ...(cli.execution === undefined ? {} : { execution: cli.execution }),
    ...(cli.fusedAttention ? { fusedAttention: true } : {}),
    ...(cli.fusedMlp ? { fusedMlp: true } : {}),
    ...(cli.fusedRouting ? { fusedRouting: true } : {}),
    ...(cli.residentLayers ? { residentLayers: true } : {}),
    ...(cli.singleTokenSubmission ? { singleTokenSubmission: true } : {}),
    ...(cli.collectConfidence ? { collectConfidence: true } : {}),
    ...(cli.toolParity ? { toolParity: true } : {}),
  };

  process.stderr.write("running in WebView...\n");
  const payload = JSON.stringify(benchOptions);
  page = (await view.evaluate(`window.runNeedleBench(${payload})`)) as typeof page;
} finally {
  view.close();
  server.stop();
}

const report = {
  when: new Date().toISOString(),
  environment: {
    runtime: `Bun ${Bun.version} WebView`,
    platform: `${process.platform} ${process.arch}`,
    userAgent: page.userAgent,
    webgpu: page.webgpu,
  },
  options: {
    weights: modelPath,
    tokens: cli.tokens,
    warmup: cli.warmup,
    runs: cli.runs,
    prompt: cli.prompt,
    minimumGpuRows: cli.minimumGpuRows ?? 1024,
    execution: cli.execution ?? "adaptive",
    fusedAttention: cli.fusedAttention,
    fusedMlp: cli.fusedMlp,
    fusedRouting: cli.fusedRouting,
    residentLayers: cli.residentLayers,
    singleTokenSubmission: cli.singleTokenSubmission,
    collectConfidence: cli.collectConfidence,
    toolParity: cli.toolParity,
  },
  results: page.results,
};

if (cli.json) {
  console.log(JSON.stringify(report, null, 2));
} else {
  console.log("\nNeedle 2 backend benchmark (browser WebGPU)\n");
  console.log(`${report.environment.runtime} on ${report.environment.platform}`);
  console.log(`WebGPU: ${page.webgpu ? "yes" : "no"}`);
  console.log(`prompt tokens generated: ${cli.tokens}; warmup ${cli.warmup}; runs ${cli.runs}\n`);
  console.log(formatTable(page.results));
  for (const result of page.results) {
    if (result.confidence !== undefined)
      console.log(`${result.backend} confidence: ${result.confidence.toFixed(6)}`);
    if (result.toolRawCall !== undefined)
      console.log(
        `${result.backend} tool call: ${result.toolRawCall} (confidence ${result.toolConfidence}, ${result.toolElapsedMs?.toFixed(0)} ms)`,
      );
    if (result.error) console.log(`\n${result.backend}: ${result.error}`);
  }
}
