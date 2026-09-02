#!/usr/bin/env bun
import { mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { loadWeights } from "../src/weights/source.ts";

const scripts = dirname(fileURLToPath(import.meta.url));
const packageRoot = join(scripts, "..");
const outdir = join(packageRoot, ".resident-test");
mkdirSync(outdir, { recursive: true });

const modelSource = process.env.NEEDLE_MODEL_PATH ?? "download";
process.stderr.write(`loading resident browser fixture (${modelSource})...\n`);
const model = await loadWeights(modelSource, {
  onProgress: ({ loaded, total, cached }) => {
    if (total && process.stderr.isTTY)
      process.stderr.write(`\rmodel ${loaded}/${total}${cached ? " (cached)" : ""}`);
  },
});
process.stderr.write("\n");
const modelBody = new Uint8Array(model).slice().buffer;

const build = await Bun.build({
  entrypoints: [join(scripts, "resident-browser-suite.ts")],
  outdir,
  target: "browser",
  format: "esm",
  sourcemap: "none",
});
if (!build.success) {
  console.error(build.logs);
  throw new Error("Failed to bundle resident browser suite");
}

const html = `<!doctype html><meta charset="utf-8"><title>needle resident suite</title>
<script type="module" src="/resident-browser-suite.js"></script>`;
const server = Bun.serve({
  port: 0,
  fetch(request) {
    const pathname = new URL(request.url).pathname;
    if (pathname === "/")
      return new Response(html, { headers: { "content-type": "text/html; charset=utf-8" } });
    if (pathname === "/resident-browser-suite.js")
      return new Response(Bun.file(join(outdir, "resident-browser-suite.js")), {
        headers: { "content-type": "text/javascript; charset=utf-8" },
      });
    if (pathname === "/model.cact")
      return new Response(modelBody, {
        headers: { "content-type": "application/octet-stream" },
      });
    return new Response("not found", { status: 404 });
  },
});

const origin = `http://127.0.0.1:${server.port}`;
const backend =
  process.env.RESIDENT_WEBVIEW_BACKEND ?? (process.platform === "darwin" ? "webkit" : "chrome");
const browserBackend =
  backend === "chrome"
    ? {
        type: "chrome" as const,
        url: false as const,
        argv: [
          "--disable-gpu=false",
          "--enable-unsafe-webgpu",
          "--enable-unsafe-swiftshader",
          "--use-angle=swiftshader",
          "--ignore-gpu-blocklist",
        ],
      }
    : "webkit";
const view = new Bun.WebView({ width: 800, height: 600, backend: browserBackend });
try {
  await view.navigate(origin);
  await view.evaluate(`(() => new Promise((resolve, reject) => {
    const started = Date.now();
    const poll = () => {
      if (typeof window.runNeedleResidentSuite === "function") return resolve(true);
      if (Date.now() - started > 30_000) return reject(new Error("resident suite did not load"));
      setTimeout(poll, 50);
    };
    poll();
  }))()`);
  const hasAdapter = await view.evaluate(
    "navigator.gpu?.requestAdapter().then((adapter) => Boolean(adapter)) ?? false",
  );
  if (!hasAdapter && process.env.RESIDENT_ALLOW_NO_WEBGPU === "1") {
    const message = `${backend} WebGPU adapter is unavailable; resident suite skipped`;
    console.log(JSON.stringify({ skipped: true, backend, reason: message }, null, 2));
    process.stderr.write(
      process.env.GITHUB_ACTIONS === "true" ? `::warning::${message}\n` : `${message}\n`,
    );
  } else {
    if (!hasAdapter) throw new Error(`${backend} WebGPU adapter is unavailable`);
    const result = await view.evaluate(
      `window.runNeedleResidentSuite(${JSON.stringify(`${origin}/model.cact`)})`,
    );
    console.log(JSON.stringify(result, null, 2));
    process.stderr.write("resident browser suite passed\n");
  }
} finally {
  view.close();
  server.stop();
}
