#!/usr/bin/env bun
import { mkdir, rm } from "node:fs/promises";
import { join } from "node:path";

const packageRoot = join(import.meta.dir, "..");
const root = join(packageRoot, ".package-smoke");
const consumer = join(root, "consumer");
const tarball = join(root, "needle.js-0.1.0.tgz");
await rm(root, { recursive: true, force: true });
await mkdir(consumer, { recursive: true });

async function run(command: string[], cwd = packageRoot): Promise<string> {
  const process = Bun.spawn(command, { cwd, stdout: "pipe", stderr: "pipe" });
  const [stdout, stderr, exitCode] = await Promise.all([
    new Response(process.stdout).text(),
    new Response(process.stderr).text(),
    process.exited,
  ]);
  if (exitCode !== 0) throw new Error(`${command.join(" ")} failed:\n${stdout}${stderr}`);
  return stdout;
}

await run(["bun", "run", "build"]);
await run(["bun", "pm", "pack", "--destination", root, "--ignore-scripts"]);
const files = await run(["tar", "-tzf", tarball], root);
for (const required of [
  "package/package.json",
  "package/dist/index.js",
  "package/dist/cpu.js",
  "package/dist/typegpu.js",
  "package/dist/resident/session.js",
  "package/README.md",
  "package/ARCHITECTURE.md",
  "package/LICENSE",
]) {
  if (!files.includes(required)) throw new Error(`Packed tarball is missing ${required}`);
}
if (files.includes("vgpu")) throw new Error("Packed tarball still contains vgpu files");

await Bun.write(
  join(consumer, "package.json"),
  JSON.stringify(
    {
      private: true,
      type: "module",
      dependencies: {
        "needle.js": "file:../needle.js-0.1.0.tgz",
        typegpu: "0.12.4",
        typescript: "5.9.3",
      },
    },
    null,
    2,
  ),
);
await Bun.write(
  join(consumer, "smoke.mjs"),
  `import { NeedleModel, createNeedle } from "needle.js";
import { createNeedleCPU } from "needle.js/cpu";
import { createNeedleTypeGPU } from "needle.js/typegpu";

for (const [name, value] of Object.entries({ NeedleModel, createNeedle, createNeedleCPU, createNeedleTypeGPU })) {
  if (typeof value !== "function") throw new Error(name + " is not exported");
}
let removed = false;
try { await import("needle.js/vgpu"); } catch { removed = true; }
if (!removed) throw new Error("removed needle.js/vgpu export unexpectedly resolved");
console.log("package runtime imports passed");
`,
);
await Bun.write(
  join(consumer, "smoke.ts"),
  `import type { BackendSelection } from "needle.js";
import type { TypeGPUBackendOptions, TypeGPUExecution } from "needle.js/typegpu";

const backend: BackendSelection = "typegpu";
const execution: TypeGPUExecution = "resident";
const options: TypeGPUBackendOptions = { execution };
void backend;
void options;
`,
);
await Bun.write(
  join(consumer, "tsconfig.json"),
  JSON.stringify(
    {
      compilerOptions: {
        target: "ES2022",
        module: "ESNext",
        moduleResolution: "Bundler",
        strict: true,
        skipLibCheck: true,
        types: ["@webgpu/types"],
        noEmit: true,
      },
      include: ["smoke.ts"],
    },
    null,
    2,
  ),
);
await run(["bun", "install"], consumer);
await run(["bun", "run", "smoke.mjs"], consumer);
await run(["bunx", "tsc", "-p", "tsconfig.json"], consumer);
console.log(`package smoke passed: ${tarball}`);
