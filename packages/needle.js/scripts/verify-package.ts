#!/usr/bin/env bun
import { join } from "node:path";

const root = join(import.meta.dir, "..");
const manifest = await Bun.file(join(root, "package.json")).json();

function check(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

check(manifest.name === "needle.js", "unexpected package name");
check(
  typeof manifest.version === "string" &&
    /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$/.test(
      manifest.version,
    ),
  `invalid package version: ${manifest.version}`,
);
check(manifest.exports?.["./cpu"], "CPU export is missing");
check(manifest.exports?.["./typegpu"], "TypeGPU export is missing");
check(!manifest.exports?.["./vgpu"], "removed vgpu export is present");
check(manifest.peerDependencies?.typegpu, "TypeGPU peer dependency is missing");
check(manifest.peerDependenciesMeta?.typegpu?.optional === true, "TypeGPU peer must be optional");
check(!manifest.peerDependencies?.vgpu, "removed vgpu peer dependency is present");
check(manifest.files?.includes("ARCHITECTURE.md"), "architecture decision is not packed");

for (const path of [
  "dist/index.js",
  "dist/index.d.ts",
  "dist/cpu.js",
  "dist/cpu.d.ts",
  "dist/typegpu.js",
  "dist/typegpu.d.ts",
  "dist/resident/session.js",
  "README.md",
  "ARCHITECTURE.md",
  "LICENSE",
  "NOTICE",
]) {
  check(await Bun.file(join(root, path)).exists(), `required package file is missing: ${path}`);
}

const packedRoots = manifest.files as string[];
check(!packedRoots.includes("benchmarks"), "machine-specific benchmarks must not ship in npm");
check(!packedRoots.includes("scripts"), "development scripts must not ship in npm");

for await (const relative of new Bun.Glob("**/*").scan({
  cwd: join(root, "dist"),
  onlyFiles: true,
})) {
  check(
    !relative.toLowerCase().includes("vgpu"),
    `removed vgpu artifact remains: dist/${relative}`,
  );
  check(!relative.endsWith(".cact"), `model weights must not be packed: dist/${relative}`);
}

console.log("package files and publication invariants passed");
