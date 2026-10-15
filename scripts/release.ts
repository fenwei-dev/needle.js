import { mkdir, rm } from "node:fs/promises";
import { join } from "node:path";

export const registry = "https://registry.npmjs.org/";
export const repository = "git+https://github.com/fenwei-dev/needle.js.git";
export const releasePackages = [
  { name: "@fenwei/needle.js", directory: "packages/needle.js" },
  { name: "@fenwei/needle-ai-provider", directory: "packages/needle-ai-provider" },
  { name: "@fenwei/pi-provider-needle", directory: "packages/pi-provider-needle" },
] as const;
const corePackageName = releasePackages[0].name;
const root = join(import.meta.dir, "..");
const output = join(root, ".release");

export interface ReleaseManifest {
  name: string;
  version: string;
  private?: boolean;
  repository: { url: string; directory: string };
  publishConfig: { access: string; registry: string };
  dependencies?: Record<string, string>;
  peerDependencies?: Record<string, string>;
  devDependencies?: Record<string, string>;
  exports: Record<string, string | Record<string, string>>;
}

export function packageDirectory(name: string): string {
  const entry = releasePackages.find((pkg) => pkg.name === name);
  if (!entry) throw new Error(`Unknown release package: ${name}`);
  return entry.directory;
}

export function tarballName(manifest: Pick<ReleaseManifest, "name" | "version">): string {
  return `${manifest.name.replace(/^@/, "").replaceAll("/", "-")}-${manifest.version}.tgz`;
}

export function validateRelease(manifests: ReleaseManifest[], ref = "", dryRun = true): string {
  const version = manifests[0]?.version;
  if (
    manifests.length !== releasePackages.length ||
    !version ||
    !/^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/.test(version)
  ) {
    throw new Error("Release requires three packages with a stable semver version");
  }
  for (const [index, manifest] of manifests.entries()) {
    if (manifest.name !== releasePackages[index]?.name || manifest.private) {
      throw new Error("Unexpected or private release package");
    }
    if (manifest.version !== version) {
      throw new Error("All three package versions must match");
    }
    if (
      manifest.repository?.url !== repository ||
      manifest.repository.directory !== packageDirectory(manifest.name) ||
      manifest.publishConfig?.access !== "public" ||
      manifest.publishConfig.registry !== registry
    ) {
      throw new Error(`Invalid publication metadata for ${manifest.name}`);
    }
    if (index > 0) {
      const range =
        manifest.dependencies?.[corePackageName] ?? manifest.peerDependencies?.[corePackageName];
      if (!range || !Bun.semver.satisfies(version, range)) {
        throw new Error(`${manifest.name} must accept ${corePackageName}@${version}`);
      }
    }
  }
  if ((!dryRun || ref.startsWith("refs/tags/")) && ref !== `refs/tags/v${version}`) {
    throw new Error(`Publishing requires tag v${version} matching every package version`);
  }
  return version;
}

export async function isPublished(
  manifest: ReleaseManifest,
  request: (url: string, options?: RequestInit) => Promise<Response> = fetch,
): Promise<boolean> {
  let response: Response;
  try {
    response = await request(
      `${registry}${encodeURIComponent(manifest.name)}/${manifest.version}`,
      {
        signal: AbortSignal.timeout(30_000),
      },
    );
  } catch {
    throw new Error(`Unable to check npm for ${manifest.name}; refusing to publish`);
  }
  if (response.status === 404) return false;
  if (response.status !== 200) {
    throw new Error(`npm lookup failed for ${manifest.name} (HTTP ${response.status})`);
  }
  const published = await response.json();
  if (
    published.name !== manifest.name ||
    published.version !== manifest.version ||
    published.repository?.url !== repository
  ) {
    throw new Error(`Existing npm metadata does not match this repository: ${manifest.name}`);
  }
  return true;
}

async function readManifests(): Promise<ReleaseManifest[]> {
  return Promise.all(
    releasePackages.map(({ directory }) => Bun.file(join(root, directory, "package.json")).json()),
  );
}

async function run(command: string[], cwd = root, capture = false): Promise<string> {
  const child = Bun.spawn(command, {
    cwd,
    stdout: capture ? "pipe" : "inherit",
    stderr: "inherit",
  });
  const stdout = capture ? await new Response(child.stdout).text() : "";
  if ((await child.exited) !== 0) throw new Error(`${command[0]} ${command[1]} failed`);
  return stdout;
}

function tarball(manifest: ReleaseManifest): string {
  return join(output, tarballName(manifest));
}

async function pack(manifests: ReleaseManifest[]): Promise<void> {
  await run(["bun", "run", "build:packages"]);
  await run(["bun", "run", "verify:package"], join(root, "packages/needle.js"));
  await rm(output, { recursive: true, force: true });
  await mkdir(output, { recursive: true });

  for (const manifest of manifests) {
    await run(
      ["bun", "pm", "pack", "--destination", output, "--ignore-scripts"],
      join(root, packageDirectory(manifest.name)),
      true,
    );
    const files = new Set(
      (await run(["tar", "-tzf", tarball(manifest)], root, true)).trim().split("\n"),
    );
    const exports = Object.values(manifest.exports).flatMap((value) =>
      typeof value === "string" ? [value] : Object.values(value),
    );
    const coreFiles =
      manifest.name === corePackageName
        ? ["ARCHITECTURE.md", "NOTICE", "LICENSES/Apache-2.0.txt", "dist/resident/session.js"]
        : [];
    for (const file of ["package.json", "README.md", "LICENSE", ...coreFiles, ...exports]) {
      if (!files.has(`package/${file.replace(/^\.\//, "")}`)) {
        throw new Error(`${manifest.name} tarball is missing ${file}`);
      }
    }
  }

  const consumer = join(output, "consumer");
  await mkdir(consumer);
  const ai = manifests[1];
  const pi = manifests[2];
  if (!ai || !pi) throw new Error("Provider manifests are missing");
  await Bun.write(
    join(consumer, "package.json"),
    JSON.stringify({
      private: true,
      type: "module",
      dependencies: {
        ...Object.fromEntries(manifests.map((m) => [m.name, `file:../${tarballName(m)}`])),
        ai: ai.devDependencies?.ai,
        "@earendil-works/pi-ai": pi.devDependencies?.["@earendil-works/pi-ai"],
        "@earendil-works/pi-coding-agent": pi.devDependencies?.["@earendil-works/pi-coding-agent"],
        typegpu: "0.12.4",
        typescript: "5.9.3",
      },
    }),
  );
  await Bun.write(
    join(consumer, "smoke.mjs"),
    `import { NeedleModel, createNeedle } from "@fenwei/needle.js";
import { createNeedleCPU } from "@fenwei/needle.js/cpu";
import { createNeedleTypeGPU } from "@fenwei/needle.js/typegpu";
import { hasEmbeddedWeights } from "@fenwei/needle.js/embedded";
import { createNeedleProvider } from "@fenwei/needle-ai-provider";
import { createNeedlePiProvider } from "@fenwei/pi-provider-needle";
import extension from "@fenwei/pi-provider-needle/extension";

for (const [name, value] of Object.entries({ NeedleModel, createNeedle, createNeedleCPU,
  createNeedleTypeGPU, hasEmbeddedWeights, createNeedleProvider, createNeedlePiProvider, extension })) {
  if (typeof value !== "function") throw new Error(name + " is not exported");
}
console.log("All three package tarballs import successfully");
`,
  );
  await Bun.write(
    join(consumer, "smoke.ts"),
    `import type { BackendSelection } from "@fenwei/needle.js";
import type { TypeGPUBackendOptions } from "@fenwei/needle.js/typegpu";
import type { NeedleProviderSettings } from "@fenwei/needle-ai-provider";
import type { NeedlePiProviderOptions } from "@fenwei/pi-provider-needle";
import extension from "@fenwei/pi-provider-needle/extension";

const backend: BackendSelection = "cpu";
const gpu: TypeGPUBackendOptions = { execution: "resident" };
const ai: NeedleProviderSettings = { backend };
const pi: NeedlePiProviderOptions = { backend };
void [gpu, ai, pi, extension];
`,
  );
  await Bun.write(
    join(consumer, "tsconfig.json"),
    JSON.stringify({
      compilerOptions: {
        target: "ES2022",
        module: "NodeNext",
        moduleResolution: "NodeNext",
        strict: true,
        skipLibCheck: true,
        types: ["@webgpu/types"],
        noEmit: true,
      },
      include: ["smoke.ts"],
    }),
  );
  await run(
    ["npm", "install", "--ignore-scripts", "--no-audit", "--no-fund", "--registry", registry],
    consumer,
  );
  await run(["node", "smoke.mjs"], consumer);
  await run(["bun", "run", "smoke.mjs"], consumer);
  await run([join(consumer, "node_modules/.bin/tsc"), "-p", "tsconfig.json"], consumer);
  console.log("Validated release tarballs are in .release/");
}

async function publish(manifests: ReleaseManifest[], dryRun: boolean): Promise<void> {
  if (!dryRun) {
    if (
      process.env.GITHUB_ACTIONS !== "true" ||
      process.env.GITHUB_REPOSITORY !== "fenwei-dev/needle.js"
    ) {
      throw new Error("Publish through the Publish npm packages GitHub Actions workflow");
    }
    validateRelease(manifests, process.env.GITHUB_REF, false);
  }
  // Finish all registry checks before publishing anything. Only an explicit 404 means unpublished.
  const pending: ReleaseManifest[] = [];
  for (const manifest of manifests) {
    if (!(await Bun.file(tarball(manifest)).exists())) {
      throw new Error("Missing tarballs; run bun run test:package first");
    }
    if (!dryRun && (await isPublished(manifest))) {
      console.log(`Already published: ${manifest.name}@${manifest.version}; skipping`);
    } else {
      pending.push(manifest);
    }
  }
  for (const manifest of pending) {
    await run([
      "npm",
      "publish",
      tarball(manifest),
      "--ignore-scripts",
      "--access",
      "public",
      "--registry",
      registry,
      "--tag",
      "latest",
      dryRun ? "--dry-run" : "--provenance",
    ]);
  }
}

if (import.meta.main) {
  const manifests = await readManifests();
  validateRelease(manifests);
  switch (process.argv[2]) {
    case "check": {
      const version = validateRelease(
        manifests,
        process.env.GITHUB_REF,
        process.env.RELEASE_DRY_RUN === "true",
      );
      console.log(`Release metadata validated: v${version}`);
      break;
    }
    case "pack":
      await pack(manifests);
      break;
    case "dry-run":
      await publish(manifests, true);
      break;
    case "publish":
      await publish(manifests, false);
      break;
    default:
      throw new Error("Usage: bun scripts/release.ts <check|pack|dry-run|publish>");
  }
}
