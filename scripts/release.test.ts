import { describe, expect, test } from "bun:test";
import {
  isPublished,
  packageDirectory,
  type ReleaseManifest,
  registry,
  releasePackages,
  repository,
  tarballName,
  validateRelease,
} from "./release";

function manifests(): ReleaseManifest[] {
  return releasePackages.map(({ name, directory }) => ({
    name,
    version: "1.2.3",
    repository: { url: repository, directory },
    publishConfig: { access: "public", registry },
    ...(name === "@fenwei/needle-ai-provider"
      ? { peerDependencies: { "@fenwei/needle.js": ">=1.2.0" } }
      : {}),
    ...(name === "@fenwei/pi-provider-needle"
      ? { dependencies: { "@fenwei/needle.js": "^1.2.0" } }
      : {}),
    exports: { ".": { import: "./dist/index.js", types: "./dist/index.d.ts" } },
  }));
}

function core(): ReleaseManifest {
  const manifest = manifests()[0];
  if (!manifest) throw new Error("Missing fixture");
  return manifest;
}

describe("scoped package layout", () => {
  test.each([
    ["@fenwei/needle.js", "packages/needle.js", "fenwei-needle.js-1.2.3.tgz"],
    [
      "@fenwei/needle-ai-provider",
      "packages/needle-ai-provider",
      "fenwei-needle-ai-provider-1.2.3.tgz",
    ],
    [
      "@fenwei/pi-provider-needle",
      "packages/pi-provider-needle",
      "fenwei-pi-provider-needle-1.2.3.tgz",
    ],
  ])("maps %s to its existing directory and flat tarball filename", (name, directory, filename) => {
    expect(packageDirectory(name)).toBe(directory);
    expect(tarballName({ name, version: "1.2.3" })).toBe(filename);
  });

  test("rejects unknown packages rather than deriving a filesystem path from the name", () => {
    expect(() => packageDirectory("@other/needle.js")).toThrow("Unknown release package");
    expect(() => packageDirectory("../needle.js")).toThrow("Unknown release package");
  });
});

describe("release validation", () => {
  test("accepts a matching stable tag and compatible adapter dependencies", () => {
    expect(validateRelease(manifests(), "refs/tags/v1.2.3", false)).toBe("1.2.3");
  });

  test("allows a branch only for dry runs", () => {
    expect(validateRelease(manifests(), "refs/heads/main", true)).toBe("1.2.3");
    expect(() => validateRelease(manifests(), "refs/heads/main", false)).toThrow("requires tag");
    expect(() => validateRelease(manifests(), "", false)).toThrow("requires tag");
  });

  test("rejects mismatched tags even for dry runs", () => {
    expect(() => validateRelease(manifests(), "refs/tags/v1.2.4", true)).toThrow("requires tag");
  });

  test("rejects mixed package versions", () => {
    const packages = manifests().map((m) => ({
      ...m,
      version: m.name === "@fenwei/needle.js" ? "1.2.4" : "1.2.3",
    }));
    expect(() => validateRelease(packages)).toThrow("versions must match");
  });

  test("does not put prereleases or invalid versions on latest", () => {
    for (const version of ["1.2.3-beta.1", "v1.2.3", "01.2.3", "1.2", "1.2.3+build"]) {
      expect(() => validateRelease(manifests().map((m) => ({ ...m, version })))).toThrow(
        "stable semver",
      );
    }
  });

  test("requires exactly the three public packages in dependency order", () => {
    expect(() => validateRelease(manifests().slice(1))).toThrow();
    expect(() => validateRelease(manifests().reverse())).toThrow("Unexpected");
    expect(() => validateRelease(manifests().map((m) => ({ ...m, private: true })))).toThrow(
      "private",
    );
  });

  test("rejects the old unscoped npm names", () => {
    expect(() =>
      validateRelease(manifests().map((m) => ({ ...m, name: m.name.replace("@fenwei/", "") }))),
    ).toThrow("Unexpected");
  });

  test("keeps repository directories separate from scoped npm names", () => {
    expect(() =>
      validateRelease(
        manifests().map((m) => ({
          ...m,
          repository: { ...m.repository, directory: `packages/${m.name}` },
        })),
      ),
    ).toThrow("publication metadata");
  });

  test("requires correct repository metadata and public npm access", () => {
    expect(() =>
      validateRelease(
        manifests().map((m) => ({
          ...m,
          repository: { ...m.repository, url: "https://example.test/other" },
        })),
      ),
    ).toThrow("publication metadata");
    expect(() =>
      validateRelease(
        manifests().map((m) => ({
          ...m,
          publishConfig: { ...m.publishConfig, access: "restricted" },
        })),
      ),
    ).toThrow("publication metadata");
  });

  test("rejects adapters that do not accept the released core version", () => {
    const packages = manifests().map((m) =>
      m.name === "@fenwei/pi-provider-needle"
        ? { ...m, dependencies: { "@fenwei/needle.js": "^2.0.0" } }
        : m,
    );
    expect(() => validateRelease(packages)).toThrow("must accept @fenwei/needle.js");
  });
});

describe("publication workflow authentication", () => {
  test("requires OIDC and does not inject the bootstrap token", async () => {
    const contents = await Bun.file(
      new URL("../.github/workflows/publish-npm.yml", import.meta.url),
    ).text();
    const workflow = Bun.YAML.parse(contents) as {
      jobs: {
        publish: {
          environment: string;
          permissions: Record<string, string>;
          steps: { run?: string; env?: Record<string, string> }[];
        };
      };
    };
    const job = workflow.jobs.publish;
    const publishStep = job.steps.find((step) => step.run === "bun run release:publish");
    expect(job.environment).toBe("npm");
    expect(job.permissions["id-token"]).toBe("write");
    expect(publishStep?.env?.NODE_AUTH_TOKEN).toBe("");
    expect(contents).not.toContain("secrets.NPM_TOKEN");
  });
});

describe("npm publication preflight", () => {
  test("URL-encodes the scoped package name as one registry path segment", async () => {
    let requestedUrl = "";
    await isPublished(core(), async (url) => {
      requestedUrl = url;
      return new Response(null, { status: 404 });
    });
    expect(requestedUrl).toBe("https://registry.npmjs.org/%40fenwei%2Fneedle.js/1.2.3");
  });

  test("only 404 means the version has not been published", async () => {
    expect(await isPublished(core(), async () => new Response(null, { status: 404 }))).toBe(false);
  });

  test("skips an existing version from this repository on a retry", async () => {
    expect(await isPublished(core(), async () => Response.json(core()))).toBe(true);
  });

  test("fails closed on registry or authentication errors", async () => {
    for (const status of [401, 403, 429, 500, 503]) {
      await expect(isPublished(core(), async () => new Response(null, { status }))).rejects.toThrow(
        `HTTP ${status}`,
      );
    }
  });

  test("fails closed on network failures", async () => {
    await expect(
      isPublished(core(), async () => {
        throw new Error("network failure");
      }),
    ).rejects.toThrow("refusing to publish");
  });

  test("does not silently skip a conflicting package or malformed response", async () => {
    await expect(
      isPublished(core(), async () =>
        Response.json({ ...core(), repository: { url: "https://example.test/other" } }),
      ),
    ).rejects.toThrow("does not match");
    await expect(isPublished(core(), async () => new Response("invalid JSON"))).rejects.toThrow();
  });
});
