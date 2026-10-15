# Publishing to npm

The [Publish npm packages](../.github/workflows/publish-npm.yml) GitHub Actions workflow publishes these public packages in dependency order:

1. `@fenwei/needle.js`
2. `@fenwei/needle-ai-provider`
3. `@fenwei/pi-provider-needle`

The private monorepo root and website are never published. Releases currently use one stable version across all three packages. Prereleases are rejected rather than accidentally published under `latest`.

## One-time setup and first publication

npm trusted publishing is configured in an existing package's settings. Since these are new packages, bootstrap their first publication with a short-lived npm granular access token:

1. Ensure your npm account has permission to create and publish public packages in the **`@fenwei`** scope, including all three names above. If `fenwei` is an npm organization, your account needs the appropriate organization/team permissions. A registry 404 alone does not confirm publishing permission.
2. Create a GitHub environment named **`npm`** in `fenwei-dev/needle.js`. Configure required reviewers and restrict deployment to release tags as appropriate. Allow `main` too if you want manual dry runs from that branch.
3. Create a short-lived npm granular token with read/write permission for these packages in the **`@fenwei`** scope and **bypass 2FA** enabled for non-interactive publishing. Add it directly in GitHub as an environment secret named **`NPM_TOKEN`** (a repository secret also works). An existing token limited to unscoped packages may need its permissions updated. Never commit the token or put it in a workflow input, command line, or chat.
4. Commit and push the workflow and release changes to `main`, then follow the release steps below using an unused release version and tag.
5. After all three packages exist on npm, configure trusted publishing for each package as described below. Remove the GitHub `NPM_TOKEN` secret and revoke the bootstrap token when the transition is verified.

The secret is exposed only to the final publication step. Builds, tests, clean-consumer installs, and dry runs do not receive it. npm lifecycle scripts are disabled when publishing the already-built tarballs.

## Configure npm trusted publishing

On npmjs.com, open **Settings → Trusted publishing** for each of the three packages and add a GitHub Actions publisher:

| Setting | Value |
| --- | --- |
| Organization or user | `fenwei-dev` |
| Repository | `needle.js` |
| Workflow filename | `publish-npm.yml` |
| Environment name | `npm` |
| Allowed actions | Enable direct publishing with `npm publish` |

Enter only the workflow filename, not `.github/workflows/publish-npm.yml`. The npm scope is `@fenwei`, but the GitHub organization is still `fenwei-dev`; use the GitHub identity shown above for trusted publishing. Current npm configurations default to staged publishing; this workflow uses direct `npm publish`, so explicitly allow that action.

The workflow uses GitHub-hosted runners, Node.js 24, npm 11 (trusted publishing requires npm ≥11.5.1), and `id-token: write`. npm obtains short-lived credentials through OIDC. Every actual publish requests provenance, and each manifest declares the matching public repository and workspace directory.

After verifying OIDC publication, npm recommends setting package publishing access to **Require two-factor authentication and disallow tokens**. Protect release tags and require environment approval to control who can publish.

References: [npm trusted publishing](https://docs.npmjs.com/trusted-publishers/) and [npm provenance](https://docs.npmjs.com/generating-provenance-statements/).

## Validate without publishing

Locally:

```bash
bun install --frozen-lockfile
bun run typecheck
bun run test
bun run test:package
bun run release:dry-run
```

`test:package` builds all three workspaces, verifies the core package invariants, and creates `.release/*.tgz` with Bun's workspace-aware packer. It installs those exact tarballs in a clean npm consumer and checks Node.js imports, Bun imports, and TypeScript declarations, including the pi extension entry point. No model download or inference is needed.

Scoped npm names are independent of workspace directories: for example, `@fenwei/needle.js` lives in `packages/needle.js` and packs to `fenwei-needle.js-VERSION.tgz`.

The npm dry run uses those tarballs and does not publish anything. Actual publication is guarded so it can run only in this repository's GitHub Actions workflow on a matching release tag.

For a CI dry run, open **Actions → Publish npm packages → Run workflow**, select `main`, and leave **dry_run** checked. The separate **Validate package release** workflow also validates all three tarballs without publishing. Both workflows retain the tarballs as artifacts, not the clean consumer or its dependencies.

## Release

1. Update `version` in all three `packages/*/package.json` files to the same stable version. Update adapter dependency/peer ranges if needed so they accept the new `@fenwei/needle.js` version.
2. Run `bun install` to refresh `bun.lock`, then run the validation commands above.
3. Commit and push the release changes, using a Conventional Commit message such as `chore(release): prepare v0.1.1`.
4. Tag that exact commit and push the tag (replace the example with the chosen version):

   ```bash
   git tag v0.1.1
   git push origin v0.1.1
   ```

5. Approve the `npm` environment deployment if required, and monitor **Publish npm packages**.

The tag must exactly match all three manifest versions. Branch pushes do not publish. To manually publish or retry, select the matching tag in **Run workflow** and uncheck **dry_run**. A non-dry run on a branch is rejected before installing dependencies.

Do not move an existing release tag, even if its workflow failed. Rerunning the same tag retries the same source commit; package-name or source changes require a fresh version and tag.

The workflow checks the registry before publishing any package. Already-published versions from this repository are skipped, which allows rerunning after a partial failure. Authentication, network, malformed-response, and non-404 registry errors stop the release instead of being treated as missing versions. npm versions are immutable; if an already-published package needs changes, bump the release version rather than moving the tag.
