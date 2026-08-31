// @ts-check
import preact from "@astrojs/preact";
import starlight from "@astrojs/starlight";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "astro/config";

export default defineConfig({
  site: "https://fenwei-dev.github.io",
  base: "/needle.js",
  integrations: [
    preact(),
    starlight({
      title: "needle.js",
      description: "Needle 2 inference and tool calling in TypeScript",
      logo: { src: "./src/assets/logo.svg", replacesTitle: false },
      favicon: "/favicon.svg",
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/fenwei-dev/needle.js",
        },
      ],
      editLink: {
        baseUrl: "https://github.com/fenwei-dev/needle.js/edit/main/apps/website/",
      },
      customCss: ["./src/styles/global.css"],
      sidebar: [
        {
          label: "Overview",
          items: [
            { label: "Introduction", slug: "" },
            { label: "Browser demo", slug: "demo" },
          ],
        },
        {
          label: "Get started",
          items: [
            { label: "needle.js", slug: "getting-started/needle-js" },
            { label: "Weights and deployment", slug: "getting-started/weights" },
          ],
        },
        {
          label: "Integrations",
          items: [
            { label: "Vercel AI SDK", slug: "integrations/ai-sdk" },
            { label: "pi-ai and pi agent", slug: "integrations/pi-ai" },
          ],
        },
        {
          label: "Tutorials",
          items: [
            { label: "Inside the model", slug: "tutorials/model-structure" },
            { label: "The .cact format", slug: "tutorials/cact-format" },
            { label: "1. Loader and tokenizer", slug: "tutorials/engine-1-loader" },
            { label: "2. Kernels and token step", slug: "tutorials/engine-2-forward-pass" },
            { label: "3. Generation and grammar", slug: "tutorials/engine-3-generation" },
          ],
        },
        {
          label: "Reference",
          items: [
            { label: "Tool calling", slug: "reference/tool-calling" },
            { label: "Inference backends", slug: "reference/backends" },
            { label: "Licenses and attribution", slug: "reference/licenses" },
          ],
        },
      ],
    }),
  ],
  vite: {
    plugins: [tailwindcss()],
    resolve: {
      // The demo is browser-only; keep vgpu's optional Node adapter out of client chunks.
      alias: { "vgpu/node": "vgpu" },
    },
  },
});
