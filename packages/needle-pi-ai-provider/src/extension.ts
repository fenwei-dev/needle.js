import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import type { BackendSelection } from "needle.js";
import { createNeedlePiProvider } from "./provider.js";

export default function needlePiExtension(pi: ExtensionAPI): void {
  const provider = createNeedlePiProvider({
    weights: process.env.NEEDLE_MODEL_PATH ?? "download",
    backend: readBackend(process.env.NEEDLE_BACKEND),
  });

  pi.registerProvider(provider);
  pi.on("session_shutdown", async () => {
    await provider.dispose();
  });
}

function readBackend(value: string | undefined): BackendSelection {
  if (value === "typegpu" || value === "vgpu" || value === "auto") return value;
  return "cpu";
}
