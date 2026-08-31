export * from "./index.js";

import type { VGPUBackendOptions } from "./backends/vgpu.js";
import { type LoadModelOptions, NeedleModel } from "./model/model.js";
import { Needle, type NeedleOptions } from "./tools/agent.js";

export async function loadNeedleVGPU(
  options: LoadModelOptions & { readonly backendOptions?: VGPUBackendOptions } = {},
): Promise<NeedleModel> {
  return NeedleModel.load({ ...options, backend: "vgpu" });
}

export async function createNeedleVGPU(
  options: NeedleOptions & { readonly backendOptions?: VGPUBackendOptions } = {},
): Promise<Needle> {
  return Needle.create({ ...options, backend: "vgpu" });
}

export {
  createVGPUBackend,
  VGPUBackend,
  type VGPUBackendOptions,
} from "./backends/vgpu.js";
