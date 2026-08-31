export * from "./index.js";
import { Needle, type NeedleOptions } from "./tools/agent.js";
import { NeedleModel, type LoadModelOptions } from "./model/model.js";
import type { VGPUBackendOptions } from "./backends/vgpu.js";

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
  VGPUBackend,
  createVGPUBackend,
  type VGPUBackendOptions,
} from "./backends/vgpu.js";
