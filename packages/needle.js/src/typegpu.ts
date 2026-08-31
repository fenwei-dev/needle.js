export * from "./index.js";
import { Needle, type NeedleOptions } from "./tools/agent.js";
import { NeedleModel, type LoadModelOptions } from "./model/model.js";
import type { TypeGPUBackendOptions } from "./backends/typegpu.js";

export async function loadNeedleTypeGPU(
  options: LoadModelOptions & { readonly backendOptions?: TypeGPUBackendOptions } = {},
): Promise<NeedleModel> {
  return NeedleModel.load({ ...options, backend: "typegpu" });
}

export async function createNeedleTypeGPU(
  options: NeedleOptions & { readonly backendOptions?: TypeGPUBackendOptions } = {},
): Promise<Needle> {
  return Needle.create({ ...options, backend: "typegpu" });
}

export {
  TypeGPUBackend,
  createTypeGPUBackend,
  type TypeGPUBackendOptions,
} from "./backends/typegpu.js";
