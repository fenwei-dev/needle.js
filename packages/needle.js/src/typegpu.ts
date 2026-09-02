export * from "./index.js";

import type { TypeGPUBackendOptions } from "./backends/typegpu.js";
import { type LoadModelOptions, NeedleModel } from "./model/model.js";
import { Needle, type NeedleOptions } from "./tools/agent.js";

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
  createTypeGPUBackend,
  TypeGPUBackend,
  type TypeGPUBackendOptions,
  type TypeGPUDiagnostics,
  type TypeGPUExecution,
  type TypeGPUResidentStage,
} from "./backends/typegpu.js";
