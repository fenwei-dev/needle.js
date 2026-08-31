export { CpuBackend, createCpuBackend } from "./backends/cpu.js";
export * from "./index.js";

import { type LoadModelOptions, NeedleModel } from "./model/model.js";
import { Needle, type NeedleOptions } from "./tools/agent.js";

export async function loadNeedleCPU(options: LoadModelOptions = {}): Promise<NeedleModel> {
  return NeedleModel.load({ ...options, backend: "cpu" });
}

export async function createNeedleCPU(options: NeedleOptions = {}): Promise<Needle> {
  return Needle.create({ ...options, backend: "cpu" });
}

export {
  cqMatvec,
  cqMatvecPrepared,
  denseMatvec,
  dequantizeCqRow,
  fastWalshHadamard,
  prepareCqActivation,
} from "./backends/cq.js";
