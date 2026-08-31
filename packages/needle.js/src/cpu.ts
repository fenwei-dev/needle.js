export * from "./index.js";
export { CpuBackend, createCpuBackend } from "./backends/cpu.js";
import { Needle, type NeedleOptions } from "./tools/agent.js";
import { NeedleModel, type LoadModelOptions } from "./model/model.js";

export async function loadNeedleCPU(options: LoadModelOptions = {}): Promise<NeedleModel> {
  return NeedleModel.load({ ...options, backend: "cpu" });
}

export async function createNeedleCPU(options: NeedleOptions = {}): Promise<Needle> {
  return Needle.create({ ...options, backend: "cpu" });
}

export {
  cqMatvec,
  cqMatvecPrepared,
  dequantizeCqRow,
  prepareCqActivation,
  fastWalshHadamard,
  denseMatvec,
} from "./backends/cq.js";
