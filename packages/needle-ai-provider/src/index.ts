export type { NeedleModelPoolOptions } from "./model-pool.js";
export {
  type NeedleAgentLike,
  NeedleLanguageModel,
  type NeedleLanguageModelConfig,
  type NeedleModelId,
} from "./needle-language-model.js";
export {
  type ConvertedNeedlePrompt,
  convertPrompt,
  type NeedlePromptEvent,
} from "./prompt.js";
export {
  createNeedleProvider,
  type NeedleProvider,
  type NeedleProviderSettings,
  needle,
} from "./provider.js";
