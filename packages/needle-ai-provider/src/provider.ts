import {
  type EmbeddingModelV4,
  type ImageModelV4,
  type LanguageModelV4,
  NoSuchModelError,
  type ProviderV4,
} from "@ai-sdk/provider";
import type { NeedleOptions } from "needle.js";
import { NeedleModelPool, type NeedleModelPoolOptions } from "./model-pool.js";
import {
  type NeedleAgentLike,
  NeedleLanguageModel,
  type NeedleLanguageModelConfig,
  type NeedleModelId,
} from "./needle-language-model.js";

export interface NeedleProviderSettings extends NeedleModelPoolOptions {
  readonly providerName?: string;
  readonly defaultModelId?: "needle-2" | "needle2";
  readonly agentOptions?: Omit<
    NeedleOptions,
    "model" | "weights" | "backend" | "backendOptions" | "tools" | "system"
  >;
  readonly _internal?: Pick<NeedleLanguageModelConfig, "createAgent" | "now" | "generateId">;
}

export interface NeedleProvider extends ProviderV4 {
  (modelId?: NeedleModelId): LanguageModelV4;
  languageModel(modelId: string): LanguageModelV4;
  model(modelId?: NeedleModelId): LanguageModelV4;
  embeddingModel(modelId: string): EmbeddingModelV4;
  imageModel(modelId: string): ImageModelV4;
  dispose(): Promise<void>;
}

const SUPPORTED_MODEL_IDS = new Set(["needle-2", "needle2"]);

export function createNeedleProvider(settings: NeedleProviderSettings = {}): NeedleProvider {
  const providerName = settings.providerName ?? "needle";
  const defaultModelId = settings.defaultModelId ?? "needle-2";
  const pool = new NeedleModelPool(settings);

  const createLanguageModel = (modelId: string = defaultModelId): LanguageModelV4 => {
    if (!SUPPORTED_MODEL_IDS.has(modelId)) {
      throw new NoSuchModelError({ modelId, modelType: "languageModel" });
    }
    return new NeedleLanguageModel(modelId, {
      provider: `${providerName}.language-model`,
      pool,
      ...(settings.agentOptions === undefined ? {} : { agentOptions: settings.agentOptions }),
      ...(settings._internal?.createAgent === undefined
        ? {}
        : { createAgent: settings._internal.createAgent }),
      ...(settings._internal?.now === undefined ? {} : { now: settings._internal.now }),
      ...(settings._internal?.generateId === undefined
        ? {}
        : { generateId: settings._internal.generateId }),
    });
  };

  const noEmbeddingModel = (modelId: string): never => {
    throw new NoSuchModelError({ modelId, modelType: "embeddingModel" });
  };
  const noImageModel = (modelId: string): never => {
    throw new NoSuchModelError({ modelId, modelType: "imageModel" });
  };

  return Object.assign(createLanguageModel, {
    specificationVersion: "v4" as const,
    languageModel: createLanguageModel,
    model: createLanguageModel,
    embeddingModel: noEmbeddingModel,
    imageModel: noImageModel,
    dispose: () => pool.dispose(),
  });
}

export const needle = createNeedleProvider();

export type { NeedleAgentLike };
