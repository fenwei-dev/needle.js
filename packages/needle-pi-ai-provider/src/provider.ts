import { createProvider, type Model, type Provider } from "@earendil-works/pi-ai";
import {
  createNeedlePiApi,
  NEEDLE_PI_API,
  type NeedlePiApiId,
  type NeedlePiApiOptions,
} from "./api.js";

export const NEEDLE_PI_PROVIDER_ID = "needle";
export const NEEDLE_PI_MODEL_ID = "needle-2";

export interface NeedlePiProviderOptions extends NeedlePiApiOptions {
  readonly providerId?: string;
  readonly providerName?: string;
  readonly modelId?: string;
  readonly modelName?: string;
}

export interface NeedlePiProvider extends Provider<NeedlePiApiId> {
  dispose(): Promise<void>;
}

export function createNeedlePiProvider(options: NeedlePiProviderOptions = {}): NeedlePiProvider {
  const providerId = options.providerId ?? NEEDLE_PI_PROVIDER_ID;
  const model = createNeedlePiModel({
    providerId,
    modelId: options.modelId ?? NEEDLE_PI_MODEL_ID,
    modelName: options.modelName ?? "Needle 2 (Local)",
  });
  const api = createNeedlePiApi(options);
  const provider = createProvider({
    id: providerId,
    name: options.providerName ?? "Needle.js",
    baseUrl: "local://needle",
    auth: {
      apiKey: {
        name: "Local Needle model",
        async resolve() {
          return { auth: {}, source: "local model" };
        },
      },
    },
    models: [model],
    api,
  });
  return Object.assign(provider, { dispose: () => api.dispose() });
}

export function createNeedlePiModel(
  options: {
    readonly providerId?: string;
    readonly modelId?: string;
    readonly modelName?: string;
  } = {},
): Model<NeedlePiApiId> {
  return {
    id: options.modelId ?? NEEDLE_PI_MODEL_ID,
    name: options.modelName ?? "Needle 2 (Local)",
    api: NEEDLE_PI_API,
    provider: options.providerId ?? NEEDLE_PI_PROVIDER_ID,
    baseUrl: "local://needle",
    reasoning: true,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 2_048,
    maxTokens: 512,
  };
}

export const needlePiProvider = createNeedlePiProvider;
