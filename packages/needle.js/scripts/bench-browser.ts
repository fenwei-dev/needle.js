import type { BackendSelection, GenerationResult } from "../src/index.js";
import { BOS_TOKEN_ID, Needle, NeedleModel } from "../src/index.js";

type BackendName = "cpu" | "typegpu" | "vgpu";

export interface BrowserBenchOptions {
  backends: BackendName[];
  tokens: number;
  warmup: number;
  runs: number;
  prompt: string;
  modelUrl: string;
  minimumGpuRows?: number;
  fusedAttention?: boolean;
  fusedMlp?: boolean;
  fusedRouting?: boolean;
  residentLayers?: boolean;
  singleTokenSubmission?: boolean;
  collectConfidence?: boolean;
  toolParity?: boolean;
}

export interface RunSample {
  elapsedMs: number;
  generatedTokens: number;
  tokensPerSecond: number;
  promptTokens: number;
}

export interface BackendResult {
  backend: BackendName;
  status: "ok" | "skipped" | "error";
  error?: string;
  loadMs?: number;
  warmupMs?: number;
  runs: RunSample[];
  generatedText?: string;
  finishReason?: GenerationResult["finishReason"];
  confidence?: number;
  toolRawCall?: string;
  toolConfidence?: number | null;
  toolElapsedMs?: number;
}

async function measureBackend(
  backend: BackendName,
  options: BrowserBenchOptions,
): Promise<BackendResult> {
  const startedLoad = performance.now();
  let model: NeedleModel;
  try {
    model = await NeedleModel.load({
      weights: { kind: "url", url: options.modelUrl, cache: false },
      backend: backend as BackendSelection,
      backendOptions:
        backend === "cpu"
          ? undefined
          : {
              ...(options.minimumGpuRows === undefined
                ? {}
                : { minimumGpuRows: options.minimumGpuRows }),
              ...(backend === "typegpu" && options.fusedAttention ? { fusedAttention: true } : {}),
              ...(backend === "typegpu" && options.fusedMlp ? { fusedMlp: true } : {}),
              ...(backend === "typegpu" && options.fusedRouting ? { fusedRouting: true } : {}),
              ...(backend === "typegpu" && options.residentLayers ? { residentLayers: true } : {}),
              ...(backend === "typegpu" && options.singleTokenSubmission
                ? { singleTokenSubmission: true }
                : {}),
            },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return {
      backend,
      status: /webgpu|gpu|adapter|typegpu|vgpu|BACKEND|not supported/i.test(message)
        ? "skipped"
        : "error",
      error: message,
      runs: [],
    };
  }

  const loadMs = performance.now() - startedLoad;
  let confidence: number | undefined;
  if (options.collectConfidence) {
    const runtime = model.createRuntime({ collectConfidence: true });
    const promptIds = [BOS_TOKEN_ID, ...model.tokenizer.encode(options.prompt)];
    runtime.reset({ maximumLength: promptIds.length });
    await runtime.prefill(promptIds);
    confidence = await runtime.resolveConfidence();
  }
  const generateOptions = { maxNewTokens: options.tokens, temperature: 0 as const };

  try {
    const warmupStarted = performance.now();
    for (let index = 0; index < options.warmup; index++) {
      await model.generate(options.prompt, generateOptions);
    }
    const warmupMs = options.warmup > 0 ? performance.now() - warmupStarted : 0;

    const runs: RunSample[] = [];
    let last: GenerationResult | undefined;
    for (let index = 0; index < options.runs; index++) {
      last = await model.generate(options.prompt, generateOptions);
      runs.push({
        elapsedMs: last.elapsedMs,
        generatedTokens: last.generatedTokens,
        tokensPerSecond: last.tokensPerSecond,
        promptTokens: last.promptTokens,
      });
    }

    let toolRawCall: string | undefined;
    let toolConfidence: number | null | undefined;
    let toolElapsedMs: number | undefined;
    if (options.toolParity) {
      const agent = new Needle(model, {
        tools: [
          {
            name: "turn_on_flashlight",
            description: "Turn on the flashlight",
            parameters: { type: "object", properties: {} },
          },
        ],
      });
      const toolStarted = performance.now();
      const response = await agent.complete("Turn on the flashlight.");
      toolElapsedMs = performance.now() - toolStarted;
      toolRawCall = response.rawCall;
      toolConfidence = response.confidence;
    }
    return {
      backend,
      status: "ok",
      loadMs,
      warmupMs,
      runs,
      ...(last ? { generatedText: last.text, finishReason: last.finishReason } : {}),
      ...(confidence === undefined ? {} : { confidence }),
      ...(toolRawCall === undefined ? {} : { toolRawCall }),
      ...(toolConfidence === undefined ? {} : { toolConfidence }),
      ...(toolElapsedMs === undefined ? {} : { toolElapsedMs }),
    };
  } catch (error) {
    return {
      backend,
      status: "error",
      error: error instanceof Error ? error.message : String(error),
      loadMs,
      runs: [],
    };
  } finally {
    await model.dispose();
  }
}

declare global {
  interface Window {
    runNeedleBench: (options: BrowserBenchOptions) => Promise<{
      webgpu: boolean;
      userAgent: string;
      results: BackendResult[];
    }>;
  }
}

window.runNeedleBench = async (options) => {
  const webgpu = Boolean(navigator.gpu);
  const results: BackendResult[] = [];
  for (const backend of options.backends) {
    console.log(`benchmarking ${backend}...`);
    results.push(await measureBackend(backend, options));
  }
  return { webgpu, userAgent: navigator.userAgent, results };
};
