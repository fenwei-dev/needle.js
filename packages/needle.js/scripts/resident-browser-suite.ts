import { BOS_TOKEN_ID, Needle, NeedleModel, type ResidentTokenSelection } from "../src/index.js";

interface SuiteResult {
  readonly userAgent: string;
  readonly webgpu: boolean;
  readonly greedyText: string;
  readonly promptConfidenceCpu: number;
  readonly promptConfidenceGpu: number;
  readonly toolCall: string;
  readonly toolConfidenceCpu: number | null;
  readonly toolConfidenceGpu: number | null;
  readonly longSelection: number;
  readonly elapsedMs: number;
}

function check(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

async function confidence(model: NeedleModel, prompt: string): Promise<number> {
  const ids = [BOS_TOKEN_ID, ...model.tokenizer.encode(prompt)];
  const runtime = model.createRuntime({ collectConfidence: true });
  runtime.reset({ maximumLength: ids.length + 1, sinkLength: Math.min(8, ids.length) });
  await runtime.prefill(ids);
  const value = await runtime.resolveConfidence();
  check(value !== undefined, "confidence head did not produce a value");
  return value;
}

async function selected(
  model: NeedleModel,
  ids: readonly number[],
  sinkLength: number,
  runtimeOptions: { kvCache?: "int8" | "float32" } = {},
): Promise<ResidentTokenSelection> {
  const runtime = model.createRuntime({ collectConfidence: false, ...runtimeOptions });
  runtime.reset({ maximumLength: ids.length + 1, sinkLength });
  return runtime.prefillSelected(ids);
}

async function flashlight(model: NeedleModel) {
  const agent = new Needle(model, {
    tools: [
      {
        name: "turn_on_flashlight",
        description: "Turn on the flashlight",
        parameters: { type: "object", properties: {} },
      },
    ],
  });
  return agent.complete("Turn on the flashlight.");
}

declare global {
  interface Window {
    runNeedleResidentSuite(modelUrl: string): Promise<SuiteResult>;
  }
}

window.runNeedleResidentSuite = async (modelUrl) => {
  const started = performance.now();
  check(Boolean(navigator.gpu), "navigator.gpu is unavailable");
  const weights = { kind: "url" as const, url: modelUrl, cache: false };
  const cpu = await NeedleModel.load({ weights, backend: "cpu" });
  const gpu = await NeedleModel.load({
    weights,
    backend: "typegpu",
    backendOptions: { minimumGpuRows: 0, residentLayers: true },
  });
  try {
    const prompt = "The most surprising thing about local inference is";
    const [cpuGreedy, gpuGreedy] = await Promise.all([
      cpu.generate(prompt, { maxNewTokens: 16, temperature: 0 }),
      gpu.generate(prompt, { maxNewTokens: 16, temperature: 0 }),
    ]);
    check(cpuGreedy.text === gpuGreedy.text, "resident greedy output differs from CPU");

    const confidencePrompt = "Turn on the flashlight.";
    const promptConfidenceCpu = await confidence(cpu, confidencePrompt);
    const promptConfidenceGpu = await confidence(gpu, confidencePrompt);
    check(
      Math.abs(promptConfidenceCpu - promptConfidenceGpu) < 1e-5,
      `resident prompt confidence differs: ${promptConfidenceCpu} vs ${promptConfidenceGpu}`,
    );

    const cpuTool = await flashlight(cpu);
    const gpuTool = await flashlight(gpu);
    const expected = '[{"name":"turn_on_flashlight","arguments":{}}]';
    check(cpuTool.rawCall === expected, "CPU flashlight fixture changed");
    check(gpuTool.rawCall === expected, "resident constrained tool call differs");
    check(
      cpuTool.confidence !== null &&
        gpuTool.confidence !== null &&
        Math.abs(cpuTool.confidence - gpuTool.confidence) < 0.02,
      "resident tool confidence is outside tolerance",
    );

    const repeatedToken = gpu.tokenizer.encode("x").at(-1);
    check(repeatedToken !== undefined, "failed to tokenize long-context fixture");
    const longIds = [BOS_TOKEN_ID, ...Array.from({ length: 270 }, () => repeatedToken)];
    const cpuLong = await selected(cpu, longIds, 8);
    const gpuLong = await selected(gpu, longIds, 8);
    check(cpuLong.id === gpuLong.id, "KV-window/prefix-sink selection differs");

    const firstReset = await selected(gpu, [BOS_TOKEN_ID, repeatedToken, repeatedToken], 1);
    const secondReset = await selected(gpu, [BOS_TOKEN_ID, repeatedToken, repeatedToken], 1);
    check(firstReset.id === secondReset.id, "resident reset is not deterministic");

    const cpuFloat = await selected(cpu, [BOS_TOKEN_ID, repeatedToken, repeatedToken], 0, {
      kvCache: "float32",
    });
    const gpuFloat = await selected(gpu, [BOS_TOKEN_ID, repeatedToken, repeatedToken], 0, {
      kvCache: "float32",
    });
    check(cpuFloat.id === gpuFloat.id, "float32-KV fallback selection differs");

    const aborted = new AbortController();
    aborted.abort("resident cancellation fixture");
    const abortRuntime = gpu.createRuntime({ collectConfidence: false });
    abortRuntime.reset({ maximumLength: 2 });
    let cancellationObserved = false;
    try {
      await abortRuntime.prefill([BOS_TOKEN_ID], aborted.signal);
    } catch {
      cancellationObserved = true;
    }
    check(cancellationObserved, "resident cancellation did not reject");

    return {
      userAgent: navigator.userAgent,
      webgpu: Boolean(navigator.gpu),
      greedyText: gpuGreedy.text,
      promptConfidenceCpu,
      promptConfidenceGpu,
      toolCall: gpuTool.rawCall,
      toolConfidenceCpu: cpuTool.confidence,
      toolConfidenceGpu: gpuTool.confidence,
      longSelection: gpuLong.id,
      elapsedMs: performance.now() - started,
    };
  } finally {
    await cpu.dispose();
    await gpu.dispose();
  }
};
