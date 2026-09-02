import { BOS_TOKEN_ID, Needle, NeedleModel, type ResidentTokenSelection } from "../src/index.js";

interface CorpusCase {
  readonly id: string;
  readonly prompt: string;
}

interface CorpusEntry {
  readonly id: string;
  readonly prompt: string;
  readonly callsMatch: boolean;
  readonly cpuConfidence: number | null;
  readonly gpuConfidence: number | null;
  readonly confidenceDelta: number | null;
  readonly thresholdMismatches: readonly number[];
}

interface CorpusResult {
  readonly entries: readonly CorpusEntry[];
  readonly callMismatches: number;
  readonly thresholdMismatches: number;
  readonly maximumConfidenceDelta: number;
  readonly elapsedMs: number;
}

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
    runNeedleConfidenceCorpus(
      modelUrl: string,
      cases: readonly CorpusCase[],
    ): Promise<CorpusResult>;
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
    backendOptions: { execution: "resident" },
  });
  try {
    const prompt = "The most surprising thing about local inference is";
    const [cpuGreedy, gpuGreedy] = await Promise.all([
      cpu.generate(prompt, { maxNewTokens: 16, temperature: 0 }),
      gpu.generate(prompt, { maxNewTokens: 16, temperature: 0 }),
    ]);
    check(
      cpuGreedy.text === gpuGreedy.text,
      `resident greedy output differs from CPU: ${JSON.stringify(cpuGreedy.text)} vs ${JSON.stringify(gpuGreedy.text)}`,
    );

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

const corpusTools = [
  {
    name: "turn_on_flashlight",
    description: "Turn on the flashlight",
    parameters: { type: "object" as const, properties: {} },
  },
  {
    name: "turn_off_flashlight",
    description: "Turn off the flashlight",
    parameters: { type: "object" as const, properties: {} },
  },
  {
    name: "get_weather",
    description: "Get the current weather in a city",
    parameters: {
      type: "object" as const,
      properties: { city: { type: "string" as const } },
      required: ["city"],
    },
  },
  {
    name: "set_timer",
    description: "Start a timer for a number of seconds",
    parameters: {
      type: "object" as const,
      properties: { seconds: { type: "integer" as const, minimum: 1 } },
      required: ["seconds"],
    },
  },
  {
    name: "set_volume",
    description: "Set or mute the speaker volume",
    parameters: {
      type: "object" as const,
      properties: {
        percent: { type: "integer" as const, minimum: 0, maximum: 100 },
        muted: { type: "boolean" as const },
      },
    },
  },
  {
    name: "set_brightness",
    description: "Set screen brightness percentage",
    parameters: {
      type: "object" as const,
      properties: { percent: { type: "integer" as const, minimum: 0, maximum: 100 } },
      required: ["percent"],
    },
  },
  {
    name: "send_message",
    description: "Send a text message to a person",
    parameters: {
      type: "object" as const,
      properties: {
        recipient: { type: "string" as const },
        message: { type: "string" as const },
      },
      required: ["recipient", "message"],
    },
  },
  {
    name: "create_calendar_event",
    description: "Create an event in the calendar",
    parameters: {
      type: "object" as const,
      properties: {
        title: { type: "string" as const },
        date: { type: "string" as const },
        time: { type: "string" as const },
      },
      required: ["title", "date"],
    },
  },
  {
    name: "convert_units",
    description: "Convert a numeric value between units",
    parameters: {
      type: "object" as const,
      properties: {
        value: { type: "number" as const },
        from: { type: "string" as const },
        to: { type: "string" as const },
      },
      required: ["value", "from", "to"],
    },
  },
  {
    name: "set_light",
    description: "Control a room light and optional brightness",
    parameters: {
      type: "object" as const,
      properties: {
        room: { type: "string" as const },
        on: { type: "boolean" as const },
        brightness: { type: "integer" as const, minimum: 0, maximum: 100 },
      },
      required: ["room", "on"],
    },
  },
  {
    name: "create_reminder",
    description: "Create a reminder for a task and time",
    parameters: {
      type: "object" as const,
      properties: {
        text: { type: "string" as const },
        time: { type: "string" as const },
      },
      required: ["text", "time"],
    },
  },
] as const;

window.runNeedleConfidenceCorpus = async (modelUrl, cases) => {
  const started = performance.now();
  check(Boolean(navigator.gpu), "navigator.gpu is unavailable");
  const weights = { kind: "url" as const, url: modelUrl, cache: false };
  const cpu = await NeedleModel.load({ weights, backend: "cpu" });
  const gpu = await NeedleModel.load({
    weights,
    backend: "typegpu",
    backendOptions: { execution: "resident" },
  });
  const entries: CorpusEntry[] = [];
  try {
    for (const current of cases) {
      const cpuAgent = new Needle(cpu, { tools: corpusTools, maximumRetrievedTools: 5 });
      const gpuAgent = new Needle(gpu, { tools: corpusTools, maximumRetrievedTools: 5 });
      const cpuResult = await cpuAgent.complete(current.prompt, {
        maxNewTokens: 96,
        reasoningTokens: 32,
      });
      const gpuResult = await gpuAgent.complete(current.prompt, {
        maxNewTokens: 96,
        reasoningTokens: 32,
      });
      const cpuCalls = JSON.stringify(
        cpuResult.functionCalls.map((call) => ({ name: call.name, arguments: call.arguments })),
      );
      const gpuCalls = JSON.stringify(
        gpuResult.functionCalls.map((call) => ({ name: call.name, arguments: call.arguments })),
      );
      const confidenceDelta =
        cpuResult.confidence === null || gpuResult.confidence === null
          ? null
          : Math.abs(cpuResult.confidence - gpuResult.confidence);
      const thresholdMismatches = [0.7, 0.8, 0.9].filter(
        (threshold) =>
          (cpuResult.confidence !== null && cpuResult.confidence >= threshold) !==
          (gpuResult.confidence !== null && gpuResult.confidence >= threshold),
      );
      entries.push({
        id: current.id,
        prompt: current.prompt,
        callsMatch: cpuCalls === gpuCalls,
        cpuConfidence: cpuResult.confidence,
        gpuConfidence: gpuResult.confidence,
        confidenceDelta,
        thresholdMismatches,
      });
    }
  } finally {
    await cpu.dispose();
    await gpu.dispose();
  }
  return {
    entries,
    callMismatches: entries.filter((entry) => !entry.callsMatch).length,
    thresholdMismatches: entries.reduce(
      (total, entry) => total + entry.thresholdMismatches.length,
      0,
    ),
    maximumConfidenceDelta: Math.max(0, ...entries.map((entry) => entry.confidenceDelta ?? 0)),
    elapsedMs: performance.now() - started,
  };
};
