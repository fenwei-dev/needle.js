import type { NeedleModel } from "needle.js";
import { useRef, useState } from "preact/hooks";

type DemoMode = "raw" | "ai-sdk" | "pi-agent";

const snippets: Record<DemoMode, string> = {
  raw: `const agent = new Needle(model, { tools: [turnOnFlashlight] });
const response = await agent.run("Turn on the flashlight.");`,
  "ai-sdk": `const result = await generateText({
  model: provider("needle-2"),
  prompt,
  tools: { turn_on_flashlight: tool({ ... }) },
  stopWhen: stepCountIs(3),
});`,
  "pi-agent": `const agent = new Agent({
  initialState: { model, tools: [flashlightTool] },
  streamFn: models.streamSimple.bind(models),
});
await agent.prompt(prompt);`,
};

const labels: Record<DemoMode, string> = {
  raw: "needle.js",
  "ai-sdk": "AI SDK",
  "pi-agent": "pi agent",
};

export default function NeedleDemo() {
  const [mode, setMode] = useState<DemoMode>("raw");
  const [prompt, setPrompt] = useState("Turn on the flashlight.");
  const [status, setStatus] = useState("Model not loaded");
  const [output, setOutput] = useState("Run an example to see its local result.");
  const [running, setRunning] = useState(false);
  const modelRef = useRef<NeedleModel>();

  async function loadModel(): Promise<NeedleModel> {
    if (modelRef.current) return modelRef.current;
    const { NeedleModel } = await import("needle.js");
    setStatus("Preparing the 13.7 MB model…");
    const model = await NeedleModel.load({
      weights: "download",
      backend: "cpu",
      onProgress: ({ loaded, total, cached }) => {
        const percentage = total ? ` ${Math.round((loaded / total) * 100)}%` : "";
        setStatus(cached ? "Loading cached model…" : `Downloading model…${percentage}`);
      },
    });
    modelRef.current = model;
    setStatus(`Ready · ${model.backend.name}`);
    return model;
  }

  async function run() {
    setRunning(true);
    setOutput("");
    const started = performance.now();
    try {
      const model = await loadModel();
      const result =
        mode === "raw"
          ? await runRaw(model, prompt)
          : mode === "ai-sdk"
            ? await runAiSdk(model, prompt)
            : await runPiAgent(model, prompt);
      setOutput(
        `${result}\n\nCompleted locally in ${((performance.now() - started) / 1_000).toFixed(2)}s`,
      );
    } catch (error) {
      setOutput(`Error: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setRunning(false);
    }
  }

  return (
    <section class="demo-shell not-content overflow-hidden rounded-2xl border border-purple-400/20 shadow-2xl shadow-purple-950/20">
      <div class="flex flex-wrap items-center justify-between gap-3 border-b border-purple-400/15 px-4 py-3">
        <div class="flex gap-1 rounded-xl bg-black/20 p-1" role="tablist" aria-label="Demo API">
          {(Object.keys(labels) as DemoMode[]).map((item) => (
            <button
              key={item}
              type="button"
              role="tab"
              aria-selected={mode === item}
              class={`rounded-lg px-3 py-2 text-sm font-semibold transition ${
                mode === item
                  ? "bg-purple-500 text-white shadow"
                  : "text-slate-300 hover:bg-white/5 hover:text-white"
              }`}
              onClick={() => setMode(item)}
            >
              {labels[item]}
            </button>
          ))}
        </div>
        <span class="text-xs text-slate-400">{status}</span>
      </div>

      <div class="grid gap-0 lg:grid-cols-[1.05fr_0.95fr]">
        <div class="space-y-4 p-5">
          <label class="block text-sm font-semibold text-slate-200" for="needle-demo-prompt">
            Prompt
          </label>
          <textarea
            id="needle-demo-prompt"
            value={prompt}
            onInput={(event) => setPrompt(event.currentTarget.value)}
            rows={4}
            class="w-full resize-y rounded-xl border border-purple-300/20 bg-black/25 p-3 font-mono text-sm text-slate-100 outline-none transition focus:border-purple-400 focus:ring-2 focus:ring-purple-500/20"
          />
          <button
            type="button"
            disabled={running || !prompt.trim()}
            onClick={run}
            class="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-amber-400 to-purple-500 px-5 py-2.5 font-bold text-slate-950 shadow-lg shadow-purple-950/20 transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {running ? "Running locally…" : `Run with ${labels[mode]}`}
          </button>
          <p class="text-xs leading-relaxed text-slate-400">
            The first run downloads the official archive from Hugging Face. Inference stays in this
            tab; prompts and tool data are not sent to a server.
          </p>
        </div>

        <div class="border-t border-purple-400/15 bg-black/20 lg:border-t-0 lg:border-l">
          <div class="border-b border-purple-400/15 px-4 py-2 text-xs font-bold tracking-widest text-purple-300 uppercase">
            Example
          </div>
          <pre class="min-h-36 overflow-x-auto p-4 text-xs leading-relaxed text-slate-300">
            <code>{snippets[mode]}</code>
          </pre>
        </div>
      </div>

      <div class="border-t border-purple-400/15">
        <div class="px-4 py-2 text-xs font-bold tracking-widest text-amber-300 uppercase">
          Output
        </div>
        <pre class="max-h-96 min-h-32 overflow-auto bg-[#0b0911]/70 p-4 text-xs leading-relaxed whitespace-pre-wrap text-emerald-200">
          <code>{output}</code>
        </pre>
      </div>
    </section>
  );
}

async function runRaw(model: NeedleModel, prompt: string): Promise<string> {
  const { Needle, defineTool } = await import("needle.js");
  const flashlight = defineTool<Record<string, never>, { flashlight: "on" }>({
    name: "turn_on_flashlight",
    description: "Turn on the device flashlight",
    parameters: { type: "object", properties: {} },
    execute: () => ({ flashlight: "on" }),
  });
  const agent = new Needle(model, { tools: [flashlight] });
  const response = await agent.run(prompt, { maxSteps: 2 });
  return JSON.stringify(
    {
      api: "needle.js",
      calls: response.functionCalls,
      results: response.results,
      confidence: response.confidence,
      metrics: response.metrics,
    },
    null,
    2,
  );
}

async function runAiSdk(model: NeedleModel, prompt: string): Promise<string> {
  const [{ generateText, jsonSchema, stepCountIs, tool }, { createNeedleProvider }] =
    await Promise.all([import("ai"), import("needle-ai-provider")]);
  const provider = createNeedleProvider({ model, disposeModel: false });
  const result = await generateText({
    model: provider(),
    prompt,
    stopWhen: stepCountIs(2),
    tools: {
      turn_on_flashlight: tool({
        description: "Turn on the device flashlight",
        inputSchema: jsonSchema<Record<string, never>>({
          type: "object",
          properties: {},
          additionalProperties: false,
        }),
        execute: async () => ({ flashlight: "on" }),
      }),
    },
  });
  await provider.dispose();
  return JSON.stringify(
    {
      api: "Vercel AI SDK",
      text: result.text,
      calls: result.toolCalls,
      results: result.toolResults,
      steps: result.steps.length,
    },
    null,
    2,
  );
}

async function runPiAgent(model: NeedleModel, prompt: string): Promise<string> {
  const [{ Agent }, piAi, { createNeedlePiProvider }] = await Promise.all([
    import("@earendil-works/pi-agent-core"),
    import("@earendil-works/pi-ai"),
    import("needle-pi-ai-provider"),
  ]);
  const provider = createNeedlePiProvider({ model, disposeModel: false });
  const models = piAi.createModels();
  models.setProvider(provider);
  const piModel = models.getModel("needle", "needle-2");
  if (!piModel) throw new Error("Needle pi model was not registered");

  const events: string[] = [];
  const agent = new Agent({
    initialState: {
      systemPrompt: "Use the available local tools and keep the response concise.",
      model: piModel,
      thinkingLevel: "off",
      tools: [
        {
          name: "turn_on_flashlight",
          label: "Turn on flashlight",
          description: "Turn on the device flashlight",
          parameters: piAi.Type.Object({}),
          execute: async () => ({
            content: [{ type: "text" as const, text: JSON.stringify({ flashlight: "on" }) }],
            details: {},
          }),
        },
      ],
    },
    streamFn: models.streamSimple.bind(models),
  });

  agent.subscribe((event) => {
    if (event.type === "tool_execution_start") events.push(`tool:start ${event.toolName}`);
    if (event.type === "tool_execution_end") events.push(`tool:end ${event.toolName}`);
    if (event.type === "message_update" && event.assistantMessageEvent.type === "thinking_delta") {
      events.push(`thinking: ${event.assistantMessageEvent.delta}`);
    }
  });
  await agent.prompt(prompt);
  await provider.dispose();

  return JSON.stringify(
    {
      api: "pi Agent",
      events,
      messages: agent.state.messages.map((message) => ({
        role: message.role,
        content: "content" in message ? message.content : undefined,
      })),
    },
    null,
    2,
  );
}
