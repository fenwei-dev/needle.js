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
    <section class="demo-shell not-content">
      <header class="demo-toolbar">
        <div class="demo-tabs" role="tablist" aria-label="Demo API">
          {(Object.keys(labels) as DemoMode[]).map((item) => (
            <button
              key={item}
              type="button"
              role="tab"
              aria-selected={mode === item}
              class={`demo-tab${mode === item ? " is-active" : ""}`}
              onClick={() => setMode(item)}
            >
              {labels[item]}
            </button>
          ))}
        </div>
        <span class="demo-status">
          <span class={`demo-status-dot${modelRef.current ? " is-ready" : ""}`} />
          {status}
        </span>
      </header>

      <div class="demo-grid">
        <div class="demo-input-panel">
          <label class="demo-label" for="needle-demo-prompt">
            Prompt
          </label>
          <textarea
            id="needle-demo-prompt"
            value={prompt}
            onInput={(event) => setPrompt(event.currentTarget.value)}
            rows={4}
            class="demo-textarea"
          />
          <button
            type="button"
            disabled={running || !prompt.trim()}
            onClick={run}
            class="demo-run-button"
          >
            {running ? "Running locally…" : `Run with ${labels[mode]}`}
          </button>
          <p class="demo-note">
            The first run downloads the official 13.7 MB archive. Prompts, tool arguments, and
            results remain in this browser tab.
          </p>
        </div>

        <div class="demo-code-panel">
          <div class="demo-panel-header">Example</div>
          <pre class="demo-code">
            <code>{snippets[mode]}</code>
          </pre>
        </div>
      </div>

      <div class="demo-output-panel">
        <div class="demo-panel-header">Output</div>
        <pre class="demo-output">
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
