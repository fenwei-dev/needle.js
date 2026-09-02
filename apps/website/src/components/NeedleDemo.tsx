import type { NeedleModel } from "needle.js";
import { useEffect, useRef, useState } from "preact/hooks";
import CozyRoom3D, { type RoomState } from "./CozyRoom3D";

type DemoBackend = "resident" | "cpu";
type DemoMode = "raw" | "ai-sdk" | "pi-agent";
type DeviceName = "ceiling_light" | "curtain" | "fan";

interface ChatMessage {
  readonly id: number;
  readonly role: "assistant" | "user" | "error";
  readonly text: string;
}

interface RoomController {
  readonly state: RoomState;
  update(patch: Partial<RoomState>): void;
  action(message: string): void;
}

interface DemoResult {
  readonly reply: string;
  readonly raw: unknown;
}

const initialRoom: RoomState = {
  lightOn: true,
  curtainOpen: true,
  fanOn: true,
  fanSpeed: "low",
};

const suggestions = [
  "Turn off the ceiling light",
  "Close the curtains",
  "Start the fan at medium speed",
  "What's the status of the room?",
  "Turn everything off",
] as const;

const backendSnippets: Record<DemoBackend, string> = {
  resident: `const model = await NeedleModel.load({
  weights: "download",
  backend: "typegpu",
  backendOptions: { execution: "resident" },
});`,
  cpu: `const model = await NeedleModel.load({
  weights: "download",
  backend: "cpu",
});`,
};

const snippets: Record<DemoMode, string> = {
  raw: `const agent = new Needle(model, {
  tools: [getDeviceStatus, setCeilingLight, setCurtain, setFan],
});
const response = await agent.run(prompt, { maxSteps: 4 });`,
  "ai-sdk": `const result = await generateText({
  model: provider("needle-2"),
  prompt,
  tools: roomTools,
  stopWhen: stepCountIs(4),
});`,
  "pi-agent": `const agent = new Agent({
  initialState: { model, tools: roomTools },
  streamFn: models.streamSimple.bind(models),
});
await agent.prompt(prompt);`,
};

const modeLabels: Record<DemoMode, string> = {
  raw: "needle.js",
  "ai-sdk": "Vercel AI SDK",
  "pi-agent": "pi Agent",
};

const backendLabels: Record<DemoBackend, string> = {
  resident: "Resident WebGPU",
  cpu: "Pure TypeScript",
};

export default function NeedleDemo() {
  const [mode, setMode] = useState<DemoMode>("raw");
  const [backend, setBackend] = useState<DemoBackend>("resident");
  const [room, setRoom] = useState<RoomState>(initialRoom);
  const [prompt, setPrompt] = useState("");
  const [status, setStatus] = useState("Model not loaded");
  const [rawOutput, setRawOutput] = useState("Run an instruction to inspect the SDK response.");
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 1,
      role: "assistant",
      text: "Hi! I’m Momo, your room helper. Ask me to check or control the light, curtains, and fan.",
    },
  ]);
  const [running, setRunning] = useState(false);
  const modelRef = useRef<NeedleModel>();
  const loadedForRef = useRef<DemoBackend>();
  const roomRef = useRef(room);
  const chatMessagesRef = useRef<HTMLDivElement | null>(null);
  const messageIdRef = useRef(1);
  roomRef.current = room;

  useEffect(
    () => () => {
      void modelRef.current?.dispose();
    },
    [],
  );

  useEffect(() => {
    const messageList = chatMessagesRef.current;
    if (messageList) messageList.scrollTop = messageList.scrollHeight;
  }, [messages, running]);

  async function selectBackend(next: DemoBackend) {
    if (next === backend) return;
    const previous = modelRef.current;
    modelRef.current = undefined;
    loadedForRef.current = undefined;
    setBackend(next);
    setStatus("Model not loaded");
    if (previous) await previous.dispose();
  }

  async function loadModel(): Promise<NeedleModel> {
    if (modelRef.current && loadedForRef.current === backend) return modelRef.current;
    const { NeedleModel } = await import("needle.js");
    setStatus("Preparing 13.7 MB model…");
    const progress = ({
      loaded,
      total,
      cached,
    }: {
      loaded: number;
      total?: number;
      cached: boolean;
    }) => {
      const percentage = total ? ` ${Math.round((loaded / total) * 100)}%` : "";
      setStatus(cached ? "Loading cached model…" : `Downloading model…${percentage}`);
    };
    let model: NeedleModel;
    let fallback = false;
    if (backend === "resident") {
      try {
        model = await NeedleModel.load({
          weights: "download",
          backend: "typegpu",
          backendOptions: { execution: "resident" },
          onProgress: progress,
        });
      } catch (error) {
        fallback = true;
        setStatus("WebGPU unavailable · loading CPU fallback…");
        console.warn("needle.js resident WebGPU unavailable; using CPU", error);
        model = await NeedleModel.load({
          weights: "download",
          backend: "cpu",
          onProgress: progress,
        });
      }
    } else {
      model = await NeedleModel.load({ weights: "download", backend: "cpu", onProgress: progress });
    }
    modelRef.current = model;
    loadedForRef.current = backend;
    setStatus(
      `Ready · ${model.backend.name}${fallback ? " · fallback" : backend === "resident" ? " · resident" : ""}`,
    );
    return model;
  }

  function addMessage(role: ChatMessage["role"], text: string) {
    setMessages((current) => [...current, { id: ++messageIdRef.current, role, text }]);
  }

  async function run(instruction = prompt) {
    const text = instruction.trim();
    if (!text || running) return;
    addMessage("user", text);
    setPrompt("");
    setRunning(true);
    const actions: string[] = [];
    const controller: RoomController = {
      get state() {
        return roomRef.current;
      },
      update(patch) {
        const next = { ...roomRef.current, ...patch };
        roomRef.current = next;
        setRoom(next);
      },
      action(message) {
        actions.push(message);
      },
    };
    try {
      const model = await loadModel();
      const started = performance.now();
      const result =
        mode === "raw"
          ? await runRaw(model, text, controller)
          : mode === "ai-sdk"
            ? await runAiSdk(model, text, controller)
            : await runPiAgent(model, text, controller);
      const elapsedMs = performance.now() - started;
      addMessage(
        "assistant",
        actions.length > 0 ? actions.join(" ") : result.reply || "Done — the room is up to date.",
      );
      setRawOutput(
        JSON.stringify(
          {
            sdk: modeLabels[mode],
            backend: model.backend.name,
            elapsedMs: Math.round(elapsedMs),
            room: roomRef.current,
            response: result.raw,
          },
          null,
          2,
        ),
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      addMessage("error", `I couldn’t complete that: ${message}`);
      setRawOutput(`Error: ${message}`);
    } finally {
      setRunning(false);
    }
  }

  return (
    <section class="smart-room-demo not-content mx-auto w-full max-w-[76rem] text-[#352c3d]">
      <div class="smart-room-stage grid h-[30rem] grid-cols-[minmax(0,1.45fr)_minmax(20rem,0.72fr)] overflow-hidden rounded-[1.4rem] border-[3px] border-[#6e5775] bg-[#fff8f1] shadow-[0_1rem_0_#d7b7cb,0_1.6rem_3rem_rgb(72_46_75_/_0.2)] max-lg:h-auto max-lg:grid-cols-1 max-md:rounded-2xl max-md:border-2">
        <div
          class={`room-visual-card relative min-w-0 overflow-hidden bg-linear-to-br from-[#f7dff0] to-[#eecddd] transition-[filter] duration-500 max-lg:h-[34rem] max-md:h-[27rem] ${room.lightOn ? "is-lit" : "saturate-[0.86] brightness-[0.84]"}`}
        >
          <CozyRoom3D state={room} />
          <section
            class="room-device-strip absolute right-[0.6rem] bottom-[0.6rem] left-[0.6rem] z-[2] grid grid-cols-3 gap-[0.35rem]"
            aria-label="Current room device status"
          >
            <DeviceStatus
              icon="✦"
              label="Ceiling light"
              value={room.lightOn ? "On" : "Off"}
              active={room.lightOn}
            />
            <DeviceStatus
              icon="▥"
              label="Curtains"
              value={room.curtainOpen ? "Open" : "Closed"}
              active={room.curtainOpen}
            />
            <DeviceStatus
              icon="⌘"
              label="Fan"
              value={room.fanOn ? `${capitalize(room.fanSpeed)} · on` : "Off"}
              active={room.fanOn}
            />
          </section>
        </div>

        <aside class="room-chat-panel flex h-full min-h-0 min-w-0 flex-col overflow-hidden border-l-[3px] border-[#6e5775] bg-[radial-gradient(circle_at_90%_0%,rgb(248_194_211_/_0.4),transparent_35%),#fffaf3] max-lg:h-[34rem] max-lg:border-t-[3px] max-lg:border-l-0">
          <header class="room-chat-header flex shrink-0 items-center gap-2 border-b border-dashed border-[#dcc6d4] px-3 py-2.5">
            <div
              class="room-assistant-avatar grid size-8 place-items-center rounded-[0.65rem] border-[1.5px] border-[#765c7c] bg-[#d8c2ef] text-base text-white shadow-[0.12rem_0.12rem_0_#987fa0]"
              aria-hidden="true"
            >
              ☁
            </div>
            <div>
              <strong class="block text-[0.78rem] tracking-[-0.01em]">Momo Home</strong>
              <span class="mt-px block text-[0.56rem] leading-tight text-[#837487]">
                <i
                  class={`room-online-dot mr-1 inline-block size-[0.35rem] rounded-full ${modelRef.current ? "is-ready bg-[#6ab281]" : "bg-[#b2a8b5]"}`}
                />{" "}
                {status}
              </span>
            </div>
          </header>

          <div
            ref={chatMessagesRef}
            class="room-chat-messages flex min-h-0 flex-1 basis-0 flex-col gap-2 overflow-y-auto overscroll-contain p-4 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
            aria-live="polite"
          >
            {messages.map((message) => (
              <div
                key={message.id}
                class={`room-message is-${message.role} max-w-[88%] rounded-xl rounded-bl-[0.25rem] border-[1.5px] px-[0.65rem] py-2 text-[0.68rem] leading-[1.35] ${
                  message.role === "user"
                    ? "self-end rounded-br-[0.25rem] rounded-bl-xl border-[#8c7193] bg-[#8c7193] text-white shadow-[0_0.14rem_0_#604c66]"
                    : message.role === "error"
                      ? "border-[#e8a09c] bg-[#fff0ee] text-[#8d413d] shadow-[0_0.14rem_0_#eadce5]"
                      : "border-[#dfcbd7] bg-white shadow-[0_0.14rem_0_#eadce5]"
                }`}
              >
                {message.text}
              </div>
            ))}
            {running && (
              <div class="room-message is-assistant is-typing flex w-12 gap-1 rounded-xl rounded-bl-[0.25rem] border-[1.5px] border-[#dfcbd7] bg-white px-[0.65rem] py-2 shadow-[0_0.14rem_0_#eadce5]">
                <i class="size-[0.38rem] animate-[room-bounce_900ms_infinite_alternate] rounded-full bg-[#a58ba8]" />
                <i class="size-[0.38rem] animate-[room-bounce_900ms_150ms_infinite_alternate] rounded-full bg-[#a58ba8]" />
                <i class="size-[0.38rem] animate-[room-bounce_900ms_300ms_infinite_alternate] rounded-full bg-[#a58ba8]" />
              </div>
            )}
          </div>

          <section
            class="room-suggestions flex shrink-0 gap-[0.3rem] overflow-x-auto border-t border-[rgb(110_87_117_/_0.12)] bg-[rgb(232_213_225_/_0.24)] px-3 py-1.5 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
            aria-label="Suggested instructions"
          >
            {suggestions.map((suggestion) => (
              <button
                type="button"
                class="shrink-0 cursor-pointer rounded-full border border-[#cdb5c5] bg-[#f3e7ef] px-2 py-1 text-[0.56rem] leading-tight font-semibold text-[#69576e] hover:border-[#8c7193] hover:bg-[#ead9e5] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--sl-color-accent)] disabled:cursor-not-allowed disabled:opacity-50"
                disabled={running}
                onClick={() => setPrompt(suggestion)}
              >
                {suggestion}
              </button>
            ))}
          </section>

          <div class="shrink-0 border-t border-[rgb(110_87_117_/_0.12)]">
            <div class="room-composer m-2 grid grid-cols-[minmax(0,1fr)_auto] gap-2 rounded-2xl border-2 border-[#bba5b7] bg-white p-1 shadow-[inset_0_0.12rem_0.25rem_rgb(80_55_82_/_0.08)] transition-[border-color,box-shadow] focus-within:border-[#8c7193] focus-within:ring-2 focus-within:ring-[rgb(140_113_147_/_0.16)]">
              <textarea
                class="w-full resize-none border-0 bg-transparent p-[0.38rem] text-[0.75rem] leading-[1.45] text-[#3d3240] outline-none [scrollbar-width:none] focus:outline-none focus-visible:outline-none [&::-webkit-scrollbar]:hidden"
                id="needle-demo-prompt"
                value={prompt}
                onInput={(event) => setPrompt(event.currentTarget.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    void run();
                  }
                }}
                rows={1}
                placeholder="Ask Momo to control the room…"
                aria-label="Room instruction"
              />
              <button
                type="button"
                class="self-center cursor-pointer rounded-[0.72rem] border-2 border-[#684f70] bg-[#83658b] px-[0.72rem] py-[0.52rem] text-[0.7rem] font-extrabold text-white shadow-[0_0.1rem_0_#523d58] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--sl-color-accent)] disabled:cursor-not-allowed disabled:opacity-50"
                disabled={running || !prompt.trim()}
                onClick={() => void run()}
              >
                {running ? "…" : "Send"}
              </button>
            </div>
          </div>
        </aside>
      </div>

      <section
        class="demo-config-card mt-8 overflow-hidden rounded-xl border border-[var(--sl-color-gray-5)] bg-[var(--sl-color-bg-nav)] text-[var(--sl-color-white)] shadow-[0_0.6rem_1.6rem_rgb(0_0_0_/_0.1)]"
        aria-label="Demo configuration"
      >
        <div class="demo-config-heading flex items-center justify-between gap-4 border-b border-[var(--sl-color-gray-5)] p-4 max-md:flex-col max-md:items-start">
          <div>
            <span class="block text-[0.65rem] font-bold tracking-[0.08em] text-[var(--sl-color-gray-3)] uppercase">
              Playground configuration
            </span>
            <strong class="mt-[0.2rem] block text-[0.85rem]">
              Choose the runtime and integration surface
            </strong>
          </div>
          <code class="text-[0.68rem] text-[var(--sl-color-accent-high)]">
            {modelRef.current?.backend.name ?? "Not initialized"}
          </code>
        </div>
        <div class="demo-config-grid grid grid-cols-2 gap-4 p-4 max-md:grid-cols-1">
          <ConfigGroup label="Execution backend">
            {(Object.keys(backendLabels) as DemoBackend[]).map((item) => (
              <ConfigButton
                active={backend === item}
                disabled={running}
                onClick={() => void selectBackend(item)}
              >
                {backendLabels[item]}
              </ConfigButton>
            ))}
          </ConfigGroup>
          <ConfigGroup label="SDK interface">
            {(Object.keys(modeLabels) as DemoMode[]).map((item) => (
              <ConfigButton active={mode === item} disabled={running} onClick={() => setMode(item)}>
                {modeLabels[item]}
              </ConfigButton>
            ))}
          </ConfigGroup>
        </div>
      </section>

      <div class="demo-inspector-grid mt-4 grid grid-cols-2 gap-4 max-md:grid-cols-1">
        <section class="demo-inspector-panel min-w-0 overflow-hidden rounded-xl border border-[var(--sl-color-gray-5)] bg-[var(--sl-color-black)] text-[var(--sl-color-gray-1)]">
          <header class="flex items-center justify-between border-b border-[var(--sl-color-gray-5)] px-[0.8rem] py-[0.65rem] text-[0.68rem] font-bold text-[var(--sl-color-gray-2)]">
            <span>Sample code</span>
            <code class="text-[0.62rem] text-[var(--sl-color-gray-3)]">{modeLabels[mode]}</code>
          </header>
          <pre class="m-0 min-h-72 max-h-[28rem] overflow-auto p-4 font-mono text-[0.7rem] leading-[1.65]">
            <code>{`${backendSnippets[backend]}\n\n${snippets[mode]}`}</code>
          </pre>
        </section>
        <section class="demo-inspector-panel min-w-0 overflow-hidden rounded-xl border border-[var(--sl-color-gray-5)] bg-[var(--sl-color-black)] text-[var(--sl-color-gray-1)]">
          <header class="flex items-center justify-between border-b border-[var(--sl-color-gray-5)] px-[0.8rem] py-[0.65rem] text-[0.68rem] font-bold text-[var(--sl-color-gray-2)]">
            <span>Raw output</span>
            <code class="text-[0.62rem] text-[var(--sl-color-gray-3)]">JSON</code>
          </header>
          <pre class="demo-raw-output m-0 min-h-72 max-h-[28rem] overflow-auto whitespace-pre-wrap p-4 font-mono text-[0.7rem] leading-[1.65]">
            <code>{rawOutput}</code>
          </pre>
        </section>
      </div>
    </section>
  );
}

function DeviceStatus({
  icon,
  label,
  value,
  active,
}: {
  icon: string;
  label: string;
  value: string;
  active: boolean;
}) {
  return (
    <div class="room-device-status flex min-w-0 select-none items-center gap-[0.35rem] rounded-[0.65rem] border-[1.5px] border-[rgb(83_62_91_/_0.15)] bg-[rgb(255_250_246_/_0.94)] px-[0.4rem] py-[0.38rem] shadow-[0_0.2rem_0.55rem_rgb(72_46_75_/_0.1)] backdrop-blur-xl">
      <span
        class={`grid size-[1.45rem] shrink-0 place-items-center rounded-[0.48rem] text-[0.72rem] font-black ${active ? "bg-[#ffe098] text-[#8b5c18]" : "bg-[#eadde8] text-[#705778]"}`}
      >
        {icon}
      </span>
      <div class="min-w-0">
        <strong class="block overflow-hidden text-[0.6rem] text-ellipsis whitespace-nowrap">
          {label}
        </strong>
        <small class="mt-[0.04rem] block overflow-hidden text-[0.55rem] text-ellipsis whitespace-nowrap text-[#7d7082]">
          {value}
        </small>
      </div>
    </div>
  );
}

function ConfigGroup({ label, children }: { label: string; children: preact.ComponentChildren }) {
  return (
    <fieldset class="m-0 border-0 p-0">
      <legend class="mb-2 text-[0.68rem] font-bold text-[var(--sl-color-gray-2)]">{label}</legend>
      <div class="flex flex-wrap gap-[0.4rem]">{children}</div>
    </fieldset>
  );
}

function ConfigButton({
  active,
  disabled,
  onClick,
  children,
}: {
  active: boolean;
  disabled: boolean;
  onClick: () => void;
  children: preact.ComponentChildren;
}) {
  return (
    <button
      type="button"
      aria-pressed={active}
      disabled={disabled}
      class={`cursor-pointer rounded-md border px-[0.65rem] py-[0.48rem] text-[0.68rem] font-semibold focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--sl-color-accent)] disabled:cursor-not-allowed disabled:opacity-50 ${
        active
          ? "is-active border-[var(--sl-color-accent)] bg-[var(--sl-color-accent-low)] text-[var(--sl-color-accent-high)]"
          : "border-[var(--sl-color-gray-5)] bg-[var(--sl-color-black)] text-[var(--sl-color-gray-2)]"
      }`}
      onClick={onClick}
    >
      {children}
    </button>
  );
}

function capitalize(value: string) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function describeStatus(state: RoomState) {
  return `The ceiling light is ${state.lightOn ? "on" : "off"}, the curtains are ${state.curtainOpen ? "open" : "closed"}, and the fan is ${state.fanOn ? `on at ${state.fanSpeed} speed` : "off"}.`;
}

function deviceStatus(state: RoomState, device: DeviceName) {
  if (device === "ceiling_light") return { device, on: state.lightOn };
  if (device === "curtain") return { device, open: state.curtainOpen };
  return { device, on: state.fanOn, speed: state.fanSpeed };
}

async function runRaw(
  model: NeedleModel,
  prompt: string,
  room: RoomController,
): Promise<DemoResult> {
  const { Needle, defineTool } = await import("needle.js");
  const getStatus = defineTool<{ device?: DeviceName }, unknown>({
    name: "get_device_status",
    description:
      "Get current status of the ceiling light, electric curtain, fan, or all room devices",
    parameters: {
      type: "object",
      properties: {
        device: { type: "string", enum: ["ceiling_light", "curtain", "fan"] },
      },
    },
    execute: ({ device }) => {
      const result = device ? deviceStatus(room.state, device) : room.state;
      room.action(
        device ? `${capitalize(device.replace("_", " "))} checked.` : describeStatus(room.state),
      );
      return result;
    },
  });
  const setLight = defineTool<{ on: boolean }, { on: boolean }>({
    name: "set_ceiling_light",
    description: "Turn the bedroom ceiling light on or off",
    parameters: {
      type: "object",
      properties: { on: { type: "boolean" } },
      required: ["on"],
    },
    execute: ({ on }) => {
      room.update({ lightOn: on });
      room.action(`Ceiling light turned ${on ? "on" : "off"}.`);
      return { on };
    },
  });
  const setCurtain = defineTool<{ open: boolean }, { open: boolean }>({
    name: "set_curtain",
    description: "Open or close the electric bedroom curtains",
    parameters: {
      type: "object",
      properties: { open: { type: "boolean" } },
      required: ["open"],
    },
    execute: ({ open }) => {
      room.update({ curtainOpen: open });
      room.action(`Curtains ${open ? "opened" : "closed"}.`);
      return { open };
    },
  });
  const setFan = defineTool<
    { on: boolean; speed?: RoomState["fanSpeed"] },
    { on: boolean; speed: RoomState["fanSpeed"] }
  >({
    name: "set_fan",
    description: "Turn the electric fan on or off and optionally set low, medium, or high speed",
    parameters: {
      type: "object",
      properties: {
        on: { type: "boolean" },
        speed: { type: "string", enum: ["low", "medium", "high"] },
      },
      required: ["on"],
    },
    execute: ({ on, speed = room.state.fanSpeed }) => {
      room.update({ fanOn: on, fanSpeed: speed });
      room.action(`Fan turned ${on ? `on at ${speed} speed` : "off"}.`);
      return { on, speed };
    },
  });
  const agent = new Needle(model, { tools: [getStatus, setLight, setCurtain, setFan] });
  const response = await agent.run(prompt, { maxSteps: 4 });
  return {
    reply: response.reasoning,
    raw: {
      calls: response.functionCalls,
      results: response.results,
      confidence: response.confidence,
      metrics: response.metrics,
    },
  };
}

async function runAiSdk(
  model: NeedleModel,
  prompt: string,
  room: RoomController,
): Promise<DemoResult> {
  const [{ generateText, jsonSchema, stepCountIs, tool }, { createNeedleProvider }] =
    await Promise.all([import("ai"), import("needle-ai-provider")]);
  const provider = createNeedleProvider({ model, disposeModel: false });
  const result = await generateText({
    model: provider(),
    prompt,
    stopWhen: stepCountIs(4),
    tools: {
      get_device_status: tool({
        description: "Get the status of one smart-bedroom device or the whole room",
        inputSchema: jsonSchema<{ device?: DeviceName }>({
          type: "object",
          properties: { device: { type: "string", enum: ["ceiling_light", "curtain", "fan"] } },
          additionalProperties: false,
        }),
        execute: async ({ device }) => {
          room.action(
            device
              ? `${capitalize(device.replace("_", " "))} checked.`
              : describeStatus(room.state),
          );
          return device ? deviceStatus(room.state, device) : room.state;
        },
      }),
      set_ceiling_light: tool({
        description: "Turn the bedroom ceiling light on or off",
        inputSchema: jsonSchema<{ on: boolean }>({
          type: "object",
          properties: { on: { type: "boolean" } },
          required: ["on"],
          additionalProperties: false,
        }),
        execute: async ({ on }) => {
          room.update({ lightOn: on });
          room.action(`Ceiling light turned ${on ? "on" : "off"}.`);
          return { on };
        },
      }),
      set_curtain: tool({
        description: "Open or close the electric bedroom curtains",
        inputSchema: jsonSchema<{ open: boolean }>({
          type: "object",
          properties: { open: { type: "boolean" } },
          required: ["open"],
          additionalProperties: false,
        }),
        execute: async ({ open }) => {
          room.update({ curtainOpen: open });
          room.action(`Curtains ${open ? "opened" : "closed"}.`);
          return { open };
        },
      }),
      set_fan: tool({
        description: "Turn the electric fan on or off and set its speed",
        inputSchema: jsonSchema<{ on: boolean; speed?: RoomState["fanSpeed"] }>({
          type: "object",
          properties: {
            on: { type: "boolean" },
            speed: { type: "string", enum: ["low", "medium", "high"] },
          },
          required: ["on"],
          additionalProperties: false,
        }),
        execute: async ({ on, speed = room.state.fanSpeed }) => {
          room.update({ fanOn: on, fanSpeed: speed });
          room.action(`Fan turned ${on ? `on at ${speed} speed` : "off"}.`);
          return { on, speed };
        },
      }),
    },
  });
  await provider.dispose();
  return {
    reply: result.text,
    raw: {
      text: result.text,
      calls: result.toolCalls,
      results: result.toolResults,
      steps: result.steps.length,
    },
  };
}

async function runPiAgent(
  model: NeedleModel,
  prompt: string,
  room: RoomController,
): Promise<DemoResult> {
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
  const textResult = (value: unknown) => ({
    content: [{ type: "text" as const, text: JSON.stringify(value) }],
    details: {},
  });
  const agent = new Agent({
    initialState: {
      systemPrompt: "Control the cozy bedroom with the available local tools.",
      model: piModel,
      thinkingLevel: "off",
      tools: [
        {
          name: "get_device_status",
          label: "Check room device",
          description: "Get the current status of all smart-bedroom devices",
          parameters: piAi.Type.Object({}),
          execute: async () => {
            room.action(describeStatus(room.state));
            return textResult(room.state);
          },
        },
        {
          name: "set_ceiling_light",
          label: "Set ceiling light",
          description: "Turn the bedroom ceiling light on or off",
          parameters: piAi.Type.Object({ on: piAi.Type.Boolean() }),
          execute: async (_id, params: unknown) => {
            const { on } = params as { on: boolean };
            room.update({ lightOn: on });
            room.action(`Ceiling light turned ${on ? "on" : "off"}.`);
            return textResult({ on });
          },
        },
        {
          name: "set_curtain",
          label: "Set electric curtain",
          description: "Open or close the electric bedroom curtains",
          parameters: piAi.Type.Object({ open: piAi.Type.Boolean() }),
          execute: async (_id, params: unknown) => {
            const { open } = params as { open: boolean };
            room.update({ curtainOpen: open });
            room.action(`Curtains ${open ? "opened" : "closed"}.`);
            return textResult({ open });
          },
        },
        {
          name: "set_fan",
          label: "Set electric fan",
          description: "Turn the fan on or off and set low, medium, or high speed",
          parameters: piAi.Type.Object({
            on: piAi.Type.Boolean(),
            speed: piAi.Type.Optional(
              piAi.Type.Union([
                piAi.Type.Literal("low"),
                piAi.Type.Literal("medium"),
                piAi.Type.Literal("high"),
              ]),
            ),
          }),
          execute: async (_id, params: unknown) => {
            const args = params as { on: boolean; speed?: RoomState["fanSpeed"] };
            const speed = args.speed ?? room.state.fanSpeed;
            room.update({ fanOn: args.on, fanSpeed: speed });
            room.action(`Fan turned ${args.on ? `on at ${speed} speed` : "off"}.`);
            return textResult({ on: args.on, speed });
          },
        },
      ],
    },
    streamFn: models.streamSimple.bind(models),
  });

  agent.subscribe((event) => {
    if (event.type === "tool_execution_start") events.push(`tool:start ${event.toolName}`);
    if (event.type === "tool_execution_end") events.push(`tool:end ${event.toolName}`);
  });
  await agent.prompt(prompt);
  await provider.dispose();
  const lastMessage = agent.state.messages.at(-1);
  return {
    reply: lastMessage && "content" in lastMessage ? JSON.stringify(lastMessage.content) : "",
    raw: {
      events,
      messages: agent.state.messages.map((message) => ({
        role: message.role,
        content: "content" in message ? message.content : undefined,
      })),
    },
  };
}
