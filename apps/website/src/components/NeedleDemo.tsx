import { useEffect, useRef, useState } from "preact/hooks";
import CozyRoom3D from "./CozyRoom3D";
import type {
  DemoBackend,
  DemoMode,
  RoomState,
  WorkerEvent,
  WorkerResultEvent,
} from "./needle-demo-protocol";

interface ChatMessage {
  readonly id: number;
  readonly role: "assistant" | "user" | "error";
  readonly text: string;
}

interface PendingRun {
  resolve(event: WorkerResultEvent): void;
  reject(error: Error): void;
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
  const [actualBackend, setActualBackend] = useState<string>();
  const workerRef = useRef<Worker>();
  const pendingRunsRef = useRef(new Map<number, PendingRun>());
  const chatMessagesRef = useRef<HTMLDivElement | null>(null);
  const messageIdRef = useRef(1);
  const requestIdRef = useRef(0);

  useEffect(() => {
    const worker = new Worker(new URL("./needle-demo.worker.ts", import.meta.url), {
      type: "module",
    });
    workerRef.current = worker;
    worker.onmessage = ({ data }: MessageEvent<WorkerEvent>) => {
      if (data.type === "status") {
        setStatus(data.status);
        if (data.backend) setActualBackend(data.backend);
        return;
      }
      const pending = pendingRunsRef.current.get(data.id);
      if (!pending) return;
      pendingRunsRef.current.delete(data.id);
      if (data.type === "result") pending.resolve(data);
      else pending.reject(new Error(data.message));
    };
    worker.onerror = (event) => {
      const error = new Error(event.message || "Inference worker failed");
      for (const pending of pendingRunsRef.current.values()) pending.reject(error);
      pendingRunsRef.current.clear();
    };
    return () => {
      worker.postMessage({ type: "dispose" });
      worker.terminate();
      for (const pending of pendingRunsRef.current.values())
        pending.reject(new Error("Inference worker was disposed"));
      pendingRunsRef.current.clear();
      workerRef.current = undefined;
    };
  }, []);

  useEffect(() => {
    const messageList = chatMessagesRef.current;
    if (messageList) messageList.scrollTop = messageList.scrollHeight;
  }, [messages, running]);

  function selectBackend(next: DemoBackend) {
    if (next === backend) return;
    workerRef.current?.postMessage({ type: "dispose" });
    setBackend(next);
    setActualBackend(undefined);
    setStatus("Model not loaded");
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
    try {
      const worker = workerRef.current;
      if (!worker) throw new Error("Inference worker is not ready");
      const id = ++requestIdRef.current;
      const result = await new Promise<WorkerResultEvent>((resolve, reject) => {
        pendingRunsRef.current.set(id, { resolve, reject });
        worker.postMessage({ type: "run", id, mode, backend, prompt: text, room });
      });
      setRoom(result.room);
      setActualBackend(result.backend);
      addMessage(
        "assistant",
        result.actions.length > 0
          ? result.actions.join(" ")
          : result.reply || "Done — the room is up to date.",
      );
      setRawOutput(
        JSON.stringify(
          {
            sdk: modeLabels[mode],
            backend: result.backend,
            elapsedMs: Math.round(result.elapsedMs),
            room: result.room,
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
                  class={`room-online-dot mr-1 inline-block size-[0.35rem] rounded-full ${actualBackend ? "is-ready bg-[#6ab281]" : "bg-[#b2a8b5]"}`}
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
            {actualBackend ?? "Not initialized"}
          </code>
        </div>
        <div class="demo-config-grid grid grid-cols-2 gap-4 p-4 max-md:grid-cols-1">
          <ConfigGroup label="Execution backend">
            {(Object.keys(backendLabels) as DemoBackend[]).map((item) => (
              <ConfigButton
                active={backend === item}
                disabled={running}
                onClick={() => selectBackend(item)}
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
