import type { NeedleModel } from "needle.js";
import type {
  DemoBackend,
  RoomState,
  WorkerEvent,
  WorkerRequest,
  WorkerRunRequest,
} from "./needle-demo-protocol";

interface RoomController {
  readonly state: RoomState;
  update(patch: Partial<RoomState>): void;
  action(message: string): void;
}

interface DemoResult {
  readonly reply: string;
  readonly raw: unknown;
}

const workerScope = globalThis as unknown as {
  postMessage(event: WorkerEvent): void;
  addEventListener(type: "message", listener: (event: MessageEvent<WorkerRequest>) => void): void;
};

let model: NeedleModel | undefined;
let loadedFor: DemoBackend | undefined;

workerScope.addEventListener("message", (event) => {
  if (event.data.type === "dispose") {
    const previous = model;
    model = undefined;
    loadedFor = undefined;
    void previous?.dispose();
    return;
  }
  void runRequest(event.data);
});

async function runRequest(request: WorkerRunRequest) {
  const actions: string[] = [];
  let roomState = { ...request.room };
  const room: RoomController = {
    get state() {
      return roomState;
    },
    update(patch) {
      roomState = { ...roomState, ...patch };
    },
    action(message) {
      actions.push(message);
    },
  };
  try {
    const activeModel = await loadModel(request);
    const started = performance.now();
    const result =
      request.mode === "raw"
        ? await runRaw(activeModel, request.prompt, room)
        : request.mode === "ai-sdk"
          ? await runAiSdk(activeModel, request.prompt, room)
          : await runPiAgent(activeModel, request.prompt, room);
    workerScope.postMessage({
      type: "result",
      id: request.id,
      reply: result.reply,
      raw: result.raw,
      room: roomState,
      actions,
      backend: activeModel.backend.name,
      elapsedMs: performance.now() - started,
    });
  } catch (error) {
    workerScope.postMessage({
      type: "error",
      id: request.id,
      message: error instanceof Error ? error.message : String(error),
    });
  }
}

async function loadModel(request: WorkerRunRequest): Promise<NeedleModel> {
  if (model && loadedFor === request.backend) return model;
  const previous = model;
  model = undefined;
  loadedFor = undefined;
  if (previous) await previous.dispose();
  const { NeedleModel } = await import("needle.js");
  postStatus(request.id, "Preparing 13.7 MB model…");
  const onProgress = ({
    loaded,
    total,
    cached,
  }: {
    loaded: number;
    total?: number;
    cached: boolean;
  }) => {
    const percentage = total ? ` ${Math.round((loaded / total) * 100)}%` : "";
    postStatus(request.id, cached ? "Loading cached model…" : `Downloading model…${percentage}`);
  };
  let fallback = false;
  if (request.backend === "resident") {
    try {
      model = await NeedleModel.load({
        weights: "download",
        backend: "typegpu",
        backendOptions: { execution: "resident" },
        onProgress,
      });
    } catch (error) {
      fallback = true;
      postStatus(request.id, "WebGPU unavailable · loading CPU fallback…");
      console.warn("needle.js resident WebGPU unavailable in worker; using CPU", error);
      model = await NeedleModel.load({ weights: "download", backend: "cpu", onProgress });
    }
  } else {
    model = await NeedleModel.load({ weights: "download", backend: "cpu", onProgress });
  }
  loadedFor = request.backend;
  postStatus(
    request.id,
    `Ready · ${model.backend.name}${fallback ? " · fallback" : request.backend === "resident" ? " · resident" : ""}`,
    model.backend.name,
  );
  return model;
}

function postStatus(id: number, status: string, backend?: string) {
  workerScope.postMessage({ type: "status", id, status, ...(backend ? { backend } : {}) });
}

function describeStatus(state: RoomState) {
  return `The ceiling light is ${state.lightOn ? "on" : "off"}, the curtains are ${state.curtainOpen ? "open" : "closed"}, and the fan is ${state.fanOn ? `on at ${state.fanSpeed} speed` : "off"}.`;
}

function retrievedToolLimit(prompt: string) {
  const normalized = prompt.toLowerCase();
  if (/\b(all|everything)\b/.test(normalized)) return 4;
  const mentionedDevices = ["light", "curtain", "fan"].filter((device) =>
    normalized.includes(device),
  ).length;
  return mentionedDevices || (/\b(status|state|room)\b/.test(normalized) ? 1 : 2);
}

async function runRaw(
  model: NeedleModel,
  prompt: string,
  room: RoomController,
): Promise<DemoResult> {
  const { Needle, defineTool } = await import("needle.js");
  const getStatus = defineTool<Record<string, never>, RoomState>({
    name: "get_device_status",
    description: "Read all bedroom device states",
    parameters: { type: "object", properties: {} },
    execute: () => {
      room.action(describeStatus(room.state));
      return room.state;
    },
  });
  const setLight = defineTool<{ on: boolean }, { on: boolean }>({
    name: "set_ceiling_light",
    description: "Switch the ceiling light on or off",
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
    description: "Open or close the curtains",
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
    description: "Switch the fan on or off and choose its speed",
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
  const agent = new Needle(model, {
    tools: [getStatus, setLight, setCurtain, setFan],
    maximumRetrievedTools: retrievedToolLimit(prompt),
  });
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
  const provider = createNeedleProvider({
    model,
    disposeModel: false,
    agentOptions: { maximumRetrievedTools: retrievedToolLimit(prompt) },
  });
  const result = await generateText({
    model: provider(),
    prompt,
    stopWhen: stepCountIs(4),
    tools: {
      get_device_status: tool({
        description: "Read all bedroom device states",
        inputSchema: jsonSchema<Record<string, never>>({
          type: "object",
          properties: {},
          additionalProperties: false,
        }),
        execute: async () => {
          room.action(describeStatus(room.state));
          return room.state;
        },
      }),
      set_ceiling_light: tool({
        description: "Switch the ceiling light on or off",
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
        description: "Open or close the curtains",
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
        description: "Switch the fan on or off and choose its speed",
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
  const provider = createNeedlePiProvider({
    model,
    disposeModel: false,
    agentOptions: { maximumRetrievedTools: retrievedToolLimit(prompt) },
  });
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
          description: "Switch the ceiling light on or off",
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
          description: "Open or close the curtains",
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
          description: "Switch the fan on or off and choose its speed",
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
