import { describe, expect, test } from "bun:test";
import {
  type AssistantMessageEvent,
  type Context,
  createModels,
  type Tool,
  Type,
} from "@earendil-works/pi-ai";
import type { NeedleModel, NeedleResponse } from "needle.js";
import { createNeedlePiProvider } from "../src/index.js";

function fakeModel(text = "local text"): NeedleModel {
  return {
    backend: { kind: "cpu" },
    generate: async () => ({
      text,
      tokenIds: [1, 2],
      tokenLogProbabilities: [-0.1, -0.2],
      finishReason: "eos",
      promptTokens: 10,
      generatedTokens: 2,
      elapsedMs: 20,
      tokensPerSecond: 100,
    }),
    dispose: async () => {},
  } as unknown as NeedleModel;
}

function response(overrides: Partial<NeedleResponse> = {}): NeedleResponse {
  const calls = overrides.functionCalls ?? [];
  return {
    type: calls.length > 0 ? "call" : "respond",
    success: true,
    error: null,
    errorCode: null,
    functionCalls: calls,
    function_calls: calls,
    reasoning: "selected the matching tool",
    confidence: 0.88,
    metrics: {
      promptTokens: 18,
      reasoningTokens: 3,
      callTokens: calls.length > 0 ? 5 : 0,
      prefillMs: 10,
      decodeMs: 5,
      prefillTokensPerSecond: 1_800,
      decodeTokensPerSecond: 1_000,
    },
    rawCall: calls.length > 0 ? JSON.stringify(calls) : "[]",
    selectedTools: ["weather"],
    ...overrides,
  };
}

const weatherTool: Tool = {
  name: "weather",
  description: "Get weather for a city",
  parameters: Type.Object({ city: Type.String() }),
};

const baseContext: Context = {
  systemPrompt: "You control local tools.",
  messages: [{ role: "user", content: "Weather in Paris?", timestamp: 1 }],
  tools: [weatherTool],
};

describe("needle-pi-ai-provider", () => {
  test("registers a keyless local model in a Models collection", async () => {
    const provider = createNeedlePiProvider({ model: fakeModel() });
    const models = createModels();
    models.setProvider(provider);
    expect(models.getModel("needle", "needle-2")).toMatchObject({
      api: "needle-local",
      contextWindow: 2_048,
      reasoning: true,
    });
    expect(await models.getAvailable("needle")).toHaveLength(1);
    await provider.dispose();
  });

  test("streams standard pi-ai thinking and tool-call events", async () => {
    const provider = createNeedlePiProvider({
      model: fakeModel(),
      _internal: {
        now: () => 123,
        createAgent: () => ({
          complete: async () =>
            response({
              functionCalls: [{ id: "call_1", name: "weather", arguments: { city: "Paris" } }],
            }),
        }),
      },
    });
    const model = provider.getModels()[0];
    if (!model) throw new Error("Missing Needle model");
    const events: AssistantMessageEvent[] = [];
    const stream = provider.streamSimple(model, baseContext);
    for await (const event of stream) events.push(event);
    const result = await stream.result();

    expect(events.map((event) => event.type)).toEqual([
      "start",
      "thinking_start",
      "thinking_delta",
      "thinking_end",
      "toolcall_start",
      "toolcall_delta",
      "toolcall_end",
      "done",
    ]);
    expect(result.stopReason).toBe("toolUse");
    expect(result.timestamp).toBe(123);
    expect(result.content).toContainEqual({
      type: "toolCall",
      id: "call_1",
      name: "weather",
      arguments: { city: "Paris" },
    });
  });

  test("works through Models.completeSimple and replays tool results", async () => {
    const inputs: string[] = [];
    const provider = createNeedlePiProvider({
      model: fakeModel(),
      _internal: {
        createAgent: () => {
          let step = 0;
          return {
            complete: async (text) => {
              inputs.push(text);
              step += 1;
              return step === 1
                ? response({
                    functionCalls: [
                      { id: "call_1", name: "weather", arguments: { city: "Paris" } },
                    ],
                  })
                : response();
            },
          };
        },
      },
    });
    const models = createModels();
    models.setProvider(provider);
    const model = models.getModel("needle", "needle-2");
    if (!model) throw new Error("Missing Needle model");
    const context: Context = {
      ...baseContext,
      messages: [
        ...baseContext.messages,
        {
          role: "assistant",
          content: [
            {
              type: "toolCall",
              id: "call_1",
              name: "weather",
              arguments: { city: "Paris" },
            },
          ],
          api: "needle-local",
          provider: "needle",
          model: "needle-2",
          usage: {
            input: 18,
            output: 8,
            cacheRead: 0,
            cacheWrite: 0,
            totalTokens: 26,
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
          },
          stopReason: "toolUse",
          timestamp: 2,
        },
        {
          role: "toolResult",
          toolCallId: "call_1",
          toolName: "weather",
          content: [{ type: "text", text: '{"sky":"sunny","temperature":24}' }],
          isError: false,
          timestamp: 3,
        },
      ],
    };
    const result = await models.completeSimple(model, context);
    expect(inputs).toEqual(["Weather in Paris?", '[{"sky":"sunny","temperature":24}]']);
    expect(result.stopReason).toBe("stop");
    expect(result.content).toContainEqual({
      type: "text",
      text: '{"sky":"sunny","temperature":24}',
    });
  });

  test("falls back to raw generation when no tools are supplied", async () => {
    const provider = createNeedlePiProvider({ model: fakeModel("raw local response") });
    const model = provider.getModels()[0];
    if (!model) throw new Error("Missing Needle model");
    const result = await provider
      .streamSimple(model, {
        messages: [{ role: "user", content: "Hello", timestamp: 1 }],
      })
      .result();
    expect(result.stopReason).toBe("stop");
    expect(result.content).toEqual([{ type: "text", text: "raw local response" }]);
  });

  test("turns provider failures into pi-ai error messages", async () => {
    const provider = createNeedlePiProvider({
      model: fakeModel(),
      _internal: {
        createAgent: () => ({
          complete: async () => {
            throw new Error("local failure");
          },
        }),
      },
    });
    const model = provider.getModels()[0];
    if (!model) throw new Error("Missing Needle model");
    const result = await provider.streamSimple(model, baseContext).result();
    expect(result.stopReason).toBe("error");
    expect(result.errorMessage).toBe("local failure");
  });
});
