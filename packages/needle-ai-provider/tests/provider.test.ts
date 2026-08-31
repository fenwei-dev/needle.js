import { describe, expect, test } from "bun:test";
import type { LanguageModelV4CallOptions } from "@ai-sdk/provider";
import { generateText, jsonSchema, stepCountIs, streamText, tool } from "ai";
import type { NeedleModel, NeedleResponse } from "needle.js";
import { createNeedleProvider } from "../src/index.js";

function fakeModel(text = "local output"): NeedleModel {
  return {
    backend: { kind: "cpu" },
    generate: async () => ({
      text,
      tokenIds: [1, 2],
      tokenLogProbabilities: [-0.1, -0.2],
      finishReason: "eos",
      promptTokens: 12,
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
    reasoning: "matched the request",
    confidence: 0.9,
    metrics: {
      promptTokens: 20,
      reasoningTokens: 3,
      callTokens: calls.length > 0 ? 5 : 0,
      prefillMs: 10,
      decodeMs: 5,
      prefillTokensPerSecond: 2_000,
      decodeTokensPerSecond: 1_000,
    },
    rawCall: calls.length > 0 ? JSON.stringify(calls) : "[]",
    selectedTools: ["weather"],
    ...overrides,
  };
}

const userPrompt: LanguageModelV4CallOptions["prompt"] = [
  { role: "user", content: [{ type: "text", text: "Weather in Paris?" }] },
];

const weatherTool = {
  type: "function" as const,
  name: "weather",
  description: "Get weather",
  inputSchema: {
    type: "object" as const,
    properties: { city: { type: "string" as const } },
    required: ["city"],
  },
};

describe("needle-ai-provider", () => {
  test("works with AI SDK generateText in raw mode", async () => {
    const provider = createNeedleProvider({
      model: fakeModel("hello from Needle"),
      _internal: {
        now: () => new Date("2026-01-01T00:00:00Z"),
        generateId: () => "response-1",
      },
    });
    const result = await generateText({
      model: provider("needle-2"),
      prompt: "Hello",
    });
    expect(result.text).toBe("hello from Needle");
    expect(result.usage).toMatchObject({ inputTokens: 12, outputTokens: 2 });
    await provider.dispose();
  });

  test("maps Needle calls to AI SDK tool-call content", async () => {
    const inputs: string[] = [];
    const provider = createNeedleProvider({
      model: fakeModel(),
      _internal: {
        createAgent: () => ({
          complete: async (text) => {
            inputs.push(text);
            return response({
              functionCalls: [{ id: "call_1", name: "weather", arguments: { city: "Paris" } }],
            });
          },
        }),
        generateId: () => "response-2",
      },
    });
    const result = await provider("needle-2").doGenerate({
      prompt: userPrompt,
      tools: [weatherTool],
    });
    expect(inputs).toEqual(["Weather in Paris?"]);
    expect(result.finishReason.unified).toBe("tool-calls");
    expect(result.content).toContainEqual({
      type: "tool-call",
      toolCallId: "call_1",
      toolName: "weather",
      input: '{"city":"Paris"}',
    });
  });

  test("replays tool results and exposes the result as final text", async () => {
    const inputs: string[] = [];
    const responses = [
      response({
        functionCalls: [{ id: "call_1", name: "weather", arguments: { city: "Paris" } }],
      }),
      response(),
    ];
    const provider = createNeedleProvider({
      model: fakeModel(),
      _internal: {
        createAgent: () => ({
          complete: async (text) => {
            inputs.push(text);
            const next = responses.shift();
            if (!next) throw new Error("Unexpected replay step");
            return next;
          },
        }),
      },
    });
    const result = await provider("needle-2").doGenerate({
      tools: [weatherTool],
      prompt: [
        ...userPrompt,
        {
          role: "assistant",
          content: [
            {
              type: "tool-call",
              toolCallId: "call_1",
              toolName: "weather",
              input: { city: "Paris" },
            },
          ],
        },
        {
          role: "tool",
          content: [
            {
              type: "tool-result",
              toolCallId: "call_1",
              toolName: "weather",
              output: { type: "text", value: "sunny, 24 C" },
            },
          ],
        },
      ],
    });
    expect(inputs).toEqual(["Weather in Paris?", '["sunny, 24 C"]']);
    expect(result.content).toContainEqual({ type: "text", text: "sunny, 24 C" });
  });

  test("executes an AI SDK tool loop end to end", async () => {
    const provider = createNeedleProvider({
      model: fakeModel(),
      _internal: {
        createAgent: () => {
          let step = 0;
          return {
            complete: async () => {
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
    const result = await generateText({
      model: provider(),
      prompt: "Weather in Paris?",
      stopWhen: stepCountIs(2),
      tools: {
        weather: tool({
          description: "Get weather",
          inputSchema: jsonSchema<{ city: string }>({
            type: "object",
            properties: { city: { type: "string" } },
            required: ["city"],
          }),
          execute: async ({ city }) => `${city}: sunny, 24 C`,
        }),
      },
    });
    expect(result.steps).toHaveLength(2);
    expect(result.toolResults[0]?.output).toBe("Paris: sunny, 24 C");
    expect(result.text).toBe("Paris: sunny, 24 C");
  });

  test("supports JSON response schemas through a synthetic extraction tool", async () => {
    const provider = createNeedleProvider({
      model: fakeModel(),
      _internal: {
        createAgent: (_model, options) => ({
          complete: async () =>
            response({
              functionCalls: [
                {
                  id: "call_1",
                  name:
                    options.tools?.[0] && "name" in options.tools[0]
                      ? options.tools[0].name
                      : "json_response",
                  arguments: { city: "Paris", temperature: 24 },
                },
              ],
            }),
        }),
      },
    });
    const result = await provider("needle-2").doGenerate({
      prompt: userPrompt,
      responseFormat: {
        type: "json",
        name: "weather_record",
        schema: {
          type: "object",
          properties: {
            city: { type: "string" },
            temperature: { type: "number" },
          },
          required: ["city", "temperature"],
        },
      },
    });
    expect(result.content).toContainEqual({
      type: "text",
      text: '{"city":"Paris","temperature":24}',
    });
    expect(result.finishReason.unified).toBe("stop");
  });

  test("produces a valid buffered AI SDK stream", async () => {
    const provider = createNeedleProvider({ model: fakeModel("streamed locally") });
    const result = streamText({ model: provider(), prompt: "Hello" });
    expect(await result.text).toBe("streamed locally");
  });

  test("rejects unknown model IDs", () => {
    const provider = createNeedleProvider({ model: fakeModel() });
    expect(() => provider("not-a-model")).toThrow(/No such languageModel/);
  });
});
