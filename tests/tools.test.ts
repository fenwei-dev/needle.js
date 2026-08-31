import { describe, expect, test } from "bun:test";
import { defineTool, normalizeTools, serializeTools } from "../src/tools/schema.js";
import { rankToolsBM25, retrieveTools } from "../src/tools/retrieval.js";

describe("tool definitions", () => {
  test("normalizes raw and OpenAI function schemas", () => {
    const tools = normalizeTools([
      { name: "ping", parameters: { type: "object", properties: {} } },
      {
        type: "function",
        function: {
          name: "weather",
          description: "Get city weather",
          parameters: {
            type: "object",
            properties: { city: { type: "string" } },
            required: ["city"],
          },
        },
      },
    ]);
    expect(tools.map((tool) => tool.name)).toEqual(["ping", "weather"]);
    expect(JSON.parse(serializeTools(tools))[1].parameters.required).toEqual(["city"]);
  });

  test("retains generic executor types", async () => {
    const add = defineTool<{ left: number; right: number }, number>({
      name: "add",
      parameters: {
        type: "object",
        properties: { left: { type: "number" }, right: { type: "number" } },
        required: ["left", "right"],
      },
      execute: ({ left, right }) => left + right,
    });
    expect(await add.execute?.({ left: 2, right: 3 }, { callId: "x", step: 0 })).toBe(5);
  });

  test("rejects duplicates and invalid required properties", () => {
    expect(() => normalizeTools([{ name: "x" }, { name: "x" }])).toThrow(/duplicate/i);
    expect(() => normalizeTools([{
      name: "x",
      parameters: { type: "object", properties: {}, required: ["missing"] },
    }])).toThrow(/unknown parameter/i);
  });
});

describe("tool retrieval", () => {
  const tools = normalizeTools([
    { name: "weather", description: "Current forecast, rain, and temperature" },
    { name: "calendar", description: "Create and edit calendar events" },
    { name: "music", description: "Play a song or album" },
    { name: "email", description: "Send electronic mail to a contact" },
  ]);

  test("BM25 ranks semantically matching words first", () => {
    const order = rankToolsBM25("will it rain in the forecast?", tools);
    expect(tools[order[0] ?? -1]?.name).toBe("weather");
  });

  test("limits the selected catalogue", () => {
    const selected = retrieveTools("play an album", tools, { maximumTools: 2, minimumTools: 2 });
    expect(selected).toHaveLength(2);
    expect(selected[0]?.name).toBe("music");
  });
});
