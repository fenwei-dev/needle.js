import { describe, expect, test } from "bun:test";
import { JsonSchemaGrammar, ToolCallGrammar } from "../src/tools/grammar.js";
import { normalizeTools, type JsonSchema } from "../src/tools/schema.js";

const encode = (value: string) => new TextEncoder().encode(value);

const parameters: JsonSchema = {
  type: "object",
  properties: {
    enabled: { type: "boolean" },
    level: { type: "integer", minimum: 0, maximum: 10 },
    mode: { type: "string", enum: ["eco", "boost"] },
    tags: { type: "array", items: { type: "string", minLength: 1 }, minItems: 1, maxItems: 3 },
    location: {
      type: "object",
      properties: { lat: { type: "number" }, lon: { type: "number" } },
      required: ["lat", "lon"],
    },
  },
  required: ["enabled", "level", "mode", "tags"],
};

describe("JSON Schema grammar", () => {
  test("accepts nested, schema-conforming compact JSON", () => {
    const grammar = new JsonSchemaGrammar(parameters);
    const text = '{"mode":"eco","level":3,"enabled":true,"tags":["a","b"],"location":{"lon":2.5,"lat":-1}}';
    // Feed uneven chunks to exercise resumable strings/numbers/literals.
    for (let offset = 0; offset < text.length; offset += 7) {
      expect(grammar.feed(encode(text.slice(offset, offset + 7)))).toBe(true);
    }
    expect(grammar.complete).toBe(true);
    expect(grammar.value).toEqual({
      mode: "eco",
      level: 3,
      enabled: true,
      tags: ["a", "b"],
      location: { lon: 2.5, lat: -1 },
    });
  });

  test("rejects unknown, duplicate, missing, enum, and range violations", () => {
    const invalid = [
      '{"enabled":true,"level":3,"mode":"eco","tags":["x"],"unknown":1}',
      '{"enabled":true,"level":3,"mode":"eco","mode":"boost","tags":["x"]}',
      '{"enabled":true,"level":3,"mode":"eco"}',
      '{"enabled":true,"level":3,"mode":"invalid","tags":["x"]}',
      '{"enabled":true,"level":11,"mode":"eco","tags":["x"]}',
      '{"enabled":true,"level":3,"mode":"eco","tags":[]}',
      '{"enabled":true,"level":3,"mode":"eco","tags":["x"],}',
    ];
    for (const text of invalid) {
      const grammar = new JsonSchemaGrammar(parameters);
      const accepted = grammar.feed(encode(text));
      expect(accepted && grammar.complete).toBe(false);
    }
  });

  test("clones independently for token-candidate speculation", () => {
    const base = new JsonSchemaGrammar({ type: "string", enum: ["alpha", "alpine"] });
    expect(base.feed(encode('"alp'))).toBe(true);
    const left = base.clone();
    const right = base.clone();
    expect(left.feed(encode('ha"'))).toBe(true);
    expect(right.feed(encode('ine"'))).toBe(true);
    expect(left.complete).toBe(true);
    expect(right.complete).toBe(true);
    expect(left.value).toBe("alpha");
    expect(right.value).toBe("alpine");
  });
});

describe("tool-call grammar", () => {
  const tools = normalizeTools([
    { name: "configure", description: "Configure a device", parameters },
    { name: "ping", parameters: { type: "object", properties: {} } },
  ]);

  test("accepts multiple non-repeated tool calls", () => {
    const grammar = new ToolCallGrammar(tools, 2);
    const output = '[{"name":"configure","arguments":{"enabled":false,"level":5,"mode":"boost","tags":["lab"]}},{"name":"ping","arguments":{}}]';
    expect(grammar.feed(encode(output))).toBe(true);
    expect(grammar.complete).toBe(true);
    expect(grammar.calls).toEqual([
      { name: "configure", arguments: { enabled: false, level: 5, mode: "boost", tags: ["lab"] } },
      { name: "ping", arguments: {} },
    ]);
  });

  test("rejects unregistered and repeated tools", () => {
    for (const output of [
      '[{"name":"missing","arguments":{}}]',
      '[{"name":"ping","arguments":{}},{"name":"ping","arguments":{}}]',
    ]) {
      const grammar = new ToolCallGrammar(tools, 3);
      expect(grammar.feed(encode(output)) && grammar.complete).toBe(false);
    }
  });
});
