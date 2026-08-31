import { invariant, NeedleError } from "../errors.js";
import type { JsonSchema, JsonValue, NeedleTool } from "./schema.js";

const textEncoder = new TextEncoder();
const fatalDecoder = new TextDecoder("utf-8", { fatal: true });

interface ValueFrame {
  readonly kind: "value";
  readonly schema: JsonSchema;
}

interface StringFrame {
  readonly kind: "string";
  readonly schema: JsonSchema;
  readonly bytes: number[];
  started: boolean;
  escaped: boolean;
  unicodeRemaining: number;
}

interface NumberFrame {
  readonly kind: "number";
  readonly schema: JsonSchema;
  text: string;
}

interface LiteralFrame {
  readonly kind: "literal";
  readonly bytes: Uint8Array;
  readonly value: JsonValue;
  offset: number;
}

interface ChoiceCandidate {
  readonly bytes: Uint8Array;
  readonly value: JsonValue;
  offset: number;
}

interface ChoiceFrame {
  readonly kind: "choice";
  candidates: ChoiceCandidate[];
}

interface ArrayFrame {
  readonly kind: "array";
  readonly schema: JsonSchema;
  phase: "open" | "first-or-end" | "value" | "after-value";
  readonly values: JsonValue[];
}

interface ObjectFrame {
  readonly kind: "object";
  readonly schema: JsonSchema;
  phase: "open" | "key-or-end" | "key-required" | "key" | "colon" | "value" | "after-value";
  readonly value: Record<string, JsonValue>;
  readonly used: string[];
  currentKey?: string;
}

interface KeyCandidate {
  readonly name: string;
  readonly bytes: Uint8Array;
  offset: number;
}

interface KeyFrame {
  readonly kind: "key";
  candidates: KeyCandidate[];
}

type Frame =
  | ValueFrame
  | StringFrame
  | NumberFrame
  | LiteralFrame
  | ChoiceFrame
  | ArrayFrame
  | ObjectFrame
  | KeyFrame;

function cloneFrame(frame: Frame): Frame {
  switch (frame.kind) {
    case "value":
      return { ...frame };
    case "string":
      return { ...frame, bytes: [...frame.bytes] };
    case "number":
      return { ...frame };
    case "literal":
      return { ...frame };
    case "choice":
      return { ...frame, candidates: frame.candidates.map((candidate) => ({ ...candidate })) };
    case "array":
      return { ...frame, values: structuredValueClone(frame.values) as JsonValue[] };
    case "object":
      return {
        ...frame,
        value: structuredValueClone(frame.value) as Record<string, JsonValue>,
        used: [...frame.used],
      };
    case "key":
      return { ...frame, candidates: frame.candidates.map((candidate) => ({ ...candidate })) };
  }
}

function structuredValueClone(value: JsonValue): JsonValue {
  if (Array.isArray(value)) return value.map(structuredValueClone);
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([key, child]) => [key, structuredValueClone(child)]),
    );
  }
  return value;
}

function deepEqual(left: JsonValue, right: JsonValue): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}

function resolveReference(schema: JsonSchema, root: JsonSchema): JsonSchema {
  if (!schema.$ref) return schema;
  invariant(
    schema.$ref.startsWith("#/"),
    "INVALID_TOOL_SCHEMA",
    `Only local JSON Schema refs are supported; got ${schema.$ref}`,
  );
  let current: unknown = root;
  for (const rawPart of schema.$ref.slice(2).split("/")) {
    const part = rawPart.replaceAll("~1", "/").replaceAll("~0", "~");
    invariant(
      current !== null && typeof current === "object" && part in current,
      "INVALID_TOOL_SCHEMA",
      `Unresolved JSON Schema ref ${schema.$ref}`,
    );
    current = (current as Record<string, unknown>)[part];
  }
  invariant(
    current !== null && typeof current === "object",
    "INVALID_TOOL_SCHEMA",
    `JSON Schema ref ${schema.$ref} does not point to a schema`,
  );
  const { $ref: _ignored, ...local } = schema;
  return { ...(current as JsonSchema), ...local };
}

function schemaAlternatives(schema: JsonSchema, root: JsonSchema): JsonSchema[] {
  const resolved = resolveReference(schema, root);
  const choices = resolved.oneOf ?? resolved.anyOf;
  if (!choices) return [resolved];
  return choices.flatMap((choice) => schemaAlternatives(choice, root));
}

function possibleTypes(schema: JsonSchema): string[] {
  if (schema.type && typeof schema.type !== "string") return [...schema.type];
  if (typeof schema.type === "string") return [schema.type];
  if (schema.properties) return ["object"];
  if (schema.items) return ["array"];
  if (schema.const !== undefined) return [jsonType(schema.const)];
  if (schema.enum?.length) return [...new Set(schema.enum.map(jsonType))];
  return ["string"];
}

function jsonType(value: JsonValue): string {
  if (value === null) return "null";
  if (Array.isArray(value)) return "array";
  return typeof value;
}

function typeStartsWith(type: string, byte: number): boolean {
  if (type === "string") return byte === 0x22;
  if (type === "object") return byte === 0x7b;
  if (type === "array") return byte === 0x5b;
  if (type === "boolean") return byte === 0x74 || byte === 0x66;
  if (type === "null") return byte === 0x6e;
  if (type === "number" || type === "integer")
    return byte === 0x2d || (byte >= 0x30 && byte <= 0x39);
  return false;
}

function validateString(value: string, schema: JsonSchema): boolean {
  const length = Array.from(value).length;
  if (schema.minLength !== undefined && length < schema.minLength) return false;
  if (schema.maxLength !== undefined && length > schema.maxLength) return false;
  if (schema.pattern !== undefined) {
    try {
      if (!new RegExp(schema.pattern).test(value)) return false;
    } catch (cause) {
      throw new NeedleError(
        "INVALID_TOOL_SCHEMA",
        `Invalid JSON Schema pattern ${schema.pattern}`,
        { cause },
      );
    }
  }
  if (schema.format === "email" && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value)) return false;
  if (
    schema.format === "uuid" &&
    !/^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(value)
  )
    return false;
  return validateEnum(value, schema);
}

function validateNumber(value: number, schema: JsonSchema): boolean {
  if (!Number.isFinite(value)) return false;
  const types = possibleTypes(schema);
  if (types.includes("integer") && !types.includes("number") && !Number.isInteger(value))
    return false;
  if (schema.minimum !== undefined && value < schema.minimum) return false;
  if (schema.maximum !== undefined && value > schema.maximum) return false;
  if (schema.exclusiveMinimum !== undefined && value <= schema.exclusiveMinimum) return false;
  if (schema.exclusiveMaximum !== undefined && value >= schema.exclusiveMaximum) return false;
  if (schema.multipleOf !== undefined) {
    const quotient = value / schema.multipleOf;
    if (Math.abs(quotient - Math.round(quotient)) > 1e-9 * Math.max(1, Math.abs(quotient)))
      return false;
  }
  return validateEnum(value, schema);
}

function validateEnum(value: JsonValue, schema: JsonSchema): boolean {
  if (schema.const !== undefined && !deepEqual(value, schema.const)) return false;
  if (schema.enum && !schema.enum.some((candidate) => deepEqual(value, candidate))) return false;
  return true;
}

function isNumberPrefix(text: string): boolean {
  // JSON number DFA, allowing states that can become valid with more bytes.
  let index = 0;
  if (text[index] === "-") index++;
  if (index === text.length) return true;
  if (text[index] === "0") {
    index++;
    if (index < text.length && /[0-9]/.test(text[index] ?? "")) return false;
  } else if (/[1-9]/.test(text[index] ?? "")) {
    while (/[0-9]/.test(text[index] ?? "")) index++;
  } else {
    return false;
  }
  if (text[index] === ".") {
    index++;
    while (/[0-9]/.test(text[index] ?? "")) index++;
  }
  if (text[index] === "e" || text[index] === "E") {
    index++;
    if (text[index] === "+" || text[index] === "-") index++;
    while (/[0-9]/.test(text[index] ?? "")) index++;
  }
  return index === text.length;
}

function isCompleteNumber(text: string): boolean {
  return /^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$/.test(text);
}

/** Incremental compact-JSON grammar compiled from a JSON Schema. */
export class JsonSchemaGrammar {
  readonly rootSchema: JsonSchema;
  #stack: Frame[];
  #complete = false;
  #value: JsonValue | undefined;

  constructor(schema: JsonSchema) {
    this.rootSchema = schema;
    this.#stack = [{ kind: "value", schema }];
  }

  get complete(): boolean {
    if (this.#complete) return true;
    if (this.#stack.length === 1) {
      const frame = this.#stack[0];
      if (
        frame?.kind === "choice" &&
        frame.candidates.some((candidate) => candidate.offset === candidate.bytes.length)
      )
        return true;
    }
    return false;
  }

  get value(): JsonValue | undefined {
    if (this.#complete) return this.#value;
    const frame = this.#stack.length === 1 ? this.#stack[0] : undefined;
    if (frame?.kind === "choice") {
      return frame.candidates.find((candidate) => candidate.offset === candidate.bytes.length)
        ?.value;
    }
    return undefined;
  }

  clone(): JsonSchemaGrammar {
    const clone = new JsonSchemaGrammar(this.rootSchema);
    clone.#stack = this.#stack.map(cloneFrame);
    clone.#complete = this.#complete;
    clone.#value = this.#value === undefined ? undefined : structuredValueClone(this.#value);
    return clone;
  }

  feed(bytes: Uint8Array): boolean {
    for (const byte of bytes) if (!this.feedByte(byte)) return false;
    return true;
  }

  feedByte(byte: number): boolean {
    if (this.#complete) return false;
    for (let retry = 0; retry < 16; retry++) {
      const frame = this.#stack[this.#stack.length - 1];
      if (!frame) return false;
      switch (frame.kind) {
        case "value": {
          const alternatives = schemaAlternatives(frame.schema, this.rootSchema);
          const resolvedAlternatives = alternatives.map((alternative) =>
            resolveReference(alternative, this.rootSchema),
          );
          const enumValues = resolvedAlternatives.flatMap((alternative) => {
            if (alternative.const !== undefined) return [alternative.const];
            return alternative.enum ? [...alternative.enum] : [];
          });
          if (
            enumValues.length > 0 &&
            resolvedAlternatives.every(
              (alternative) => alternative.const !== undefined || alternative.enum !== undefined,
            )
          ) {
            this.#stack[this.#stack.length - 1] = {
              kind: "choice",
              candidates: enumValues.map((value) => ({
                bytes: textEncoder.encode(JSON.stringify(value)),
                value,
                offset: 0,
              })),
            };
            continue;
          }
          const matching = alternatives.filter((alternative) =>
            possibleTypes(resolveReference(alternative, this.rootSchema)).some((type) =>
              typeStartsWith(type, byte),
            ),
          );
          if (matching.length === 0) return false;
          const schema = resolveReference(matching[0] ?? frame.schema, this.rootSchema);
          const types = possibleTypes(schema);
          const selected = types.find((type) => typeStartsWith(type, byte));
          if (!selected) return false;
          if (selected === "string")
            this.#stack[this.#stack.length - 1] = {
              kind: "string",
              schema,
              bytes: [],
              started: false,
              escaped: false,
              unicodeRemaining: 0,
            };
          else if (selected === "number" || selected === "integer")
            this.#stack[this.#stack.length - 1] = { kind: "number", schema, text: "" };
          else if (selected === "boolean") {
            const value = byte === 0x74;
            this.#stack[this.#stack.length - 1] = {
              kind: "literal",
              bytes: textEncoder.encode(value ? "true" : "false"),
              value,
              offset: 0,
            };
          } else if (selected === "null")
            this.#stack[this.#stack.length - 1] = {
              kind: "literal",
              bytes: textEncoder.encode("null"),
              value: null,
              offset: 0,
            };
          else if (selected === "array")
            this.#stack[this.#stack.length - 1] = {
              kind: "array",
              schema,
              phase: "open",
              values: [],
            };
          else if (selected === "object")
            this.#stack[this.#stack.length - 1] = {
              kind: "object",
              schema,
              phase: "open",
              value: {},
              used: [],
            };
          else return false;
          continue;
        }
        case "string": {
          if (!frame.started) {
            if (byte !== 0x22) return false;
            frame.started = true;
            frame.bytes.push(byte);
            return true;
          }
          if (frame.unicodeRemaining > 0) {
            if (
              !(
                (byte >= 0x30 && byte <= 0x39) ||
                (byte >= 0x41 && byte <= 0x46) ||
                (byte >= 0x61 && byte <= 0x66)
              )
            )
              return false;
            frame.unicodeRemaining--;
            frame.bytes.push(byte);
            return true;
          }
          if (frame.escaped) {
            if (![0x22, 0x5c, 0x2f, 0x62, 0x66, 0x6e, 0x72, 0x74, 0x75].includes(byte))
              return false;
            frame.escaped = false;
            if (byte === 0x75) frame.unicodeRemaining = 4;
            frame.bytes.push(byte);
            return true;
          }
          if (byte === 0x5c) {
            frame.escaped = true;
            frame.bytes.push(byte);
            return true;
          }
          if (byte < 0x20) return false;
          frame.bytes.push(byte);
          if (byte !== 0x22) return true;
          let value: string;
          try {
            value = JSON.parse(fatalDecoder.decode(Uint8Array.from(frame.bytes))) as string;
          } catch {
            return false;
          }
          if (!validateString(value, frame.schema)) return false;
          this.#finish(value);
          return true;
        }
        case "number": {
          const character = String.fromCharCode(byte);
          if (/[0-9eE+\-.]/.test(character)) {
            const next = frame.text + character;
            if (!isNumberPrefix(next)) return false;
            frame.text = next;
            return true;
          }
          if (!isCompleteNumber(frame.text)) return false;
          const value = Number(frame.text);
          if (!validateNumber(value, frame.schema)) return false;
          this.#finish(value);
          continue;
        }
        case "literal": {
          if (byte !== frame.bytes[frame.offset]) return false;
          frame.offset++;
          if (frame.offset === frame.bytes.length) this.#finish(frame.value);
          return true;
        }
        case "choice": {
          const continuing = frame.candidates.filter(
            (candidate) =>
              candidate.offset < candidate.bytes.length &&
              candidate.bytes[candidate.offset] === byte,
          );
          if (continuing.length > 0) {
            for (const candidate of continuing) candidate.offset++;
            frame.candidates = continuing;
            return true;
          }
          const completed = frame.candidates.find(
            (candidate) => candidate.offset === candidate.bytes.length,
          );
          if (!completed) return false;
          this.#finish(completed.value);
          continue;
        }
        case "array": {
          if (frame.phase === "open") {
            if (byte !== 0x5b) return false;
            frame.phase = "first-or-end";
            return true;
          }
          const minimum = frame.schema.minItems ?? 0;
          const maximum = frame.schema.maxItems ?? 32;
          if (frame.phase === "first-or-end") {
            if (byte === 0x5d) {
              if (frame.values.length < minimum) return false;
              if (!validateEnum(frame.values, frame.schema)) return false;
              this.#finish(frame.values);
              return true;
            }
            if (frame.values.length >= maximum) return false;
            frame.phase = "value";
            this.#stack.push({ kind: "value", schema: frame.schema.items ?? {} });
            continue;
          }
          if (frame.phase === "after-value") {
            if (byte === 0x2c && frame.values.length < maximum) {
              frame.phase = "value";
              this.#stack.push({ kind: "value", schema: frame.schema.items ?? {} });
              return true;
            }
            if (byte === 0x5d && frame.values.length >= minimum) {
              if (frame.schema.uniqueItems) {
                const serialized = frame.values.map((value) => JSON.stringify(value));
                if (new Set(serialized).size !== serialized.length) return false;
              }
              if (!validateEnum(frame.values, frame.schema)) return false;
              this.#finish(frame.values);
              return true;
            }
            return false;
          }
          return false;
        }
        case "object": {
          if (frame.phase === "open") {
            if (byte !== 0x7b) return false;
            frame.phase = "key-or-end";
            return true;
          }
          const properties = frame.schema.properties ?? {};
          const required = frame.schema.required ?? [];
          const minimum = frame.schema.minProperties ?? 0;
          const maximum = frame.schema.maxProperties ?? Object.keys(properties).length;
          if (frame.phase === "key-or-end" || frame.phase === "key-required") {
            if (byte === 0x7d && frame.phase === "key-or-end") {
              if (
                frame.used.length < minimum ||
                required.some((name) => !frame.used.includes(name))
              )
                return false;
              if (!validateEnum(frame.value, frame.schema)) return false;
              this.#finish(frame.value);
              return true;
            }
            if (byte !== 0x22 || frame.used.length >= maximum) return false;
            const candidates = Object.keys(properties)
              .filter((name) => !frame.used.includes(name))
              .map((name) => ({ name, bytes: textEncoder.encode(name), offset: 0 }));
            if (candidates.length === 0) return false;
            frame.phase = "key";
            this.#stack.push({ kind: "key", candidates });
            return true;
          }
          if (frame.phase === "colon") {
            if (byte !== 0x3a || frame.currentKey === undefined) return false;
            frame.phase = "value";
            this.#stack.push({
              kind: "value",
              schema:
                properties[frame.currentKey] ??
                (typeof frame.schema.additionalProperties === "object"
                  ? frame.schema.additionalProperties
                  : {}),
            });
            return true;
          }
          if (frame.phase === "after-value") {
            if (byte === 0x2c && frame.used.length < maximum) {
              frame.phase = "key-required";
              return true;
            }
            if (
              byte === 0x7d &&
              frame.used.length >= minimum &&
              required.every((name) => frame.used.includes(name))
            ) {
              if (!validateEnum(frame.value, frame.schema)) return false;
              this.#finish(frame.value);
              return true;
            }
            return false;
          }
          return false;
        }
        case "key": {
          if (byte === 0x22) {
            const complete = frame.candidates.find(
              (candidate) => candidate.offset === candidate.bytes.length,
            );
            if (!complete) return false;
            this.#stack.pop();
            const parent = this.#stack[this.#stack.length - 1];
            if (parent?.kind !== "object") return false;
            parent.currentKey = complete.name;
            parent.phase = "colon";
            return true;
          }
          const continuing = frame.candidates.filter(
            (candidate) =>
              candidate.offset < candidate.bytes.length &&
              candidate.bytes[candidate.offset] === byte,
          );
          if (continuing.length === 0) return false;
          for (const candidate of continuing) candidate.offset++;
          frame.candidates = continuing;
          return true;
        }
      }
    }
    return false;
  }

  #finish(value: JsonValue): void {
    this.#stack.pop();
    const parent = this.#stack[this.#stack.length - 1];
    if (!parent) {
      this.#complete = true;
      this.#value = value;
      return;
    }
    if (parent.kind === "array" && parent.phase === "value") {
      parent.values.push(value);
      parent.phase = "after-value";
      return;
    }
    if (parent.kind === "object" && parent.phase === "value" && parent.currentKey !== undefined) {
      parent.value[parent.currentKey] = value;
      parent.used.push(parent.currentKey);
      delete parent.currentKey;
      parent.phase = "after-value";
      return;
    }
    throw new NeedleError(
      "INVALID_TOOL_SCHEMA",
      "Internal JSON Schema grammar stack is inconsistent",
    );
  }
}

export interface GrammarToolCall {
  readonly name: string;
  readonly arguments: Record<string, JsonValue>;
}

type ToolPhase =
  | "array-open"
  | "call-open"
  | "name-literal"
  | "tool-name"
  | "arguments-literal"
  | "arguments"
  | "call-close"
  | "after-call"
  | "done";

interface ToolNameCandidate {
  readonly index: number;
  readonly bytes: Uint8Array;
  offset: number;
}

const NAME_LITERAL = textEncoder.encode('"name":"');
const ARGUMENTS_LITERAL = textEncoder.encode(',"arguments":');

/** Continuous byte grammar for Needle's array-of-tool-calls contract. */
export class ToolCallGrammar {
  readonly tools: readonly NeedleTool[];
  readonly maximumCalls: number;
  phase: ToolPhase = "array-open";
  #literalOffset = 0;
  #nameCandidates: ToolNameCandidate[] = [];
  #currentTool = -1;
  #usedTools: number[] = [];
  #arguments: JsonSchemaGrammar | undefined;
  #calls: GrammarToolCall[] = [];

  constructor(tools: readonly NeedleTool[], maximumCalls = 4) {
    invariant(tools.length > 0, "INVALID_TOOL_SCHEMA", "Tool-call grammar needs at least one tool");
    this.tools = tools;
    this.maximumCalls = Math.max(1, maximumCalls);
  }

  get complete(): boolean {
    return this.phase === "done";
  }

  get calls(): readonly GrammarToolCall[] {
    return this.#calls;
  }

  clone(): ToolCallGrammar {
    const clone = new ToolCallGrammar(this.tools, this.maximumCalls);
    clone.phase = this.phase;
    clone.#literalOffset = this.#literalOffset;
    clone.#nameCandidates = this.#nameCandidates.map((candidate) => ({ ...candidate }));
    clone.#currentTool = this.#currentTool;
    clone.#usedTools = [...this.#usedTools];
    clone.#arguments = this.#arguments?.clone();
    clone.#calls = this.#calls.map((call) => ({
      name: call.name,
      arguments: structuredValueClone(call.arguments as unknown as JsonValue) as Record<
        string,
        JsonValue
      >,
    }));
    return clone;
  }

  feed(bytes: Uint8Array): boolean {
    for (const byte of bytes) if (!this.feedByte(byte)) return false;
    return true;
  }

  feedByte(byte: number): boolean {
    for (let retry = 0; retry < 8; retry++) {
      if (this.phase === "array-open") {
        if (byte !== 0x5b) return false;
        this.phase = "call-open";
        return true;
      }
      if (this.phase === "call-open") {
        if (byte !== 0x7b) return false;
        this.phase = "name-literal";
        this.#literalOffset = 0;
        return true;
      }
      if (this.phase === "name-literal") {
        if (byte !== NAME_LITERAL[this.#literalOffset]) return false;
        this.#literalOffset++;
        if (this.#literalOffset === NAME_LITERAL.length) {
          this.phase = "tool-name";
          this.#nameCandidates = this.tools
            .map((tool, index) => ({ index, bytes: textEncoder.encode(tool.name), offset: 0 }))
            .filter((candidate) => !this.#usedTools.includes(candidate.index));
        }
        return true;
      }
      if (this.phase === "tool-name") {
        if (byte === 0x22) {
          const candidate = this.#nameCandidates.find(
            (entry) => entry.offset === entry.bytes.length,
          );
          if (!candidate) return false;
          this.#currentTool = candidate.index;
          this.phase = "arguments-literal";
          this.#literalOffset = 0;
          return true;
        }
        const continuing = this.#nameCandidates.filter(
          (candidate) =>
            candidate.offset < candidate.bytes.length && candidate.bytes[candidate.offset] === byte,
        );
        if (continuing.length === 0) return false;
        for (const candidate of continuing) candidate.offset++;
        this.#nameCandidates = continuing;
        return true;
      }
      if (this.phase === "arguments-literal") {
        if (byte !== ARGUMENTS_LITERAL[this.#literalOffset]) return false;
        this.#literalOffset++;
        if (this.#literalOffset === ARGUMENTS_LITERAL.length) {
          const tool = this.tools[this.#currentTool];
          if (!tool) return false;
          this.#arguments = new JsonSchemaGrammar(tool.parameters);
          this.phase = "arguments";
        }
        return true;
      }
      if (this.phase === "arguments") {
        const grammar = this.#arguments;
        if (!grammar) return false;
        if (grammar.complete) {
          this.phase = "call-close";
          continue;
        }
        return grammar.feedByte(byte);
      }
      if (this.phase === "call-close") {
        if (byte !== 0x7d) return false;
        const tool = this.tools[this.#currentTool];
        const arguments_ = this.#arguments?.value;
        if (!tool || !arguments_ || Array.isArray(arguments_) || typeof arguments_ !== "object")
          return false;
        this.#calls.push({ name: tool.name, arguments: arguments_ as Record<string, JsonValue> });
        this.#usedTools.push(this.#currentTool);
        this.phase = "after-call";
        return true;
      }
      if (this.phase === "after-call") {
        if (byte === 0x5d) {
          this.phase = "done";
          return true;
        }
        if (
          byte === 0x2c &&
          this.#calls.length < this.maximumCalls &&
          this.#usedTools.length < this.tools.length
        ) {
          this.phase = "call-open";
          return true;
        }
        return false;
      }
      return false;
    }
    return false;
  }
}
