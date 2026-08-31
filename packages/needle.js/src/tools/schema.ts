import { NeedleError, invariant } from "../errors.js";

export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonValue[] | { readonly [key: string]: JsonValue };

export interface JsonSchema {
  readonly $ref?: string;
  readonly $defs?: Readonly<Record<string, JsonSchema>>;
  readonly definitions?: Readonly<Record<string, JsonSchema>>;
  readonly title?: string;
  readonly description?: string;
  readonly type?: "null" | "boolean" | "object" | "array" | "number" | "integer" | "string" | readonly ("null" | "boolean" | "object" | "array" | "number" | "integer" | "string")[];
  readonly enum?: readonly JsonValue[];
  readonly const?: JsonValue;
  readonly anyOf?: readonly JsonSchema[];
  readonly oneOf?: readonly JsonSchema[];
  readonly properties?: Readonly<Record<string, JsonSchema>>;
  readonly required?: readonly string[];
  readonly additionalProperties?: boolean | JsonSchema;
  readonly items?: JsonSchema;
  readonly minItems?: number;
  readonly maxItems?: number;
  readonly uniqueItems?: boolean;
  readonly minimum?: number;
  readonly maximum?: number;
  readonly exclusiveMinimum?: number;
  readonly exclusiveMaximum?: number;
  readonly multipleOf?: number;
  readonly minLength?: number;
  readonly maxLength?: number;
  readonly pattern?: string;
  readonly format?: string;
  readonly minProperties?: number;
  readonly maxProperties?: number;
}

export interface NeedleTool<Arguments extends Record<string, unknown> = Record<string, unknown>, Result = unknown> {
  readonly name: string;
  readonly description?: string;
  readonly parameters: JsonSchema & { readonly type: "object" };
  readonly execute?: (arguments_: Arguments, context: ToolExecutionContext) => Result | Promise<Result>;
}

export interface ToolExecutionContext {
  readonly signal?: AbortSignal;
  readonly callId: string;
  readonly step: number;
}

export interface OpenAIFunctionTool {
  readonly type: "function";
  readonly function: {
    readonly name: string;
    readonly description?: string;
    readonly parameters?: JsonSchema;
  };
}

export interface RawNeedleTool {
  readonly name: string;
  readonly description?: string;
  readonly parameters?: JsonSchema;
  readonly execute?: (arguments_: Record<string, unknown>, context: ToolExecutionContext) => unknown | Promise<unknown>;
}

export type ToolInput = NeedleTool<any, any> | RawNeedleTool | OpenAIFunctionTool;

export interface DefineToolOptions<Arguments extends Record<string, unknown>, Result> {
  readonly name: string;
  readonly description?: string;
  readonly parameters: JsonSchema & { readonly type: "object" };
  readonly execute?: (arguments_: Arguments, context: ToolExecutionContext) => Result | Promise<Result>;
}

/** Defines a schema and its implementation while retaining argument/result types. */
export function defineTool<Arguments extends Record<string, unknown>, Result = unknown>(
  options: DefineToolOptions<Arguments, Result>,
): NeedleTool<Arguments, Result> {
  validateToolName(options.name);
  return options;
}

/** A positional convenience form of {@link defineTool}. */
export function tool<Arguments extends Record<string, unknown>, Result = unknown>(
  name: string,
  description: string,
  parameters: JsonSchema & { readonly type: "object" },
  execute?: (arguments_: Arguments, context: ToolExecutionContext) => Result | Promise<Result>,
): NeedleTool<Arguments, Result> {
  return defineTool({
    name,
    description,
    parameters,
    ...(execute ? { execute } : {}),
  });
}

function validateToolName(name: string): void {
  invariant(typeof name === "string" && name.length > 0 && name.length <= 128, "INVALID_TOOL_SCHEMA", "A tool name must contain 1–128 characters");
  invariant(!/["\\\u0000-\u001f]/.test(name), "INVALID_TOOL_SCHEMA", `Tool name ${JSON.stringify(name)} contains characters that cannot be constrained safely`);
}

export function normalizeTool(input: ToolInput): NeedleTool {
  const raw: RawNeedleTool = "function" in input
    ? {
        name: input.function.name,
        ...(input.function.description === undefined ? {} : { description: input.function.description }),
        ...(input.function.parameters === undefined ? {} : { parameters: input.function.parameters }),
      }
    : input;
  validateToolName(raw.name);
  const parameters = raw.parameters ?? { type: "object", properties: {} };
  invariant(parameters.type === "object" || parameters.type === undefined, "INVALID_TOOL_SCHEMA", `Tool ${raw.name} parameters must be an object schema`);
  const properties = parameters.properties ?? {};
  for (const [name, schema] of Object.entries(properties)) {
    invariant(name.length > 0 && !/["\\\u0000-\u001f]/.test(name), "INVALID_TOOL_SCHEMA", `Tool ${raw.name} has an invalid parameter name ${JSON.stringify(name)}`);
    invariant(schema && typeof schema === "object", "INVALID_TOOL_SCHEMA", `Tool ${raw.name} parameter ${name} has no schema`);
  }
  for (const name of parameters.required ?? []) {
    invariant(name in properties, "INVALID_TOOL_SCHEMA", `Tool ${raw.name} requires unknown parameter ${name}`);
  }
  return {
    name: raw.name,
    ...(raw.description === undefined ? {} : { description: raw.description }),
    parameters: { ...parameters, type: "object", properties },
    ...(raw.execute === undefined ? {} : { execute: raw.execute }),
  };
}

export function normalizeTools(inputs: readonly ToolInput[]): NeedleTool[] {
  const tools = inputs.map(normalizeTool);
  const names = new Set<string>();
  for (const current of tools) {
    if (names.has(current.name)) throw new NeedleError("INVALID_TOOL_SCHEMA", `Duplicate tool name ${current.name}`);
    names.add(current.name);
  }
  return tools;
}

/** Removes executable functions and emits the compact format used in prompts. */
export function serializeTools(tools: readonly NeedleTool[]): string {
  return JSON.stringify(tools.map(({ name, description, parameters }) => ({
    name,
    ...(description === undefined ? {} : { description }),
    parameters,
  })));
}
