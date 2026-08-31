import type { LanguageModelV4Prompt, LanguageModelV4ToolResultOutput } from "@ai-sdk/provider";
import { InvalidPromptError, UnsupportedFunctionalityError } from "@ai-sdk/provider";
import { CHAT_MARKERS } from "needle.js";

export type NeedlePromptEvent =
  | { readonly type: "user"; readonly text: string }
  | { readonly type: "tool"; readonly text: string; readonly displayText: string };

export interface ConvertedNeedlePrompt {
  readonly system: string;
  readonly events: readonly NeedlePromptEvent[];
  readonly rawPrompt: string;
  readonly latestToolResult?: string;
}

export function convertPrompt(prompt: LanguageModelV4Prompt): ConvertedNeedlePrompt {
  if (prompt.length === 0) {
    throw new InvalidPromptError({ prompt, message: "Needle requires at least one message" });
  }

  const systems: string[] = [];
  const events: NeedlePromptEvent[] = [];
  const raw: string[] = [];
  let latestToolResult: string | undefined;

  for (const message of prompt) {
    if (message.role === "system") {
      systems.push(message.content);
      raw.push(`${CHAT_MARKERS.imStart}system\n${message.content}${CHAT_MARKERS.imEnd}\n`);
      continue;
    }

    if (message.role === "user") {
      const text = message.content
        .map((part) => {
          if (part.type === "file") return unsupportedFile();
          return part.text;
        })
        .filter(Boolean)
        .join("\n");
      if (text.length === 0) {
        throw new InvalidPromptError({ prompt, message: "Needle only accepts text user messages" });
      }
      events.push({ type: "user", text });
      raw.push(`${CHAT_MARKERS.imStart}user\n${text}${CHAT_MARKERS.imEnd}\n`);
      continue;
    }

    if (message.role === "assistant") {
      const text: string[] = [];
      const reasoning: string[] = [];
      const calls: Array<{ name: string; arguments: unknown }> = [];
      for (const part of message.content) {
        if (part.type === "text") text.push(part.text);
        else if (part.type === "reasoning") reasoning.push(part.text);
        else if (part.type === "tool-call")
          calls.push({ name: part.toolName, arguments: part.input });
        else if (part.type === "file" || part.type === "reasoning-file") unsupportedFile();
      }
      let content = "";
      if (reasoning.length > 0) {
        content += `${CHAT_MARKERS.thinkStart}\n${reasoning.join("\n")}\n${CHAT_MARKERS.thinkEnd}\n`;
      }
      content += text.join("\n");
      if (calls.length > 0) {
        content += `${CHAT_MARKERS.toolCallStart}${JSON.stringify(calls)}${CHAT_MARKERS.toolCallEnd}`;
      }
      raw.push(`${CHAT_MARKERS.imStart}assistant\n${content}${CHAT_MARKERS.imEnd}\n`);
      continue;
    }

    const results = message.content
      .filter((part) => part.type === "tool-result")
      .map((part) => ({ name: part.toolName, result: toolOutputValue(part.output) }));
    const replayValues = results.map(({ result }) => result);
    const text = JSON.stringify(replayValues);
    latestToolResult =
      results.length === 1 ? displayValue(results[0]?.result) : displayValue(results);
    events.push({ type: "tool", text, displayText: latestToolResult });
    raw.push(
      `${CHAT_MARKERS.imStart}user\n${CHAT_MARKERS.toolResultStart}${text}${CHAT_MARKERS.toolResultEnd}${CHAT_MARKERS.imEnd}\n`,
    );
  }

  raw.push(`${CHAT_MARKERS.imStart}assistant\n`);
  return {
    system: systems.join("\n"),
    events,
    rawPrompt: raw.join(""),
    ...(latestToolResult === undefined ? {} : { latestToolResult }),
  };
}

function unsupportedFile(): never {
  throw new UnsupportedFunctionalityError({ functionality: "file and custom prompt parts" });
}

function toolOutputValue(output: LanguageModelV4ToolResultOutput): unknown {
  switch (output.type) {
    case "text":
    case "error-text":
      return output.value;
    case "json":
    case "error-json":
      return output.value;
    case "execution-denied":
      return { error: output.reason ?? "Tool execution denied" };
    case "content":
      return output.value
        .filter((part) => part.type === "text")
        .map((part) => (part.type === "text" ? part.text : ""))
        .join("\n");
  }
}

function displayValue(value: unknown): string {
  return typeof value === "string" ? value : JSON.stringify(value);
}
