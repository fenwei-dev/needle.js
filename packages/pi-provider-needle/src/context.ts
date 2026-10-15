import type { Context, Message, ToolResultMessage } from "@earendil-works/pi-ai";
import { CHAT_MARKERS } from "needle.js";

export type NeedleContextEvent =
  | { readonly type: "user"; readonly text: string }
  | { readonly type: "tool"; readonly text: string; readonly displayText: string };

export interface ConvertedNeedleContext {
  readonly system: string;
  readonly events: readonly NeedleContextEvent[];
  readonly rawPrompt: string;
  readonly latestToolResult?: string;
}

export function convertContext(context: Context): ConvertedNeedleContext {
  const events: NeedleContextEvent[] = [];
  const raw: string[] = [];
  let latestToolResult: string | undefined;

  if (context.systemPrompt) {
    raw.push(`${CHAT_MARKERS.imStart}system\n${context.systemPrompt}${CHAT_MARKERS.imEnd}\n`);
  }

  for (let index = 0; index < context.messages.length; index++) {
    const message = context.messages[index];
    if (!message) continue;

    if (message.role === "user") {
      const text = messageText(message);
      if (text) {
        events.push({ type: "user", text });
        raw.push(`${CHAT_MARKERS.imStart}user\n${text}${CHAT_MARKERS.imEnd}\n`);
      }
      continue;
    }

    if (message.role === "assistant") {
      const text = message.content
        .filter((part) => part.type === "text")
        .map((part) => (part.type === "text" ? part.text : ""))
        .join("\n");
      const thinking = message.content
        .filter((part) => part.type === "thinking")
        .map((part) => (part.type === "thinking" ? part.thinking : ""))
        .join("\n");
      const calls = message.content
        .filter((part) => part.type === "toolCall")
        .map((part) =>
          part.type === "toolCall"
            ? { name: part.name, arguments: part.arguments }
            : { name: "", arguments: {} },
        );
      let content = text;
      if (thinking) {
        content = `${CHAT_MARKERS.thinkStart}\n${thinking}\n${CHAT_MARKERS.thinkEnd}\n${content}`;
      }
      if (calls.length > 0) {
        content += `${CHAT_MARKERS.toolCallStart}${JSON.stringify(calls)}${CHAT_MARKERS.toolCallEnd}`;
      }
      raw.push(`${CHAT_MARKERS.imStart}assistant\n${content}${CHAT_MARKERS.imEnd}\n`);
      continue;
    }

    const grouped: ToolResultMessage[] = [message];
    while (context.messages[index + 1]?.role === "toolResult") {
      grouped.push(context.messages[++index] as ToolResultMessage);
    }
    const values = grouped.map(toolResultValue);
    const text = JSON.stringify(values);
    latestToolResult = values.length === 1 ? displayValue(values[0]) : displayValue(values);
    events.push({ type: "tool", text, displayText: latestToolResult });
    raw.push(
      `${CHAT_MARKERS.imStart}user\n${CHAT_MARKERS.toolResultStart}${text}${CHAT_MARKERS.toolResultEnd}${CHAT_MARKERS.imEnd}\n`,
    );
  }

  raw.push(`${CHAT_MARKERS.imStart}assistant\n`);
  return {
    system: context.systemPrompt ?? "",
    events,
    rawPrompt: raw.join(""),
    ...(latestToolResult === undefined ? {} : { latestToolResult }),
  };
}

function messageText(message: Extract<Message, { role: "user" }>): string {
  if (typeof message.content === "string") return message.content;
  return message.content
    .filter((part) => part.type === "text")
    .map((part) => (part.type === "text" ? part.text : ""))
    .join("\n");
}

function toolResultValue(message: ToolResultMessage): unknown {
  const text = message.content
    .filter((part) => part.type === "text")
    .map((part) => (part.type === "text" ? part.text : ""))
    .join("\n");
  if (message.isError) return { error: text || `Tool ${message.toolName} failed` };
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return text;
  }
}

function displayValue(value: unknown): string {
  return typeof value === "string" ? value : JSON.stringify(value);
}
