import type { NeedleTokenizer } from "../model/tokenizer.js";
import { serializeTools, type NeedleTool } from "./schema.js";

function words(text: string): string[] {
  return text.toLowerCase().match(/[a-z0-9]+/g) ?? [];
}

/** Lightweight fallback retrieval for catalogues too large for the KV window. */
export function rankToolsBM25(query: string, tools: readonly NeedleTool[]): number[] {
  const documents = tools.map((tool) => words(`${tool.name} ${tool.description ?? ""}`));
  const lengths = documents.map((document) => Math.max(1, document.length));
  const averageLength = lengths.reduce((sum, length) => sum + length, 0) / Math.max(1, lengths.length);
  const scores = new Float64Array(tools.length);
  const queryWords = words(query).filter((word) => word.length >= 2);
  const k1 = 1.5;
  const b = 0.75;

  for (const word of queryWords) {
    const frequencies = documents.map((document) => document.reduce((count, token) => count + Number(token === word), 0));
    const documentFrequency = frequencies.reduce((count, frequency) => count + Number(frequency > 0), 0);
    if (documentFrequency === 0) continue;
    const inverseDocumentFrequency = Math.log(1 + (tools.length - documentFrequency + 0.5) / (documentFrequency + 0.5));
    for (let index = 0; index < tools.length; index++) {
      const frequency = frequencies[index] ?? 0;
      if (frequency === 0) continue;
      scores[index] = (scores[index] ?? 0)
        + inverseDocumentFrequency * frequency * (k1 + 1)
          / (frequency + k1 * (1 - b + b * (lengths[index] ?? 1) / averageLength));
    }
  }

  return tools.map((_, index) => index).sort((left, right) => {
    const difference = (scores[right] ?? 0) - (scores[left] ?? 0);
    return difference === 0 ? left - right : difference;
  });
}

export interface RetrieveToolsOptions {
  readonly tokenizer?: NeedleTokenizer;
  readonly tokenBudget?: number;
  readonly maximumTools?: number;
  readonly minimumTools?: number;
}

export function retrieveTools(
  query: string,
  tools: readonly NeedleTool[],
  options: RetrieveToolsOptions = {},
): NeedleTool[] {
  const maximum = Math.max(1, options.maximumTools ?? 5);
  const minimum = Math.min(maximum, Math.max(1, options.minimumTools ?? 2));
  const budget = options.tokenBudget ?? 180;
  const tokenizer = options.tokenizer;
  const fullText = serializeTools(tools);
  if (tools.length <= maximum && (!tokenizer || tokenizer.encode(fullText).length <= budget)) return [...tools];

  const ranking = rankToolsBM25(query, tools);
  let keep = Math.min(maximum, tools.length);
  while (keep > minimum && tokenizer) {
    const selected = ranking.slice(0, keep).map((index) => tools[index]).filter((tool): tool is NeedleTool => tool !== undefined);
    if (tokenizer.encode(serializeTools(selected)).length <= budget) return selected;
    keep--;
  }
  return ranking.slice(0, Math.max(minimum, keep)).map((index) => tools[index]).filter((tool): tool is NeedleTool => tool !== undefined);
}
