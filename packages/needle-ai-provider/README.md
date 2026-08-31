# needle-ai-provider

A [Vercel AI SDK](https://ai-sdk.dev/) language-model provider for running Needle 2 locally through [`needle.js`](../needle.js).

```bash
bun add ai needle-ai-provider needle.js
```

## Basic use

```ts
import { generateText } from "ai";
import { createNeedleProvider } from "needle-ai-provider";

const needle = createNeedleProvider({
  weights: "download",
  backend: "cpu",
});

const result = await generateText({
  model: needle("needle-2"),
  prompt: "Explain what a tool-calling model does.",
});

console.log(result.text);
await needle.dispose();
```

Needle 2 is primarily a tool-calling model. Raw text generation is available for AI SDK compatibility, but tool mode is the intended path.

## Tool loop

```ts
import {
  generateText,
  jsonSchema,
  stepCountIs,
  tool,
} from "ai";
import { createNeedleProvider } from "needle-ai-provider";

const needle = createNeedleProvider({ weights: "download" });

const result = await generateText({
  model: needle(),
  prompt: "What's the weather in Lagos?",
  stopWhen: stepCountIs(4),
  tools: {
    get_weather: tool({
      description: "Get the current weather for a city",
      inputSchema: jsonSchema<{ city: string }>({
        type: "object",
        properties: { city: { type: "string" } },
        required: ["city"],
      }),
      execute: async ({ city }) => ({ city, temperature: 27, sky: "clear" }),
    }),
  },
});

console.log(result.toolCalls);
console.log(result.toolResults);
console.log(result.text);
```

The provider maps AI SDK function tools to Needle's byte-constrained JSON Schema grammar. Tool calls are returned as standard AI SDK `tool-call` content. On the next AI SDK step, the provider deterministically replays the conversation into a fresh Needle session and feeds the tool results back to the model.

## Streaming

```ts
import { streamText } from "ai";

const result = streamText({
  model: needle(),
  prompt: "Hello",
});

for await (const chunk of result.textStream) {
  process.stdout.write(chunk);
}
```

The current provider emits an AI SDK-compatible buffered stream after local inference completes. Raw `needle.js` generation can provide token callbacks when token-by-token display is required.

## Structured output

AI SDK JSON response formats are implemented as a synthetic Needle extraction tool. Calls with a JSON schema return the extracted arguments as JSON text.

## Configuration

`createNeedleProvider()` accepts the normal `NeedleModel.load()` settings:

```ts
const needle = createNeedleProvider({
  weights: "./models/needle2.cact",
  backend: "typegpu",
  backendOptions: { device },
  agentOptions: {
    prefixSinkTokens: 160,
    maximumRetrievedTools: 5,
  },
});
```

A preloaded model can be shared:

```ts
const needle = createNeedleProvider({
  model,
  disposeModel: false,
});
```

Call `provider.dispose()` to release a model loaded by the provider. Caller-owned models are retained unless `disposeModel: true` is specified.

Per-call Needle options can be supplied through AI SDK provider options under the `needle` key:

```ts
providerOptions: {
  needle: {
    reasoningTokens: 128,
    prefixSinkTokens: 160,
    maxCallsPerTurn: 4,
    toolTokenBudget: 180,
    maximumRetrievedTools: 5,
  },
}
```

Unsupported AI SDK settings are returned through the standard `warnings` field. The provider implements the AI SDK v7 `LanguageModelV4` and `ProviderV4` contracts.
