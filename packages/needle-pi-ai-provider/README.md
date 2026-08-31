# needle-pi-ai-provider

A native [`@earendil-works/pi-ai`](https://www.npmjs.com/package/@earendil-works/pi-ai) provider for local Needle 2 inference, plus an installable [pi](https://pi.dev/) extension.

## pi-ai library

```bash
bun add @earendil-works/pi-ai needle-pi-ai-provider needle.js
```

```ts
import { createModels, Type, type Context, type Tool } from "@earendil-works/pi-ai";
import { createNeedlePiProvider } from "needle-pi-ai-provider";

const provider = createNeedlePiProvider({
  weights: "download",
  backend: "cpu",
});

const models = createModels();
models.setProvider(provider);

const model = models.getModel("needle", "needle-2")!;
const tools: Tool[] = [
  {
    name: "get_weather",
    description: "Get current weather for a city",
    parameters: Type.Object({ city: Type.String() }),
  },
];
const context: Context = {
  systemPrompt: "Use the available local tools.",
  messages: [{ role: "user", content: "Weather in Lagos?", timestamp: Date.now() }],
  tools,
};

const first = await models.completeSimple(model, context);
context.messages.push(first);

for (const block of first.content) {
  if (block.type !== "toolCall") continue;

  const result = { city: block.arguments.city, temperature: 27, sky: "clear" };
  context.messages.push({
    role: "toolResult",
    toolCallId: block.id,
    toolName: block.name,
    content: [{ type: "text", text: JSON.stringify(result) }],
    isError: false,
    timestamp: Date.now(),
  });
}

if (first.stopReason === "toolUse") {
  const final = await models.completeSimple(model, context);
  console.log(final.content);
}

await provider.dispose();
```

The provider exposes one zero-cost model:

| Provider | Model | API | Context | Output |
| --- | --- | --- | ---: | ---: |
| `needle` | `needle-2` | `needle-local` | 2,048 | 512 |

It emits the standard pi-ai event protocol: `start`, buffered thinking/text/tool-call content events, and `done` or `error`. Inference errors and aborts become normal pi-ai error messages rather than rejected streams.

## Install in pi

The package includes a pi extension manifest:

```bash
pi install npm:needle-pi-ai-provider
pi --provider needle --model needle-2
```

The first inference downloads and caches the official model. Configure the extension with environment variables:

```bash
# Use a local or tuned archive instead of downloading the base model
export NEEDLE_MODEL_PATH=/path/to/needle2.cact

# cpu (default), typegpu, vgpu, or auto
export NEEDLE_BACKEND=cpu

pi --provider needle --model needle-2
```

The extension registers the complete native pi-ai provider with `pi.registerProvider()`. It releases the loaded model on `session_shutdown`, including reload, session replacement, and quit.

Needle 2 has a 2,048-token context and is a compact tool-selection model, not a general-purpose coding LLM. For a practical demo, activate a small, clearly described tool set.

## Configuration

`createNeedlePiProvider()` accepts all `NeedleModel.load()` options plus:

- `model` / `disposeModel` — reuse a preloaded model and control ownership
- `agentOptions` — defaults for constrained tool calling
- `providerId`, `providerName`, `modelId`, `modelName` — catalogue identity overrides

Reasoning levels map to local reasoning-token caps:

| pi level | Tokens |
| --- | ---: |
| minimal | 32 |
| low | 64 |
| medium | 128 |
| high | 256 |
| xhigh | 384 |
| max | 512 |

Use `createNeedlePiApi()` when you need only the `ProviderStreams` implementation, or `createNeedlePiModel()` when composing a custom provider.
