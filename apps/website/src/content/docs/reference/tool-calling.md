---
title: Tool-calling reference
description: Supported schemas, constrained decoding, retrieval, sessions, and confidence.
---

## Recommended shape

Needle performs best when tools are concrete and mutually distinguishable:

```ts
const setLight = defineTool({
  name: "set_light",
  description: "Set one room light on or off and choose brightness",
  parameters: {
    type: "object",
    properties: {
      room: {
        type: "string",
        enum: ["kitchen", "study", "bedroom"],
      },
      on: { type: "boolean" },
      brightness: {
        type: "integer",
        minimum: 0,
        maximum: 100,
      },
    },
    required: ["room", "on"],
  },
  execute: async (arguments_) => updateLight(arguments_),
});
```

Prefer separate `turn_light_on` / `turn_light_off` tools when a boolean distinction proves unreliable. Prefer enums over unconstrained strings whenever the domain is closed.

## Supported JSON Schema features

The incremental grammar supports:

- strings with `enum`, `const`, length, pattern, email, and UUID constraints
- numbers and integers with inclusive/exclusive bounds and `multipleOf`
- booleans and null
- arrays with item schemas, length bounds, and uniqueness
- nested objects, declared properties, required keys, and property-count bounds
- local `$ref`, `$defs`, `definitions`, `oneOf`, and `anyOf` for common disjoint alternatives

Unknown object keys, duplicate keys, invalid enum values, malformed JSON, missing required values, and repeated tools become unreachable during decoding.

## Continuous byte grammar

For every output step, `needle.js`:

1. clones the current grammar state for each normal or byte-fallback token;
2. feeds every decoded byte in that token through the clone;
3. rejects candidates that cannot remain valid;
4. selects the highest-logit surviving token;
5. commits the winning grammar state.

A token can cross several structural boundaries, such as `},{"name":"`. This preserves the model's natural tokenization instead of teacher-forcing JSON fragments in separate passes.

## Multiple calls

A turn returns an array:

```json
[
  { "name": "create_event", "arguments": { "title": "Demo" } },
  { "name": "send_email", "arguments": { "to": "ada@example.com" } }
]
```

One tool cannot repeat within the same turn. `maxCallsPerTurn` defaults to four.

## Tool retrieval

Large catalogues are ranked with BM25 over names and descriptions. Only the highest-ranked tools that fit `toolTokenBudget` enter the prompt and grammar.

```ts
const agent = await Needle.create({
  model,
  tools: largeCatalogue,
  toolTokenBudget: 180,
  maximumRetrievedTools: 5,
});
```

An unselected tool is impossible to call, not merely less likely.

## Stateful calls

`complete()` retains the conversation transcript. After executing calls, pass a JSON-serializable result:

```ts
let response = await agent.complete("Find Ada and send a message");

const results = await Promise.all(
  response.functionCalls.map(executeCall),
);

response = await agent.complete(JSON.stringify(results));
```

`run()` automates this loop. `reset()` starts a new conversation while retaining the loaded model and tool catalogue.

## Confidence

For the base archive, confidence combines online probe-head pooling with constrained token probability. Pick a threshold from validation data for your own commands. Treat low confidence as a request for clarification or escalation, not as permission to guess.
