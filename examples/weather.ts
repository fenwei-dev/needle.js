import { createNeedle, defineTool } from "../src/index.js";

const getWeather = defineTool<{ city: string }, { city: string; tempC: number }>({
  name: "get_weather",
  description: "Get the current weather for a city",
  parameters: {
    type: "object",
    properties: {
      city: { type: "string", description: "City name" },
    },
    required: ["city"],
  },
  execute: ({ city }) => ({ city, tempC: 27 }),
});

const agent = await createNeedle({
  weights: process.env.NEEDLE_MODEL_PATH ?? "download",
  backend: "cpu",
  tools: [getWeather],
});

try {
  console.dir(await agent.run("What's the weather in Lagos right now?"), {
    depth: null,
  });
} finally {
  await agent.dispose();
}
