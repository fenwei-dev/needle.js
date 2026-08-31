import { expect, test } from "bun:test";
import { Needle, NeedleModel } from "../src/index.js";

const modelPath = process.env.NEEDLE_MODEL_PATH;
const integrationTest = modelPath ? test : test.skip;

integrationTest(
  "official Needle 2 archive produces a constrained flashlight call",
  async () => {
    if (!modelPath) throw new Error("NEEDLE_MODEL_PATH is required for this integration test");
    const model = await NeedleModel.load({ weights: modelPath, backend: "cpu" });
    expect(model.weights.geometry).toMatchObject({
      vocabularySize: 8192,
      modelDimension: 512,
      numberOfLayers: 27,
      kvWindow: 256,
    });
    const agent = new Needle(model, {
      tools: [
        {
          name: "turn_on_flashlight",
          description: "Turn on the flashlight",
          parameters: { type: "object", properties: {} },
        },
      ],
    });
    try {
      const response = await agent.complete("Turn on the flashlight.");
      expect(response.rawCall).toBe('[{"name":"turn_on_flashlight","arguments":{}}]');
      expect(response.functionCalls).toEqual([
        {
          id: "call_1",
          name: "turn_on_flashlight",
          arguments: {},
        },
      ]);
    } finally {
      await agent.dispose();
    }
  },
  30_000,
);
