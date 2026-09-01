import { expect, test } from "bun:test";
import * as d from "typegpu/data";
import { typeGpuParameterSchemas } from "../src/backends/typegpu-parameters.js";

test("TypeGPU resident parameter schemas preserve WGSL byte layouts", () => {
  expect(d.sizeOf(typeGpuParameterSchemas.query)).toBe(8);
  expect(d.sizeOf(typeGpuParameterSchemas.kv)).toBe(32);
  expect(d.sizeOf(typeGpuParameterSchemas.attention)).toBe(32);
});
