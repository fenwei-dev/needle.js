import { expect, test } from "bun:test";
import { tgpu } from "typegpu";
import * as d from "typegpu/data";
import { typeGpuCqLayout, typeGpuQueryLayout } from "../src/backends/typegpu-bindings.js";
import { typeGpuCqMatvec, typeGpuQueryNormRope } from "../src/backends/typegpu-kernels.js";
import { typeGpuParameterSchemas } from "../src/backends/typegpu-parameters.js";
import { typeGpuArraySchema } from "../src/backends/typegpu-resources.js";

test("TypeGPU resident parameter schemas preserve WGSL byte layouts", () => {
  expect(d.sizeOf(typeGpuParameterSchemas.query)).toBe(8);
  expect(d.sizeOf(typeGpuParameterSchemas.kv)).toBe(32);
  expect(d.sizeOf(typeGpuParameterSchemas.attention)).toBe(32);
  expect(d.sizeOf(typeGpuParameterSchemas.projection)).toBe(32);
  expect(d.sizeOf(typeGpuArraySchema("f32", 512))).toBe(2048);
  expect(d.sizeOf(typeGpuArraySchema("i32", 64))).toBe(256);
  expect(Object.keys(typeGpuQueryLayout.entries)).toEqual(["query", "normScale", "params"]);
  expect(typeGpuQueryLayout.entries.query.access).toBe("mutable");
  expect(typeGpuQueryLayout.entries.params.access).toBe("readonly");
  expect(Object.keys(typeGpuCqLayout.entries)).toEqual([
    "packed",
    "norms",
    "input",
    "codebook",
    "params",
    "output",
  ]);
  expect(typeGpuCqLayout.entries.output.access).toBe("mutable");

  const cqWgsl = tgpu.resolve([typeGpuCqMatvec]);
  expect(cqWgsl).toContain("needle_cq_packed_index");
  expect(cqWgsl).toContain("@compute @workgroup_size(32)");
  expect(cqWgsl).not.toContain(".$");

  const queryWgsl = tgpu.resolve([typeGpuQueryNormRope]);
  expect(queryWgsl).toContain("@compute @workgroup_size(64)");
  expect(queryWgsl).toContain("@group(0) @binding(0) var<storage, read_write> query");
  expect(queryWgsl).toContain("struct item");
  expect(queryWgsl).not.toContain(".$");
});
