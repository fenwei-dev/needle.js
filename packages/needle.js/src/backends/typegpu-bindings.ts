import { type TgpuRoot, tgpu } from "typegpu";
import * as d from "typegpu/data";
import type { QueryBindingResources, ResidentBindingFactory } from "../resident/bindings.js";
import { typeGpuParameterSchemas } from "./typegpu-parameters.js";

export const typeGpuQueryLayout = tgpu
  .bindGroupLayout({
    query: {
      storage: d.arrayOf(d.f32, 512),
      access: "mutable",
      visibility: ["compute"],
    },
    normScale: {
      storage: d.arrayOf(d.f32, 64),
      access: "readonly",
      visibility: ["compute"],
    },
    params: {
      storage: typeGpuParameterSchemas.query,
      access: "readonly",
      visibility: ["compute"],
    },
  })
  .$idx(0)
  .$name("needle.resident.query-layout");

export function createTypeGpuBindingFactory(root: TgpuRoot): ResidentBindingFactory {
  return {
    queryLayout: root.unwrap(typeGpuQueryLayout),
    createQuery(resources: QueryBindingResources) {
      const group = root.createBindGroup(typeGpuQueryLayout, {
        query: resources.query,
        normScale: resources.normScale,
        params: resources.params,
      });
      return root.unwrap(group);
    },
  };
}
