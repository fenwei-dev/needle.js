import type { TgpuRoot } from "typegpu";
import * as d from "typegpu/data";
import type { ResidentResourceFactory, ResidentScalar } from "../resident/resources.js";

export function typeGpuArraySchema(scalar: ResidentScalar, elementCount: number) {
  if (scalar === "f32") return d.arrayOf(d.f32, elementCount);
  if (scalar === "i32") return d.arrayOf(d.i32, elementCount);
  return d.arrayOf(d.u32, elementCount);
}

export function createTypeGpuResourceFactory(root: TgpuRoot): ResidentResourceFactory {
  return {
    create(scalar, label, elementCount, extraUsage = 0) {
      const buffer = root
        .createBuffer(typeGpuArraySchema(scalar, elementCount))
        .$usage("storage")
        .$addFlags(extraUsage)
        .$name(label);
      return root.unwrap(buffer);
    },
  };
}
