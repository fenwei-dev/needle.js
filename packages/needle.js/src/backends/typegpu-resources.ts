import type { TgpuRoot } from "typegpu";
import * as d from "typegpu/data";
import type { ResidentResourceFactory, ResidentScalar } from "../resident/resources.js";

export function typeGpuArraySchema(scalar: ResidentScalar, elementCount: number) {
  if (scalar === "f32") return d.arrayOf(d.f32, elementCount);
  if (scalar === "i32") return d.arrayOf(d.i32, elementCount);
  return d.arrayOf(d.u32, elementCount);
}

export function createTypeGpuResourceFactory(root: TgpuRoot): ResidentResourceFactory {
  const owned = new WeakMap<GPUBuffer, { destroy(): void }>();
  return {
    create(scalar, label, elementCount, extraUsage = 0) {
      const typed = root
        .createBuffer(typeGpuArraySchema(scalar, elementCount))
        .$usage("storage")
        .$addFlags(extraUsage)
        .$name(label);
      const buffer = root.unwrap(typed);
      owned.set(buffer, typed);
      return buffer;
    },
    destroy(buffer) {
      const typed = owned.get(buffer);
      if (!typed) return false;
      owned.delete(buffer);
      typed.destroy();
      return true;
    },
  };
}
