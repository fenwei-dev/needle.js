import type { TgpuRoot } from "typegpu";
import type {
  AttentionBindingResources,
  KvBindingResources,
  QueryBindingResources,
  ResidentBindingFactory,
  ScoreBindingResources,
} from "../resident/bindings.js";
import { typeGpuQueryNormRope } from "./typegpu-kernels.js";
import {
  typeGpuAttentionLayout,
  typeGpuKvLayout,
  typeGpuQueryLayout,
  typeGpuScoresLayout,
} from "./typegpu-layouts.js";

export {
  typeGpuAttentionLayout,
  typeGpuKvLayout,
  typeGpuQueryLayout,
  typeGpuScoresLayout,
} from "./typegpu-layouts.js";

export function createTypeGpuBindingFactory(root: TgpuRoot): ResidentBindingFactory {
  const queryPipeline = root
    .createComputePipeline({ compute: typeGpuQueryNormRope })
    .$name("needle.resident.query-pipeline");
  return {
    queryLayout: root.unwrap(typeGpuQueryLayout),
    queryPipeline: root.unwrap(queryPipeline),
    kvLayout: root.unwrap(typeGpuKvLayout),
    scoresLayout: root.unwrap(typeGpuScoresLayout),
    attentionLayout: root.unwrap(typeGpuAttentionLayout),
    createQuery(resources: QueryBindingResources) {
      return root.unwrap(root.createBindGroup(typeGpuQueryLayout, resources));
    },
    createKv(resources: KvBindingResources) {
      return root.unwrap(root.createBindGroup(typeGpuKvLayout, resources));
    },
    createScores(resources: ScoreBindingResources) {
      return root.unwrap(root.createBindGroup(typeGpuScoresLayout, resources));
    },
    createAttention(resources: AttentionBindingResources) {
      return root.unwrap(root.createBindGroup(typeGpuAttentionLayout, resources));
    },
  };
}
