import type { TgpuRoot } from "typegpu";
import type {
  AttentionBindingResources,
  CqBindingResources,
  KvBindingResources,
  QueryBindingResources,
  ResidentBindingFactory,
  ScoreBindingResources,
} from "../resident/bindings.js";
import { typeGpuCqMatvec, typeGpuQueryNormRope } from "./typegpu-kernels.js";
import {
  typeGpuAttentionLayout,
  typeGpuCqLayout,
  typeGpuKvLayout,
  typeGpuQueryLayout,
  typeGpuScoresLayout,
} from "./typegpu-layouts.js";

export {
  typeGpuAttentionLayout,
  typeGpuCqLayout,
  typeGpuKvLayout,
  typeGpuQueryLayout,
  typeGpuScoresLayout,
} from "./typegpu-layouts.js";

export function createTypeGpuBindingFactory(root: TgpuRoot): ResidentBindingFactory {
  const cqPipeline = root
    .createComputePipeline({ compute: typeGpuCqMatvec })
    .$name("needle.resident.cq-pipeline");
  const queryPipeline = root
    .createComputePipeline({ compute: typeGpuQueryNormRope })
    .$name("needle.resident.query-pipeline");
  return {
    cqLayout: root.unwrap(typeGpuCqLayout),
    cqPipeline: root.unwrap(cqPipeline),
    queryLayout: root.unwrap(typeGpuQueryLayout),
    queryPipeline: root.unwrap(queryPipeline),
    kvLayout: root.unwrap(typeGpuKvLayout),
    scoresLayout: root.unwrap(typeGpuScoresLayout),
    attentionLayout: root.unwrap(typeGpuAttentionLayout),
    createCq(resources: CqBindingResources) {
      return root.unwrap(root.createBindGroup(typeGpuCqLayout, resources));
    },
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
