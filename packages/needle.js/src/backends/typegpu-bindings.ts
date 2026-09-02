import type { TgpuRoot } from "typegpu";
import type {
  AttentionBindingResources,
  ConfidenceHeadBindingResources,
  ConfidencePoolBindingResources,
  CqBindingResources,
  KvBindingResources,
  QueryBindingResources,
  ResidentBindingFactory,
  ScoreBindingResources,
  SelectionBindingResources,
} from "../resident/bindings.js";
import { typeGpuCqMatvec, typeGpuQueryNormRope } from "./typegpu-kernels.js";
import {
  typeGpuAttentionLayout,
  typeGpuConfidenceHeadLayout,
  typeGpuConfidencePoolLayout,
  typeGpuCqLayout,
  typeGpuKvLayout,
  typeGpuQueryLayout,
  typeGpuScoresLayout,
  typeGpuSelectionLayout,
} from "./typegpu-layouts.js";

export {
  typeGpuAttentionLayout,
  typeGpuConfidenceHeadLayout,
  typeGpuConfidencePoolLayout,
  typeGpuCqLayout,
  typeGpuKvLayout,
  typeGpuQueryLayout,
  typeGpuScoresLayout,
  typeGpuSelectionLayout,
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
    confidencePoolLayout: root.unwrap(typeGpuConfidencePoolLayout),
    confidenceHeadLayout: root.unwrap(typeGpuConfidenceHeadLayout),
    selectionLayout: root.unwrap(typeGpuSelectionLayout),
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
    createConfidencePool(resources: ConfidencePoolBindingResources) {
      return root.unwrap(root.createBindGroup(typeGpuConfidencePoolLayout, resources));
    },
    createConfidenceHead(resources: ConfidenceHeadBindingResources) {
      return root.unwrap(root.createBindGroup(typeGpuConfidenceHeadLayout, resources));
    },
    createSelection(resources: SelectionBindingResources) {
      return root.unwrap(root.createBindGroup(typeGpuSelectionLayout, resources));
    },
  };
}
