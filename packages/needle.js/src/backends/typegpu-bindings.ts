import { type TgpuRoot, tgpu } from "typegpu";
import * as d from "typegpu/data";
import type {
  AttentionBindingResources,
  KvBindingResources,
  QueryBindingResources,
  ResidentBindingFactory,
  ScoreBindingResources,
} from "../resident/bindings.js";
import { typeGpuParameterSchemas } from "./typegpu-parameters.js";

const compute: ["compute"] = ["compute"];

export const typeGpuQueryLayout = tgpu
  .bindGroupLayout({
    query: { storage: d.arrayOf(d.f32, 512), access: "mutable", visibility: compute },
    normScale: { storage: d.arrayOf(d.f32, 64), access: "readonly", visibility: compute },
    params: {
      storage: typeGpuParameterSchemas.query,
      access: "readonly",
      visibility: compute,
    },
  })
  .$idx(0)
  .$name("needle.resident.query-layout");

export const typeGpuKvLayout = tgpu
  .bindGroupLayout({
    key: { storage: d.arrayOf(d.f32, 256), access: "readonly", visibility: compute },
    value: { storage: d.arrayOf(d.f32, 256), access: "readonly", visibility: compute },
    keyNorm: { storage: d.arrayOf(d.f32, 64), access: "readonly", visibility: compute },
    keyCache: {
      storage: (n: number) => d.arrayOf(d.i32, n),
      access: "mutable",
      visibility: compute,
    },
    valueCache: {
      storage: (n: number) => d.arrayOf(d.i32, n),
      access: "mutable",
      visibility: compute,
    },
    keyScales: {
      storage: (n: number) => d.arrayOf(d.f32, n),
      access: "mutable",
      visibility: compute,
    },
    valueScales: {
      storage: (n: number) => d.arrayOf(d.f32, n),
      access: "mutable",
      visibility: compute,
    },
    params: { storage: typeGpuParameterSchemas.kv, access: "readonly", visibility: compute },
  })
  .$idx(0)
  .$name("needle.resident.kv-layout");

export const typeGpuScoresLayout = tgpu
  .bindGroupLayout({
    query: { storage: d.arrayOf(d.f32, 512), access: "readonly", visibility: compute },
    keyCache: {
      storage: (n: number) => d.arrayOf(d.i32, n),
      access: "readonly",
      visibility: compute,
    },
    keyScales: {
      storage: (n: number) => d.arrayOf(d.f32, n),
      access: "readonly",
      visibility: compute,
    },
    params: {
      storage: typeGpuParameterSchemas.attention,
      access: "readonly",
      visibility: compute,
    },
    scores: { storage: d.arrayOf(d.f32, 4096), access: "mutable", visibility: compute },
  })
  .$idx(0)
  .$name("needle.resident.scores-layout");

export const typeGpuAttentionLayout = tgpu
  .bindGroupLayout({
    gate: { storage: d.arrayOf(d.f32, 512), access: "readonly", visibility: compute },
    valueCache: {
      storage: (n: number) => d.arrayOf(d.i32, n),
      access: "readonly",
      visibility: compute,
    },
    valueScales: {
      storage: (n: number) => d.arrayOf(d.f32, n),
      access: "readonly",
      visibility: compute,
    },
    params: {
      storage: typeGpuParameterSchemas.attention,
      access: "readonly",
      visibility: compute,
    },
    scores: { storage: d.arrayOf(d.f32, 4096), access: "readonly", visibility: compute },
    output: { storage: d.arrayOf(d.f32, 512), access: "mutable", visibility: compute },
  })
  .$idx(0)
  .$name("needle.resident.attention-layout");

export function createTypeGpuBindingFactory(root: TgpuRoot): ResidentBindingFactory {
  return {
    queryLayout: root.unwrap(typeGpuQueryLayout),
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
