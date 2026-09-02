import { tgpu } from "typegpu";
import * as d from "typegpu/data";
import { typeGpuParameterSchemas } from "./typegpu-parameters.js";

const compute: ["compute"] = ["compute"];

export const typeGpuCqLayout = tgpu
  .bindGroupLayout({
    packed: {
      storage: (n: number) => d.arrayOf(d.u32, n),
      access: "readonly",
      visibility: compute,
    },
    norms: {
      storage: (n: number) => d.arrayOf(d.f32, n),
      access: "readonly",
      visibility: compute,
    },
    input: {
      storage: (n: number) => d.arrayOf(d.f32, n),
      access: "readonly",
      visibility: compute,
    },
    codebook: { storage: d.arrayOf(d.f32, 28), access: "readonly", visibility: compute },
    params: {
      storage: typeGpuParameterSchemas.projection,
      access: "readonly",
      visibility: compute,
    },
    output: {
      storage: (n: number) => d.arrayOf(d.f32, n),
      access: "mutable",
      visibility: compute,
    },
  })
  .$idx(0)
  .$name("needle.resident.cq-layout");

export const typeGpuConfidencePoolLayout = tgpu
  .bindGroupLayout({
    lanes: { storage: d.arrayOf(d.f32, 2048), access: "readonly", visibility: compute },
    probes: { storage: d.arrayOf(d.f32, 4096), access: "readonly", visibility: compute },
    maxima: { storage: d.arrayOf(d.f32, 8), access: "mutable", visibility: compute },
    denominators: { storage: d.arrayOf(d.f32, 8), access: "mutable", visibility: compute },
    weighted: { storage: d.arrayOf(d.f32, 4096), access: "mutable", visibility: compute },
  })
  .$idx(0)
  .$name("needle.resident.confidence-pool-layout");

export const typeGpuConfidenceHeadLayout = tgpu
  .bindGroupLayout({
    denominators: { storage: d.arrayOf(d.f32, 8), access: "readonly", visibility: compute },
    weighted: { storage: d.arrayOf(d.f32, 4096), access: "readonly", visibility: compute },
    projection: { storage: d.arrayOf(d.f32, 4096), access: "readonly", visibility: compute },
    bias: { storage: d.arrayOf(d.f32, 1), access: "readonly", visibility: compute },
    result: { storage: d.arrayOf(d.f32, 1), access: "mutable", visibility: compute },
  })
  .$idx(0)
  .$name("needle.resident.confidence-head-layout");

export const typeGpuSelectionLayout = tgpu
  .bindGroupLayout({
    logits: { storage: d.arrayOf(d.f32, 8192), access: "readonly", visibility: compute },
    allowed: { storage: d.arrayOf(d.u32, 8192), access: "readonly", visibility: compute },
    params: { storage: d.arrayOf(d.u32, 2), access: "readonly", visibility: compute },
    result: { storage: d.arrayOf(d.u32, 2), access: "mutable", visibility: compute },
  })
  .$idx(0)
  .$name("needle.resident.selection-layout");

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
