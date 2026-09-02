import { d, tgpu } from "typegpu";
import { typeGpuQueryLayout } from "./typegpu-layouts.js";

const queryValues = tgpu.workgroupVar(d.arrayOf(d.f32, 64)).$name("needle_query_values");
const queryPartial = tgpu.workgroupVar(d.arrayOf(d.f32, 64)).$name("needle_query_partial");

/**
 * TypeGPU-owned query normalization pipeline. The body stays in WGSL so the
 * published package needs no source-transform plugin; layouts, dependencies,
 * workgroup memory, pipeline creation, and resource binding are TypeGPU-owned.
 */
export const typeGpuQueryNormRope = tgpu
  .computeFn({
    in: {
      local: d.builtin.localInvocationIndex,
      group: d.builtin.workgroupId,
    },
    workgroupSize: [64],
  })(`{
    let head = group.x;
    let index = head * 64u + local;
    let value = layout.$.query[index];
    queryPartial[local] = value * value;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
      if (local < stride) {
        queryPartial[local] = queryPartial[local] + queryPartial[local + stride];
      }
      workgroupBarrier();
    }
    let inverse = inverseSqrt(queryPartial[0] / 64.0 + 1e-6);
    queryValues[local] = (1.0 + layout.$.normScale[local]) * value * inverse;
    workgroupBarrier();
    if (local < 32u) {
      let theta = bitcast<f32>(layout.$.params.thetaBits);
      let frequency = pow(theta, -f32(2u * local) / 64.0);
      let angle = f32(layout.$.params.position) * frequency;
      let cosine = cos(angle);
      let sine = sin(angle);
      let first = queryValues[local];
      let second = queryValues[local + 32u];
      layout.$.query[head * 64u + local] = first * cosine - second * sine;
      layout.$.query[head * 64u + local + 32u] = second * cosine + first * sine;
    }
  }`)
  .$uses({ layout: typeGpuQueryLayout, queryValues, queryPartial })
  .$name("needle_query_norm_rope");
