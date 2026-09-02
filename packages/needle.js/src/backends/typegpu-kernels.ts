import { d, tgpu } from "typegpu";
import { typeGpuCqLayout, typeGpuQueryLayout } from "./typegpu-layouts.js";

const cqPartial = tgpu.workgroupVar(d.arrayOf(d.f32, 32)).$name("needle_cq_partial");

const cqPackedIndex = tgpu
  .fn(
    [d.u32, d.u32, d.u32, d.u32, d.u32],
    d.u32,
  )(`(row: u32, column: u32, recordBits: u32, rowBytes: u32, packedBase: u32) -> u32 {
    var bits = recordBits;
    if (recordBits == 5u) { bits = 2u; }
    let bitPosition = row * rowBytes * 8u + column * bits;
    let wordIndex = packedBase + (bitPosition >> 5u);
    let shift = bitPosition & 31u;
    var value = layout.$.packed[wordIndex] >> shift;
    if (shift + bits > 32u) {
      value = value | (layout.$.packed[wordIndex + 1u] << (32u - shift));
    }
    return value & ((1u << bits) - 1u);
  }`)
  .$uses({ layout: typeGpuCqLayout })
  .$name("needle_cq_packed_index");

const cqWeightValue = tgpu
  .fn(
    [d.u32, d.u32, d.u32],
    d.f32,
  )(`(index: u32, bits: u32, groupSize: u32) -> f32 {
    if (bits == 2u) { return layout.$.codebook[index]; }
    if (bits == 3u) { return layout.$.codebook[4u + index]; }
    if (bits == 4u) { return layout.$.codebook[12u + index]; }
    let centroid = 1.2240064 / sqrt(f32(groupSize));
    if (index == 3u) { return -centroid; }
    if (index == 1u) { return centroid; }
    return 0.0;
  }`)
  .$uses({ layout: typeGpuCqLayout })
  .$name("needle_cq_weight_value");

export const typeGpuCqMatvec = tgpu
  .computeFn({
    in: {
      local: d.builtin.localInvocationIndex,
      group: d.builtin.workgroupId,
    },
    workgroupSize: [32],
  })(`{
    let outputIndex = group.x;
    if (outputIndex >= layout.$.params.outputCount) { return; }
    let row = layout.$.params.rowStart + outputIndex;
    let groupSize = layout.$.params.groupSize;
    let bits = layout.$.params.bits;
    let rowBytes = layout.$.params.rowBytes;
    let groupCount = layout.$.params.groupCount;
    var total = 0.0;
    for (var weightGroup = 0u; weightGroup < groupCount; weightGroup++) {
      var dot = 0.0;
      let offset = weightGroup * groupSize;
      for (var column = local; column < groupSize; column += 32u) {
        let absoluteColumn = offset + column;
        let index = packedIndex(row, absoluteColumn, bits, rowBytes, 0u);
        dot = dot + weightValue(index, bits, groupSize) * layout.$.input[absoluteColumn];
      }
      partial[local] = dot;
      workgroupBarrier();
      for (var stride = 16u; stride > 0u; stride >>= 1u) {
        if (local < stride) {
          partial[local] = partial[local] + partial[local + stride];
        }
        workgroupBarrier();
      }
      if (local == 0u) {
        total = total + layout.$.norms[row * groupCount + weightGroup] * partial[0];
      }
      workgroupBarrier();
    }
    if (local == 0u) { layout.$.output[outputIndex] = total; }
  }`)
  .$uses({
    layout: typeGpuCqLayout,
    packedIndex: cqPackedIndex,
    weightValue: cqWeightValue,
    partial: cqPartial,
  })
  .$name("needle_cq_matvec");

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
