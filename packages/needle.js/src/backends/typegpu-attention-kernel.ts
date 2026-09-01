export const QUERY_NORM_ROPE_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> query: array<f32>;
@group(0) @binding(1) var<storage, read> norm_scale: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;

var<workgroup> query_values: array<f32, 64>;
var<workgroup> query_partial: array<f32, 64>;

@compute @workgroup_size(64)
fn query_norm_rope(
  @builtin(workgroup_id) group: vec3u,
  @builtin(local_invocation_id) local: vec3u,
) {
  let head = group.x;
  let index = head * 64u + local.x;
  let value = query[index];
  query_partial[local.x] = value * value;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (local.x < stride) {
      query_partial[local.x] = query_partial[local.x] + query_partial[local.x + stride];
    }
    workgroupBarrier();
  }
  let inverse = inverseSqrt(query_partial[0] / 64.0 + 1e-6);
  query_values[local.x] = (1.0 + norm_scale[local.x]) * value * inverse;
  workgroupBarrier();
  if (local.x < 32u) {
    let theta = bitcast<f32>(params[1]);
    let frequency = pow(theta, -f32(2u * local.x) / 64.0);
    let angle = f32(params[0]) * frequency;
    let cosine = cos(angle);
    let sine = sin(angle);
    let first = query_values[local.x];
    let second = query_values[local.x + 32u];
    query[head * 64u + local.x] = first * cosine - second * sine;
    query[head * 64u + local.x + 32u] = second * cosine + first * sine;
  }
}
`;

export const KV_NORM_ROPE_STORE_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> key_input: array<f32>;
@group(0) @binding(1) var<storage, read> value_input: array<f32>;
@group(0) @binding(2) var<storage, read> norm_scale: array<f32>;
@group(0) @binding(3) var<storage, read_write> key_cache: array<i32>;
@group(0) @binding(4) var<storage, read_write> value_cache: array<i32>;
@group(0) @binding(5) var<storage, read_write> key_scales: array<f32>;
@group(0) @binding(6) var<storage, read_write> value_scales: array<f32>;
@group(0) @binding(7) var<storage, read> params: array<u32>;

var<workgroup> key_values: array<f32, 64>;
var<workgroup> key_partial: array<f32, 64>;
var<workgroup> value_partial: array<f32, 64>;

fn cache_slot(position: u32, sink_length: u32, window: u32) -> u32 {
  if (position < sink_length) { return position; }
  return sink_length + ((position - sink_length) % window);
}

@compute @workgroup_size(64)
fn kv_norm_rope_store(
  @builtin(workgroup_id) group: vec3u,
  @builtin(local_invocation_id) local: vec3u,
) {
  let head = group.x;
  let vector_index = head * 64u + local.x;
  let raw_key = key_input[vector_index];
  let raw_value = value_input[vector_index];
  key_partial[local.x] = raw_key * raw_key;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (local.x < stride) {
      key_partial[local.x] = key_partial[local.x] + key_partial[local.x + stride];
    }
    workgroupBarrier();
  }
  let inverse = inverseSqrt(key_partial[0] / 64.0 + 1e-6);
  key_values[local.x] = (1.0 + norm_scale[local.x]) * raw_key * inverse;
  workgroupBarrier();
  if (local.x < 32u) {
    let theta = bitcast<f32>(params[6]);
    let frequency = pow(theta, -f32(2u * local.x) / 64.0);
    let angle = f32(params[1]) * frequency;
    let cosine = cos(angle);
    let sine = sin(angle);
    let first = key_values[local.x];
    let second = key_values[local.x + 32u];
    key_values[local.x] = first * cosine - second * sine;
    key_values[local.x + 32u] = second * cosine + first * sine;
  }
  workgroupBarrier();

  key_partial[local.x] = abs(key_values[local.x]);
  value_partial[local.x] = abs(raw_value);
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (local.x < stride) {
      key_partial[local.x] = max(key_partial[local.x], key_partial[local.x + stride]);
      value_partial[local.x] = max(value_partial[local.x], value_partial[local.x + stride]);
    }
    workgroupBarrier();
  }

  let layer = params[0];
  let position = params[1];
  let allocation = params[2];
  let sink_length = params[3];
  let window = params[4];
  let kv_heads = params[5];
  let slot = cache_slot(position, sink_length, window);
  let scale_index = (layer * kv_heads + head) * allocation + slot;
  let cache_index = scale_index * 64u + local.x;
  let key_scale = max(key_partial[0], 1e-12) / 127.0;
  let value_scale = max(value_partial[0], 1e-12) / 127.0;
  if (local.x == 0u) {
    key_scales[scale_index] = key_scale;
    value_scales[scale_index] = value_scale;
  }
  key_cache[cache_index] = i32(clamp(round(key_values[local.x] / key_scale), -127.0, 127.0));
  value_cache[cache_index] = i32(clamp(round(raw_value / value_scale), -127.0, 127.0));
}
`;

export const ATTENTION_SCORES_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> key_cache: array<i32>;
@group(0) @binding(2) var<storage, read> key_scales: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;
@group(0) @binding(4) var<storage, read_write> scores: array<f32>;

var<workgroup> score_partial: array<f32, 64>;

fn cache_slot(position: u32, sink_length: u32, window: u32) -> u32 {
  if (position < sink_length) { return position; }
  return sink_length + ((position - sink_length) % window);
}

fn logical_position(time: u32, prefix_count: u32, recent_low: u32) -> u32 {
  if (time < prefix_count) { return time; }
  return recent_low + time - prefix_count;
}

@compute @workgroup_size(64)
fn attention_scores(
  @builtin(workgroup_id) group: vec3u,
  @builtin(local_invocation_id) local: vec3u,
) {
  let query_head = group.x;
  let time = group.y;
  let layer = params[0];
  let position = params[1];
  let allocation = params[2];
  let sink_length = params[3];
  let window = params[4];
  let kv_heads = params[5];
  let heads = params[6];
  let repetitions = heads / kv_heads;
  let kv_head = query_head / repetitions;
  let prefix_count = min(sink_length, position + 1u);
  var recent_low = prefix_count;
  if (position + 1u > window) {
    recent_low = max(prefix_count, position + 1u - window);
  }
  let logical = logical_position(time, prefix_count, recent_low);
  let slot = cache_slot(logical, sink_length, window);
  let scale_index = (layer * kv_heads + kv_head) * allocation + slot;
  let cache_index = scale_index * 64u + local.x;
  let query_index = query_head * 64u + local.x;
  score_partial[local.x] =
    query[query_index] * f32(key_cache[cache_index]) * key_scales[scale_index];
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (local.x < stride) {
      score_partial[local.x] = score_partial[local.x] + score_partial[local.x + stride];
    }
    workgroupBarrier();
  }
  if (local.x == 0u) {
    scores[query_head * 512u + time] = score_partial[0] * 0.125;
  }
}
`;

export const ATTENTION_SOFTMAX_GATE_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> value_cache: array<i32>;
@group(0) @binding(2) var<storage, read> value_scales: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;
@group(0) @binding(4) var<storage, read> scores: array<f32>;
@group(0) @binding(5) var<storage, read_write> attention_output: array<f32>;

var<workgroup> softmax_partial: array<f32, 64>;

fn cache_slot(position: u32, sink_length: u32, window: u32) -> u32 {
  if (position < sink_length) { return position; }
  return sink_length + ((position - sink_length) % window);
}

fn logical_position(time: u32, prefix_count: u32, recent_low: u32) -> u32 {
  if (time < prefix_count) { return time; }
  return recent_low + time - prefix_count;
}

@compute @workgroup_size(64)
fn attention_softmax_gate(
  @builtin(workgroup_id) group: vec3u,
  @builtin(local_invocation_id) local: vec3u,
) {
  let query_head = group.x;
  let layer = params[0];
  let position = params[1];
  let allocation = params[2];
  let sink_length = params[3];
  let window = params[4];
  let kv_heads = params[5];
  let heads = params[6];
  let repetitions = heads / kv_heads;
  let kv_head = query_head / repetitions;
  let prefix_count = min(sink_length, position + 1u);
  var recent_low = prefix_count;
  if (position + 1u > window) {
    recent_low = max(prefix_count, position + 1u - window);
  }
  let count = prefix_count + position + 1u - recent_low;
  let score_offset = query_head * 512u;

  var local_maximum = -1e30;
  for (var time = local.x; time < count; time += 64u) {
    local_maximum = max(local_maximum, scores[score_offset + time]);
  }
  softmax_partial[local.x] = local_maximum;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (local.x < stride) {
      softmax_partial[local.x] = max(softmax_partial[local.x], softmax_partial[local.x + stride]);
    }
    workgroupBarrier();
  }
  let maximum = softmax_partial[0];
  var local_sum = 0.0;
  for (var time = local.x; time < count; time += 64u) {
    local_sum = local_sum + exp(scores[score_offset + time] - maximum);
  }
  softmax_partial[local.x] = local_sum;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (local.x < stride) {
      softmax_partial[local.x] = softmax_partial[local.x] + softmax_partial[local.x + stride];
    }
    workgroupBarrier();
  }
  let inverse_sum = 1.0 / max(softmax_partial[0], 1e-30);
  var value = 0.0;
  for (var time = 0u; time < count; time++) {
    let logical = logical_position(time, prefix_count, recent_low);
    let slot = cache_slot(logical, sink_length, window);
    let scale_index = (layer * kv_heads + kv_head) * allocation + slot;
    let cache_index = scale_index * 64u + local.x;
    let weight = exp(scores[score_offset + time] - maximum) * inverse_sum;
    value = value + weight * f32(value_cache[cache_index]) * value_scales[scale_index];
  }
  let output_index = query_head * 64u + local.x;
  let gate_value = 1.0 / (1.0 + exp(-gate[output_index]));
  attention_output[output_index] = value * gate_value;
}
`;

export const PREPARE_ATTENTION_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read_write> prepared: array<f32>;

var<workgroup> values: array<f32, 128>;

@compute @workgroup_size(128)
fn prepare_attention(
  @builtin(workgroup_id) group: vec3u,
  @builtin(local_invocation_id) local: vec3u,
) {
  let offset = group.x * 128u;
  values[local.x] = source[offset + local.x];
  workgroupBarrier();
  for (var stride = 1u; stride < 128u; stride <<= 1u) {
    if (local.x < 64u) {
      let block = (local.x / stride) * (stride * 2u);
      let inner = local.x % stride;
      let left_index = block + inner;
      let right_index = left_index + stride;
      let left = values[left_index];
      let right = values[right_index];
      values[left_index] = left + right;
      values[right_index] = left - right;
    }
    workgroupBarrier();
  }
  prepared[offset + local.x] = values[local.x] * 0.08838834764831845;
}
`;
