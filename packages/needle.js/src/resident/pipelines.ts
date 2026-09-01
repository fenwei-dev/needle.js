/// <reference types="@webgpu/types" preserve="true" />

import { CQ_MATVEC_WGSL } from "../backends/webgpu-kernel.js";
import { invariant } from "../errors.js";
import type { ResidentBindingFactory } from "./bindings.js";
import {
  ATTENTION_SCORES_WGSL,
  ATTENTION_SOFTMAX_GATE_WGSL,
  CONFIDENCE_HEAD_WGSL,
  CONFIDENCE_POOL_WGSL,
  ENGRAM_CONVOLVE_WGSL,
  ENGRAM_GATHER_WGSL,
  ENGRAM_INJECT_WGSL,
  FINAL_NORM_WGSL,
  HADAMARD_MLP_DELTA_WGSL,
  KV_NORM_ROPE_STORE_WGSL,
  MHC_PRE_WGSL,
  POST_MHC_ROUTING_WGSL,
  PREPARE_ATTENTION_WGSL,
  QUERY_NORM_ROPE_WGSL,
  RMS_LANES_WGSL,
  RMS_NORM_512_WGSL,
  SANDWICH_PREPARE_MLP_WGSL,
  SELECT_TOKEN_WGSL,
} from "./kernels.js";

export interface ResidentPipelines {
  readonly cq: GPUComputePipeline;
  readonly query: GPUComputePipeline;
  readonly kv: GPUComputePipeline;
  readonly scores: GPUComputePipeline;
  readonly attention: GPUComputePipeline;
  readonly prepare: GPUComputePipeline;
  readonly sandwich: GPUComputePipeline;
  readonly mlp: GPUComputePipeline;
  readonly routing: GPUComputePipeline;
  readonly rmsLanes: GPUComputePipeline;
  readonly mhcPre: GPUComputePipeline;
  readonly engram: GPUComputePipeline;
  readonly engramGather: GPUComputePipeline;
  readonly engramConvolve: GPUComputePipeline;
  readonly norm512: GPUComputePipeline;
  readonly finalNorm: GPUComputePipeline;
  readonly confidencePool: GPUComputePipeline;
  readonly confidenceHead: GPUComputePipeline;
  readonly selectToken: GPUComputePipeline;
}

export async function createResidentPipelines(
  device: GPUDevice,
  bindings: ResidentBindingFactory = {},
): Promise<ResidentPipelines> {
  const descriptors = [
    ["cq", CQ_MATVEC_WGSL, "main"],
    ["query", QUERY_NORM_ROPE_WGSL, "query_norm_rope"],
    ["kv", KV_NORM_ROPE_STORE_WGSL, "kv_norm_rope_store"],
    ["scores", ATTENTION_SCORES_WGSL, "attention_scores"],
    ["attention", ATTENTION_SOFTMAX_GATE_WGSL, "attention_softmax_gate"],
    ["prepare", PREPARE_ATTENTION_WGSL, "prepare_attention"],
    ["sandwich", SANDWICH_PREPARE_MLP_WGSL, "sandwich_prepare_mlp"],
    ["mlp", HADAMARD_MLP_DELTA_WGSL, "hadamard_mlp_delta"],
    ["routing", POST_MHC_ROUTING_WGSL, "post_mhc_routing"],
    ["rms-lanes", RMS_LANES_WGSL, "rms_lanes"],
    ["mhc-pre", MHC_PRE_WGSL, "mhc_pre"],
    ["engram", ENGRAM_INJECT_WGSL, "engram_inject"],
    ["engram-gather", ENGRAM_GATHER_WGSL, "engram_gather"],
    ["engram-convolve", ENGRAM_CONVOLVE_WGSL, "engram_convolve"],
    ["norm-512", RMS_NORM_512_WGSL, "rms_norm_512"],
    ["final-norm", FINAL_NORM_WGSL, "final_norm"],
    ["confidence-pool", CONFIDENCE_POOL_WGSL, "confidence_pool"],
    ["confidence-head", CONFIDENCE_HEAD_WGSL, "confidence_head"],
    ["select-token", SELECT_TOKEN_WGSL, "select_token"],
  ] as const;
  const pipelines = await Promise.all(
    descriptors.map(async ([label, source, entryPoint]) => {
      const module = device.createShaderModule({
        label: `needle.resident.${label}.shader`,
        code: source,
      });
      const explicitLayout =
        label === "query" && bindings.queryLayout
          ? device.createPipelineLayout({ bindGroupLayouts: [bindings.queryLayout] })
          : undefined;
      const descriptor: GPUComputePipelineDescriptor = {
        label: `needle.resident.${label}.pipeline`,
        layout: explicitLayout ?? "auto",
        compute: { module, entryPoint },
      };
      return device.createComputePipelineAsync
        ? device.createComputePipelineAsync(descriptor)
        : device.createComputePipeline(descriptor);
    }),
  );
  const [
    cq,
    query,
    kv,
    scores,
    attention,
    prepare,
    sandwich,
    mlp,
    routing,
    rmsLanes,
    mhcPre,
    engram,
    engramGather,
    engramConvolve,
    norm512,
    finalNorm,
    confidencePool,
    confidenceHead,
    selectToken,
  ] = pipelines;
  invariant(
    cq &&
      query &&
      kv &&
      scores &&
      attention &&
      prepare &&
      sandwich &&
      mlp &&
      routing &&
      rmsLanes &&
      mhcPre &&
      engram &&
      engramGather &&
      engramConvolve &&
      norm512 &&
      finalNorm &&
      confidencePool &&
      confidenceHead &&
      selectToken,
    "WEBGPU_UNAVAILABLE",
    "Resident pipeline compilation failed",
  );
  return {
    cq,
    query,
    kv,
    scores,
    attention,
    prepare,
    sandwich,
    mlp,
    routing,
    rmsLanes,
    mhcPre,
    engram,
    engramGather,
    engramConvolve,
    norm512,
    finalNorm,
    confidencePool,
    confidenceHead,
    selectToken,
  };
}
