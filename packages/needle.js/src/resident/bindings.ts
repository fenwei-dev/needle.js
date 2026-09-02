/// <reference types="@webgpu/types" preserve="true" />

export interface CqBindingResources {
  readonly packed: GPUBuffer;
  readonly norms: GPUBuffer;
  readonly input: GPUBuffer;
  readonly codebook: GPUBuffer;
  readonly params: GPUBuffer;
  readonly output: GPUBuffer;
}

export interface QueryBindingResources {
  readonly query: GPUBuffer;
  readonly normScale: GPUBuffer;
  readonly params: GPUBuffer;
}

export interface KvBindingResources {
  readonly key: GPUBuffer;
  readonly value: GPUBuffer;
  readonly keyNorm: GPUBuffer;
  readonly keyCache: GPUBuffer;
  readonly valueCache: GPUBuffer;
  readonly keyScales: GPUBuffer;
  readonly valueScales: GPUBuffer;
  readonly params: GPUBuffer;
}

export interface ScoreBindingResources {
  readonly query: GPUBuffer;
  readonly keyCache: GPUBuffer;
  readonly keyScales: GPUBuffer;
  readonly params: GPUBuffer;
  readonly scores: GPUBuffer;
}

export interface AttentionBindingResources {
  readonly gate: GPUBuffer;
  readonly valueCache: GPUBuffer;
  readonly valueScales: GPUBuffer;
  readonly params: GPUBuffer;
  readonly scores: GPUBuffer;
  readonly output: GPUBuffer;
}

export interface ConfidencePoolBindingResources {
  readonly lanes: GPUBuffer;
  readonly probes: GPUBuffer;
  readonly maxima: GPUBuffer;
  readonly denominators: GPUBuffer;
  readonly weighted: GPUBuffer;
}

export interface ConfidenceHeadBindingResources {
  readonly denominators: GPUBuffer;
  readonly weighted: GPUBuffer;
  readonly projection: GPUBuffer;
  readonly bias: GPUBuffer;
  readonly result: GPUBuffer;
}

export interface SelectionBindingResources {
  readonly logits: GPUBuffer;
  readonly allowed: GPUBuffer;
  readonly params: GPUBuffer;
  readonly result: GPUBuffer;
}

export interface ResidentBindingFactory {
  readonly cqLayout?: GPUBindGroupLayout;
  readonly cqPipeline?: GPUComputePipeline;
  readonly queryLayout?: GPUBindGroupLayout;
  readonly queryPipeline?: GPUComputePipeline;
  readonly kvLayout?: GPUBindGroupLayout;
  readonly scoresLayout?: GPUBindGroupLayout;
  readonly attentionLayout?: GPUBindGroupLayout;
  readonly confidencePoolLayout?: GPUBindGroupLayout;
  readonly confidenceHeadLayout?: GPUBindGroupLayout;
  readonly selectionLayout?: GPUBindGroupLayout;
  createCq?(resources: CqBindingResources): GPUBindGroup;
  createQuery?(resources: QueryBindingResources): GPUBindGroup;
  createKv?(resources: KvBindingResources): GPUBindGroup;
  createScores?(resources: ScoreBindingResources): GPUBindGroup;
  createAttention?(resources: AttentionBindingResources): GPUBindGroup;
  createConfidencePool?(resources: ConfidencePoolBindingResources): GPUBindGroup;
  createConfidenceHead?(resources: ConfidenceHeadBindingResources): GPUBindGroup;
  createSelection?(resources: SelectionBindingResources): GPUBindGroup;
}
