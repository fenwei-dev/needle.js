/// <reference types="@webgpu/types" preserve="true" />

export interface QueryBindingResources {
  readonly query: GPUBuffer;
  readonly normScale: GPUBuffer;
  readonly params: GPUBuffer;
}

export interface ResidentBindingFactory {
  readonly queryLayout?: GPUBindGroupLayout;
  createQuery?(resources: QueryBindingResources): GPUBindGroup;
}
