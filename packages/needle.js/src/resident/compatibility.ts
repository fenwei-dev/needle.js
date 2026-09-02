import type { ResidentExecutionResetOptions } from "../backends/backend.js";
import type { CactWeights } from "../model/cact.js";

export function supportsResidentExecution(
  weights: CactWeights,
  options: ResidentExecutionResetOptions,
): boolean {
  const geometry = weights.geometry;
  return (
    options.kvCache === "int8" &&
    geometry.modelDimension === 512 &&
    geometry.headDimension === 64 &&
    geometry.numberOfHeads === 8 &&
    geometry.numberOfKVHeads === 4 &&
    geometry.mhcLanes === 4 &&
    geometry.hadamardDimension === 512 &&
    geometry.kvWindow > 0 &&
    options.sinkLength + geometry.kvWindow <= 512 &&
    weights.layers.every(
      (layer) =>
        layer.queryProjection.inputSize === 512 &&
        layer.queryProjection.outputSize === 512 &&
        layer.keyProjection.inputSize === 512 &&
        layer.keyProjection.outputSize === 256 &&
        layer.keyProjection.groupSize === layer.queryProjection.groupSize &&
        layer.valueProjection.inputSize === 512 &&
        layer.valueProjection.outputSize === 256 &&
        layer.valueProjection.groupSize === layer.queryProjection.groupSize &&
        layer.gateProjection.inputSize === 512 &&
        layer.gateProjection.outputSize === 512 &&
        layer.gateProjection.groupSize === layer.queryProjection.groupSize &&
        layer.outputProjection.inputSize === 512 &&
        layer.outputProjection.outputSize === 512 &&
        layer.outputProjection.groupSize === 128 &&
        layer.hadamardD1.length === 512 &&
        layer.hadamardD2.length === 512 &&
        layer.hadamardD3.length === 512,
    )
  );
}

export function supportsResidentConfidence(weights: CactWeights): boolean {
  const confidence = weights.heads.get("confidence");
  return Boolean(
    confidence &&
      confidence.probeCount === 8 &&
      confidence.outputSize === 1 &&
      confidence.probes.length === 4096 &&
      confidence.projection.length === 4096,
  );
}
