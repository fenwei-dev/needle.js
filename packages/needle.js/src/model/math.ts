import { fastWalshHadamard } from "../backends/cq.js";

export function sigmoid(value: number): number {
  if (value >= 0) {
    const exp = Math.exp(-value);
    return 1 / (1 + exp);
  }
  const exp = Math.exp(value);
  return exp / (1 + exp);
}

export function silu(value: number): number {
  return value * sigmoid(value);
}

export function rmsNorm(
  input: Float32Array,
  scale: Float32Array,
  output = new Float32Array(input.length),
  epsilon = 1e-6,
): Float32Array {
  let sumSquares = 0;
  for (let index = 0; index < input.length; index++) {
    const value = input[index] ?? 0;
    sumSquares += value * value;
  }
  const inverse = 1 / Math.sqrt(sumSquares / input.length + epsilon);
  for (let index = 0; index < input.length; index++) {
    output[index] = (1 + (scale[index] ?? 0)) * (input[index] ?? 0) * inverse;
  }
  return output;
}

export function rmsUnit(
  input: Float32Array,
  output = new Float32Array(input.length),
  epsilon = 1e-6,
): Float32Array {
  let sumSquares = 0;
  for (const value of input) sumSquares += value * value;
  const inverse = 1 / Math.sqrt(sumSquares / input.length + epsilon);
  for (let index = 0; index < input.length; index++) output[index] = (input[index] ?? 0) * inverse;
  return output;
}

export function softmaxInPlace(values: Float32Array, length = values.length): Float32Array {
  if (length === 0) return values;
  let maximum = values[0] ?? Number.NEGATIVE_INFINITY;
  for (let index = 1; index < length; index++) maximum = Math.max(maximum, values[index] ?? Number.NEGATIVE_INFINITY);
  let total = 0;
  for (let index = 0; index < length; index++) {
    const value = Math.exp((values[index] ?? 0) - maximum);
    values[index] = value;
    total += value;
  }
  const inverse = total > 0 ? 1 / total : 0;
  for (let index = 0; index < length; index++) values[index] = (values[index] ?? 0) * inverse;
  return values;
}

/** Twenty log-space row/column normalizations, matching Needle's mHC router. */
export function sinkhorn(logits: Float32Array, size: number, iterations = 20): Float32Array {
  const values = new Float64Array(logits);
  for (let iteration = 0; iteration < iterations; iteration++) {
    for (let row = 0; row < size; row++) {
      const offset = row * size;
      let maximum = values[offset] ?? Number.NEGATIVE_INFINITY;
      for (let column = 1; column < size; column++) maximum = Math.max(maximum, values[offset + column] ?? Number.NEGATIVE_INFINITY);
      let sum = 0;
      for (let column = 0; column < size; column++) sum += Math.exp((values[offset + column] ?? 0) - maximum);
      const logSum = Math.log(sum) + maximum;
      for (let column = 0; column < size; column++) values[offset + column] = (values[offset + column] ?? 0) - logSum;
    }
    for (let column = 0; column < size; column++) {
      let maximum = values[column] ?? Number.NEGATIVE_INFINITY;
      for (let row = 1; row < size; row++) maximum = Math.max(maximum, values[row * size + column] ?? Number.NEGATIVE_INFINITY);
      let sum = 0;
      for (let row = 0; row < size; row++) sum += Math.exp((values[row * size + column] ?? 0) - maximum);
      const logSum = Math.log(sum) + maximum;
      for (let row = 0; row < size; row++) values[row * size + column] = (values[row * size + column] ?? 0) - logSum;
    }
  }
  for (let index = 0; index < logits.length; index++) logits[index] = Math.exp(values[index] ?? 0);
  return logits;
}

export function applyRope(
  vector: Float32Array,
  headCount: number,
  headDimension: number,
  position: number,
  theta: number,
): void {
  const half = headDimension >>> 1;
  for (let head = 0; head < headCount; head++) {
    const offset = head * headDimension;
    for (let dimension = 0; dimension < half; dimension++) {
      const frequency = theta ** (-(2 * dimension) / headDimension);
      const angle = position * frequency;
      const cosine = Math.cos(angle);
      const sine = Math.sin(angle);
      const first = vector[offset + dimension] ?? 0;
      const second = vector[offset + dimension + half] ?? 0;
      vector[offset + dimension] = first * cosine - second * sine;
      vector[offset + dimension + half] = second * cosine + first * sine;
    }
  }
}

export function hadamardMlp(
  input: Float32Array,
  d1: Float32Array,
  d2: Float32Array,
  d3: Float32Array,
  hadamardDimension: number,
): Float32Array {
  const values = new Float32Array(hadamardDimension);
  for (let index = 0; index < input.length; index++) values[index] = (input[index] ?? 0) * (d1[index] ?? 0);
  const inverseRoot = 1 / Math.sqrt(hadamardDimension);
  fastWalshHadamard(values);
  for (let index = 0; index < hadamardDimension; index++) {
    values[index] = silu((values[index] ?? 0) * inverseRoot * (d2[index] ?? 0));
  }
  fastWalshHadamard(values);
  const output = new Float32Array(input.length);
  for (let index = 0; index < output.length; index++) {
    output[index] = (values[index] ?? 0) * inverseRoot * (d3[index] ?? 0);
  }
  return output;
}

export function argmax(values: Float32Array): number {
  let best = 0;
  let bestValue = values[0] ?? Number.NEGATIVE_INFINITY;
  for (let index = 1; index < values.length; index++) {
    const value = values[index] ?? Number.NEGATIVE_INFINITY;
    if (value > bestValue) {
      bestValue = value;
      best = index;
    }
  }
  return best;
}

export function logSoftmaxAt(logits: Float32Array, token: number): number {
  let maximum = logits[0] ?? Number.NEGATIVE_INFINITY;
  for (let index = 1; index < logits.length; index++) maximum = Math.max(maximum, logits[index] ?? Number.NEGATIVE_INFINITY);
  let sum = 0;
  for (const logit of logits) sum += Math.exp(logit - maximum);
  return (logits[token] ?? Number.NEGATIVE_INFINITY) - maximum - Math.log(sum);
}
