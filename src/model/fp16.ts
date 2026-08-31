const F32 = new Float32Array(1);
const U32 = new Uint32Array(F32.buffer);

/** Converts an IEEE-754 binary16 bit pattern to a JavaScript number. */
export function float16ToNumber(value: number): number {
  const h = value & 0xffff;
  const sign = (h & 0x8000) << 16;
  let exponent = (h >>> 10) & 0x1f;
  let mantissa = h & 0x03ff;
  let bits: number;

  if (exponent === 0) {
    if (mantissa === 0) {
      bits = sign;
    } else {
      exponent = 127 - 15 + 1;
      while ((mantissa & 0x0400) === 0) {
        mantissa <<= 1;
        exponent--;
      }
      mantissa &= 0x03ff;
      bits = sign | (exponent << 23) | (mantissa << 13);
    }
  } else if (exponent === 0x1f) {
    bits = sign | 0x7f80_0000 | (mantissa << 13);
  } else {
    bits = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
  }

  U32[0] = bits >>> 0;
  return F32[0] ?? 0;
}

/** Converts a JavaScript number to the nearest IEEE-754 binary16 bit pattern. */
export function numberToFloat16(value: number): number {
  F32[0] = value;
  const bits = U32[0] ?? 0;
  const sign = (bits >>> 16) & 0x8000;
  const exponent = (bits >>> 23) & 0xff;
  const mantissa = bits & 0x7f_ffff;

  if (exponent === 0xff) {
    if (mantissa === 0) return sign | 0x7c00;
    return sign | 0x7e00;
  }

  const halfExponent = exponent - 127 + 15;
  if (halfExponent >= 0x1f) return sign | 0x7c00;
  if (halfExponent <= 0) {
    if (halfExponent < -10) return sign;
    const normalized = mantissa | 0x80_0000;
    const shift = 14 - halfExponent;
    let halfMantissa = normalized >>> shift;
    const remainder = normalized & ((1 << shift) - 1);
    const halfway = 1 << (shift - 1);
    if (remainder > halfway || (remainder === halfway && (halfMantissa & 1) !== 0)) {
      halfMantissa++;
    }
    return sign | halfMantissa;
  }

  let halfMantissa = mantissa >>> 13;
  const remainder = mantissa & 0x1fff;
  let outExponent = halfExponent;
  if (remainder > 0x1000 || (remainder === 0x1000 && (halfMantissa & 1) !== 0)) {
    halfMantissa++;
    if (halfMantissa === 0x0400) {
      halfMantissa = 0;
      outExponent++;
      if (outExponent >= 0x1f) return sign | 0x7c00;
    }
  }
  return sign | (outExponent << 10) | halfMantissa;
}

export function readFloat16(view: DataView, byteOffset: number): number {
  return float16ToNumber(view.getUint16(byteOffset, true));
}

export function decodeFloat16Array(
  view: DataView,
  byteOffset: number,
  length: number,
): Float32Array {
  const result = new Float32Array(length);
  for (let index = 0; index < length; index++) {
    result[index] = float16ToNumber(view.getUint16(byteOffset + index * 2, true));
  }
  return result;
}
