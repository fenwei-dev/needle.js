import { describe, expect, test } from "bun:test";
import { float16ToNumber, numberToFloat16 } from "../src/model/fp16.js";

describe("binary16", () => {
  test("decodes known bit patterns", () => {
    expect(float16ToNumber(0x0000)).toBe(0);
    expect(Object.is(float16ToNumber(0x8000), -0)).toBe(true);
    expect(float16ToNumber(0x3c00)).toBe(1);
    expect(float16ToNumber(0xc000)).toBe(-2);
    expect(float16ToNumber(0x7c00)).toBe(Infinity);
    expect(Number.isNaN(float16ToNumber(0x7e00))).toBe(true);
    expect(float16ToNumber(0x0001)).toBeCloseTo(2 ** -24, 12);
  });

  test("round-trips representative finite values", () => {
    for (const value of [0, -0, 1, -2, 0.33325, 65_504, 2 ** -14, 2 ** -24]) {
      const decoded = float16ToNumber(numberToFloat16(value));
      expect(decoded).toBeCloseTo(value, Math.abs(value) < 1e-4 ? 10 : 3);
    }
  });
});
