const floatView = new Float32Array(1);
const uintView = new Uint32Array(floatView.buffer);

export function floatToHalf(value: number): number {
  floatView[0] = value;
  const bits = uintView[0];
  const sign = (bits >>> 16) & 0x8000;
  let exponent = ((bits >>> 23) & 0xff) - 127 + 15;
  let mantissa = bits & 0x7fffff;
  if (exponent <= 0) {
    if (exponent < -10) return sign;
    mantissa = (mantissa | 0x800000) >>> (1 - exponent);
    return sign | ((mantissa + 0x1000) >>> 13);
  }
  if (exponent >= 31) return sign | 0x7c00;
  if (mantissa & 0x1000) {
    mantissa += 0x2000;
    if (mantissa & 0x800000) {
      mantissa = 0;
      exponent += 1;
      if (exponent >= 31) return sign | 0x7c00;
    }
  }
  return sign | (exponent << 10) | (mantissa >>> 13);
}

export function toFloat16Bits(values: ArrayLike<number>): Uint16Array {
  return Uint16Array.from(values, floatToHalf);
}

