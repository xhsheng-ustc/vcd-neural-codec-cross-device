// Browser reference for the desktop 20-bit fixed-point feature operations.
// BigInt is intentional here: intermediate products in decomposition may
// exceed JavaScript's exact Number range even when the output does not.

function checkedBits(bits) {
  if (![16, 20, 24].includes(bits)) throw new RangeError('Unsupported grid bits');
  return 2 ** bits;
}

function roundEvenNumber(value) {
  const lower = Math.floor(value);
  const fraction = value - lower;
  return fraction < 0.5 ? lower : fraction > 0.5 ? lower + 1 :
    (lower % 2 === 0 ? lower : lower + 1);
}

function roundEvenDiv(value, denominator) {
  const negative = value < 0n;
  const magnitude = negative ? -value : value;
  const quotient = magnitude / denominator;
  const remainder = magnitude % denominator;
  const increase = remainder * 2n > denominator ||
    (remainder * 2n === denominator && quotient % 2n === 1n);
  const result = quotient + (increase ? 1n : 0n);
  return negative ? -result : result;
}

function grid(values, scale) {
  return Array.from(values, value => {
    if (!Number.isFinite(value)) throw new RangeError('Nonfinite feature');
    return BigInt(roundEvenNumber(value * scale));
  });
}

function floatGrid(values, scale) {
  return Float32Array.from(values, value => Math.fround(Number(value) / scale));
}

function validateTensor(values, channels, height, width) {
  if (values.length !== channels * height * width) throw new RangeError('Tensor shape mismatch');
}

function index(channel, row, col, height, width) {
  return (channel * height + row) * width + col;
}

function upsample2Numerator(values, channels, height, width) {
  const result = Array(channels * height * width * 4);
  for (let c = 0; c < channels; c++) {
    for (let y = 0; y < height * 2; y++) {
      const baseY = Math.floor(y / 2);
      const y0 = y % 2 === 0 ? Math.max(baseY - 1, 0) : baseY;
      const y1 = y % 2 === 0 ? baseY : Math.min(baseY + 1, height - 1);
      const wy0 = y % 2 === 0 ? 1n : 3n;
      const wy1 = 4n - wy0;
      for (let x = 0; x < width * 2; x++) {
        const baseX = Math.floor(x / 2);
        const x0 = x % 2 === 0 ? Math.max(baseX - 1, 0) : baseX;
        const x1 = x % 2 === 0 ? baseX : Math.min(baseX + 1, width - 1);
        const wx0 = x % 2 === 0 ? 1n : 3n;
        const wx1 = 4n - wx0;
        result[index(c, y, x, height * 2, width * 2)] =
          values[index(c, y0, x0, height, width)] * wy0 * wx0 +
          values[index(c, y0, x1, height, width)] * wy0 * wx1 +
          values[index(c, y1, x0, height, width)] * wy1 * wx0 +
          values[index(c, y1, x1, height, width)] * wy1 * wx1;
      }
    }
  }
  return result;
}

export function fixedHalf(values, channels, height, width, bits = 20) {
  validateTensor(values, channels, height, width);
  if (height % 2 || width % 2) throw new RangeError('Even spatial dimensions required');
  const scale = checkedBits(bits);
  const q = grid(values, scale);
  const result = [];
  for (let c = 0; c < channels; c++) {
    for (let y = 0; y < height; y += 2) {
      for (let x = 0; x < width; x += 2) {
        const sum = q[index(c,y,x,height,width)] + q[index(c,y,x+1,height,width)] +
          q[index(c,y+1,x,height,width)] + q[index(c,y+1,x+1,height,width)];
        result.push(roundEvenDiv(sum, 4n));
      }
    }
  }
  return floatGrid(result, scale);
}

export function fixedDouble(values, channels, height, width, bits = 20) {
  validateTensor(values, channels, height, width);
  const scale = checkedBits(bits);
  const numerators = upsample2Numerator(grid(values, scale), channels, height, width);
  return floatGrid(numerators.map(value => roundEvenDiv(value, 16n)), scale);
}

export function fixedMultiply(values, quant, bits = 20, channels = null) {
  const scale = checkedBits(bits);
  const xq = grid(values, scale);
  const qq = grid(quant, scale);
  if (qq.length !== 1 && qq.length !== values.length &&
      !(channels && qq.length === channels && values.length % channels === 0))
    throw new RangeError('Quantizer shape mismatch');
  const limit = 1n << 62n;
  return floatGrid(xq.map((value, i) => {
    const quantIndex = qq.length === 1 ? 0 : qq.length === values.length
      ? i : Math.floor(i / (values.length / channels));
    const product = value * qq[quantIndex];
    if (product <= -limit || product >= limit) throw new RangeError('INT64 overflow');
    return roundEvenDiv(product, BigInt(scale));
  }), scale);
}

export function fixedDecompose(values, lowQuant, highQuant, channels, height, width, bits = 20) {
  validateTensor(values, channels, height, width);
  if (height % 2 || width % 2) throw new RangeError('Even spatial dimensions required');
  const scale = checkedBits(bits);
  const q = grid(values, scale);
  const down = [];
  for (let c = 0; c < channels; c++) {
    for (let y = 0; y < height; y += 2) {
      for (let x = 0; x < width; x += 2) {
        down.push(q[index(c,y,x,height,width)] + q[index(c,y,x+1,height,width)] +
          q[index(c,y+1,x,height,width)] + q[index(c,y+1,x+1,height,width)]);
      }
    }
  }
  const low = upsample2Numerator(down, channels, height / 2, width / 2);
  const ql = grid(lowQuant, scale);
  const qh = grid(highQuant, scale);
  if (![1, channels].includes(ql.length) || ![1, channels].includes(qh.length))
    throw new RangeError('Quantizer channel mismatch');
  const limit = 1n << 62n;
  const output = q.map((value, i) => {
    const channel = Math.floor(i / (height * width));
    const high = value * 64n - low[i];
    const numerator = low[i] * ql[ql.length === 1 ? 0 : channel] +
      high * qh[qh.length === 1 ? 0 : channel];
    if (numerator <= -limit || numerator >= limit) throw new RangeError('INT64 overflow');
    return roundEvenDiv(numerator, BigInt(64 * scale));
  });
  return floatGrid(output, scale);
}

export function exactSiluLut(values, table, stepBits = 14) {
  const step = 2 ** stepBits;
  const offset = 8 * step;
  if (table.length !== offset * 2) throw new RangeError('SiLU table length mismatch');
  return Float32Array.from(values, value => {
    const q = Math.max(-offset, Math.min(offset - 1, roundEvenNumber(Math.fround(value * step))));
    return table[q + offset];
  });
}
