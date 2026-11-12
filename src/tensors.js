export function tensor(dims, data) {
  return {dims, data: data instanceof Float32Array ? data : Float32Array.from(data)};
}

export function asTensor(ortTensor) {
  return tensor(Array.from(ortTensor.dims), ortTensor.data);
}

export function flatNumbers(values) {
  return Float32Array.from(values.flat(Infinity));
}

export function sliceChannels(input, begin, end) {
  const [, , h, w] = input.dims;
  const size = h * w;
  return tensor([1, end - begin, h, w], input.data.slice(begin * size, end * size));
}

export function concatChannels(...inputs) {
  const [, , h, w] = inputs[0].dims;
  const channels = inputs.reduce((sum, item) => sum + item.dims[1], 0);
  const output = new Float32Array(channels * h * w);
  let offset = 0;
  for (const item of inputs) {
    if (item.dims[2] !== h || item.dims[3] !== w) throw new Error('tensor shape mismatch');
    output.set(item.data, offset);
    offset += item.data.length;
  }
  return tensor([1, channels, h, w], output);
}

export function add(a, b) {
  if (a.data.length !== b.data.length) throw new Error('tensor shape mismatch');
  const output = new Float32Array(a.data.length);
  for (let i = 0; i < output.length; i++) output[i] = a.data[i] + b.data[i];
  return tensor(a.dims, output);
}

export function multiplySpatial(input, scale) {
  const [, channels, h, w] = input.dims;
  const hw = h * w;
  const output = new Float32Array(input.data.length);
  for (let c = 0; c < channels; c++) {
    for (let p = 0; p < hw; p++) output[c * hw + p] = input.data[c * hw + p] * scale.data[p];
  }
  return tensor(input.dims, output);
}

export function cropSpatial(input, h, w) {
  const [, channels, sourceH, sourceW] = input.dims;
  if (h === sourceH && w === sourceW) return input;
  const output = new Float32Array(channels * h * w);
  for (let c = 0; c < channels; c++) {
    for (let row = 0; row < h; row++) {
      output.set(input.data.subarray(c * sourceH * sourceW + row * sourceW,
                                     c * sourceH * sourceW + row * sourceW + w),
                 c * h * w + row * w);
    }
  }
  return tensor([1, channels, h, w], output);
}

export function padToMultiple(input, multiple) {
  const [, channels, h, w] = input.dims;
  const outH = Math.ceil(h / multiple) * multiple;
  const outW = Math.ceil(w / multiple) * multiple;
  if (outH === h && outW === w) return input;
  const output = new Float32Array(channels * outH * outW);
  for (let c = 0; c < channels; c++) {
    for (let row = 0; row < outH; row++) {
      for (let col = 0; col < outW; col++) {
        output[c * outH * outW + row * outW + col] =
          input.data[c * h * w + Math.min(row, h - 1) * w + Math.min(col, w - 1)];
      }
    }
  }
  return tensor([1, channels, outH, outW], output);
}

export function pixelUnshuffle(input, factor) {
  const [, channels, h, w] = input.dims;
  if (h % factor || w % factor) throw new Error('pixel unshuffle size mismatch');
  const outH = h / factor;
  const outW = w / factor;
  const output = new Float32Array(channels * factor * factor * outH * outW);
  for (let c = 0; c < channels; c++) {
    for (let subY = 0; subY < factor; subY++) {
      for (let subX = 0; subX < factor; subX++) {
        const outChannel = c * factor * factor + subY * factor + subX;
        for (let row = 0; row < outH; row++) {
          for (let col = 0; col < outW; col++) {
            output[outChannel * outH * outW + row * outW + col] =
              input.data[c * h * w + (row * factor + subY) * w + col * factor + subX];
          }
        }
      }
    }
  }
  return tensor([1, channels * factor * factor, outH, outW], output);
}

export function roundEven(value) {
  const floor = Math.floor(value);
  const fraction = value - floor;
  if (fraction < 0.5) return floor;
  if (fraction > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}

export function clamp(value, low, high) {
  return Math.max(low, Math.min(high, value));
}
