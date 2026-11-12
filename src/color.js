import {clamp, tensor} from './tensors.js';

const KR = 0.2126;
const KG = 0.7152;
const KB = 0.0722;

export function rgbaToYuv444(rgba, width, videoHeight, modelHeight) {
  const hw = width * modelHeight;
  const yuv = new Float32Array(3 * hw);
  const cb420 = new Float32Array(width * videoHeight / 4);
  const cr420 = new Float32Array(width * videoHeight / 4);
  for (let row = 0; row < videoHeight; row++) {
    for (let col = 0; col < width; col++) {
      const source = (row * width + col) * 4;
      const r = rgba[source] / 255;
      const g = rgba[source + 1] / 255;
      const b = rgba[source + 2] / 255;
      const y = KR * r + KG * g + KB * b;
      const cb = clamp(0.5 + 0.5 * (b - y) / (1 - KB), 0, 1);
      const cr = clamp(0.5 + 0.5 * (r - y) / (1 - KR), 0, 1);
      yuv[row * width + col] = y;
      const uv = Math.floor(row / 2) * (width / 2) + Math.floor(col / 2);
      cb420[uv] += cb * 0.25;
      cr420[uv] += cr * 0.25;
    }
  }
  for (let row = 0; row < modelHeight; row++) {
    for (let col = 0; col < width; col++) {
      const sourceRow = Math.min(row, videoHeight - 1);
      const p = row * width + col;
      const uv = Math.floor(sourceRow / 2) * (width / 2) + Math.floor(col / 2);
      if (row >= videoHeight) yuv[p] = yuv[sourceRow * width + col];
      yuv[hw + p] = cb420[uv];
      yuv[2 * hw + p] = cr420[uv];
    }
  }
  return tensor([1, 3, modelHeight, width], yuv);
}

export function yuv444ToRgba(frame, width, videoHeight) {
  const [, , modelHeight] = frame.dims;
  const hw = width * modelHeight;
  const rgba = new Uint8ClampedArray(width * videoHeight * 4);
  for (let row = 0; row < videoHeight; row++) {
    for (let col = 0; col < width; col++) {
      const p = row * width + col;
      const y = frame.data[p];
      const cb = frame.data[hw + p] - 0.5;
      const cr = frame.data[2 * hw + p] - 0.5;
      const r = y + (2 - 2 * KR) * cr;
      const b = y + (2 - 2 * KB) * cb;
      const g = (y - KR * r - KB * b) / KG;
      const dest = p * 4;
      rgba[dest] = clamp(r, 0, 1) * 255;
      rgba[dest + 1] = clamp(g, 0, 1) * 255;
      rgba[dest + 2] = clamp(b, 0, 1) * 255;
      rgba[dest + 3] = 255;
    }
  }
  return rgba;
}

export function yuv420Psnr(original, reconstructed, width, videoHeight) {
  const stride = width * original.dims[2];
  const squared = [0, 0, 0];
  for (let row = 0; row < videoHeight; row++) {
    for (let col = 0; col < width; col++) {
      const p = row * width + col;
      const yDiff = original.data[p] - reconstructed.data[p];
      squared[0] += yDiff * yDiff;
      if ((row & 1) || (col & 1)) continue;
      for (let plane = 1; plane < 3; plane++) {
        const offset = plane * stride + p;
        const reference = original.data[offset];
        const recovered = (reconstructed.data[offset] +
          reconstructed.data[offset + 1] +
          reconstructed.data[offset + width] +
          reconstructed.data[offset + width + 1]) * 0.25;
        const difference = reference - recovered;
        squared[plane] += difference * difference;
      }
    }
  }
  return squared.map((error, plane) => {
    const samples = width * videoHeight / (plane ? 4 : 1);
    return error === 0 ? Infinity : -10 * Math.log10(error / samples);
  });
}

export function syntheticRgba(width, height, index) {
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let row = 0; row < height; row++) {
    for (let col = 0; col < width; col++) {
      const p = (row * width + col) * 4;
      const active = col > 40 + 12 * index && col < 140 + 12 * index &&
                     row > 70 && row < height - 70;
      rgba[p] = active ? 234 : Math.round(col * 255 / width);
      rgba[p + 1] = active ? 174 : Math.round(row * 255 / height);
      rgba[p + 2] = active ? 36 : Math.round((col + row + index * 7) % 255);
      rgba[p + 3] = 255;
    }
  }
  return rgba;
}
