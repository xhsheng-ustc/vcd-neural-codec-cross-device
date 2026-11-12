// Exact signed INT64 convolution for the browser CPU path. The static WASM
// binary is built from native/exact_conv.wat; no toolchain is needed at run time.

let modulePromise;

export async function createPortableWasmConv() {
  modulePromise ??= fetch(new URL('../vendor/portable/exact_conv.wasm',
    import.meta.url)).then(async response => {
    if (!response.ok) throw new Error('Portable CPU WASM module missing');
    return WebAssembly.compile(await response.arrayBuffer());
  });
  const {exports} = await WebAssembly.instantiate(await modulePromise);
  const memory = exports.memory;
  return {
    quantize(input, step, limit) {
      if (!(step > 0) || !Number.isInteger(Math.log2(step)))
        throw new Error('Portable activation step must be a power of two');
      const outputOffset = input.byteLength;
      const pages = Math.ceil((outputOffset + input.length * 4) / 65536);
      const currentPages = memory.buffer.byteLength / 65536;
      if (pages > currentPages) memory.grow(pages - currentPages);
      new Float32Array(memory.buffer, 0, input.length).set(input);
      exports.quantize(0, outputOffset, input.length, 1 / step, limit);
      return new Int32Array(new Int32Array(memory.buffer,
        outputOffset, input.length));
    },
    silu(input, table, stepBits = 14) {
      const offset = 8 * (1 << stepBits);
      if (table.length !== offset * 2)
        throw new Error('Portable SiLU table shape mismatch');
      const inputOffset = 0;
      const tableOffset = input.byteLength;
      const outputOffset = tableOffset + table.byteLength;
      const pages = Math.ceil((outputOffset + input.byteLength) / 65536);
      const currentPages = memory.buffer.byteLength / 65536;
      if (pages > currentPages) memory.grow(pages - currentPages);
      new Float32Array(memory.buffer, inputOffset, input.length).set(input);
      new Float32Array(memory.buffer, tableOffset, table.length).set(table);
      exports.silu(inputOffset, tableOffset, outputOffset,
        input.length, 1 << stepBits, offset);
      return new Float32Array(new Float32Array(memory.buffer,
        outputOffset, input.length));
    },
    convFixed(input, weights, channels, height, width, outputChannels,
              kind, fixed, options = {}) {
      const kernel = kind === 'pointwise' ? 1 : kind === 'depthwise' ? 3
        : options.kernel;
      const stride = kind === 'general' ? options.stride : 1;
      const padding = kind === 'general' ? options.padding :
        kind === 'depthwise' ? 1 : 0;
      const outH = Math.floor((height + 2 * padding - kernel) / stride) + 1;
      const outW = Math.floor((width + 2 * padding - kernel) / stride) + 1;
      const count = outputChannels * outH * outW;
      const xOffset = 0;
      const wOffset = input.byteLength;
      const sumOffset = Math.ceil((wOffset + weights.byteLength) / 8) * 8;
      const mulOffset = sumOffset + count * 8;
      const shiftOffset = mulOffset + fixed.multiplier.byteLength;
      const biasOffset = shiftOffset + fixed.rightShift.byteLength;
      const outputOffset = biasOffset + fixed.biasQ.byteLength;
      const bytes = outputOffset + count * 4;
      const pages = Math.ceil(bytes / 65536);
      const currentPages = memory.buffer.byteLength / 65536;
      if (pages > currentPages) memory.grow(pages - currentPages);
      new Int32Array(memory.buffer, xOffset, input.length).set(input);
      new Int32Array(memory.buffer, wOffset, weights.length).set(weights);
      new Int32Array(memory.buffer, mulOffset, fixed.multiplier.length)
        .set(fixed.multiplier);
      new Int32Array(memory.buffer, shiftOffset, fixed.rightShift.length)
        .set(fixed.rightShift);
      new Int32Array(memory.buffer, biasOffset, fixed.biasQ.length)
        .set(fixed.biasQ);
      exports.conv(xOffset, wOffset, sumOffset, channels, height, width,
        outputChannels, kind === 'pointwise' ? 0 : kind === 'depthwise' ? 1 : 2,
        kernel, stride, padding, outH, outW);
      exports.dequant_fixed(sumOffset, mulOffset, shiftOffset, biasOffset,
        outputOffset, count, outH * outW);
      return new Float32Array(new Float32Array(memory.buffer,
        outputOffset, count));
    },
  };
}
