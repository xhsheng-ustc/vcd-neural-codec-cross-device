import {loadPortableAssets, loadPortableSiluLut,
  loadPortableIntraQTable} from './portable_assets.js';
import {exactConvWebGpuResident,
  releaseExactConvWebGpu} from './portable_integer_conv.js';
import {exactSiluLut, fixedDecompose, fixedDouble, fixedHalf,
  fixedMultiply} from './portable_fixed_ops.js';
import {createPortableWasmConv} from './portable_wasm_conv.js';
import {fixedDequantGpuResident, readFixedGpuBuffer}
  from './portable_fixed_dequant_gpu.js';
import {gpuSilu, gpuAdd, gpuFold, gpuPixelShuffle,
  gpuChannelMultiply, gpuConcat} from './portable_gpu_block_ops.js';
import {gpuResidentQuantize} from './portable_gpu_resident_quant.js';
import {gpuFixedMultiplyResident} from './portable_gpu_fixed_multiply.js';
import {gpuResampleResident} from './portable_gpu_resample.js';
import {gpuFixedDecompose,
  gpuFixedDecomposeResident} from './portable_gpu_decompose.js';
import {fusedPointwiseGpu, fusedSpatialGpu, releaseFusedPointwiseGpu}
  from './portable_gpu_fused_pointwise.js';
import {packedPointwiseGpu, releasePackedPointwiseGpu}
  from './portable_gpu_packed_pointwise.js';

const ASSET_CACHE = new Map();
let SILU_CACHE;
let INTRA_Q_CACHE;

function pixelShuffle(input, outputChannels, height, width, factor) {
  const result = new Float32Array(outputChannels * height * width * factor * factor);
  const inputPlane = height * width;
  const outputPlane = inputPlane * factor * factor;
  for (let channel = 0; channel < outputChannels; channel++)
    for (let subY = 0; subY < factor; subY++)
      for (let subX = 0; subX < factor; subX++) {
        const source = (channel * factor * factor + subY * factor + subX) * inputPlane;
        for (let y = 0; y < height; y++)
          for (let x = 0; x < width; x++)
            result[channel * outputPlane + (y * factor + subY) *
              (width * factor) + x * factor + subX] =
              input[source + y * width + x];
      }
  return result;
}

// Executes one exported fixed-PTQ layer with an integer Q20 epilogue.
export class PortableModelKernels {
  constructor(assets, siluTable, qTable, backend, device, wasm) {
    this.assets = assets;
    this.siluTable = siluTable;
    this.qTable = qTable;
    this.backend = backend;
    this.device = device;
    this.wasm = wasm;
    this.gpuResidentPairs = backend === 'webgpu';
    this.gpuResidentBlocks = backend === 'webgpu';
    this.fusePointwise = backend === 'webgpu';
    this.fuseSpatial = backend === 'webgpu';
    this.packedPointwise = false;
  }

  static async create(label, backend = 'cpu') {
    if (backend !== 'cpu' && backend !== 'webgpu')
      throw new Error('Portable backend must be cpu or webgpu');
    if (!ASSET_CACHE.has(label))
      ASSET_CACHE.set(label, loadPortableAssets(label));
    SILU_CACHE ??= loadPortableSiluLut();
    if (label === 'i') INTRA_Q_CACHE ??= loadPortableIntraQTable();
    const [assets, siluTable, qTable] = await Promise.all([
      ASSET_CACHE.get(label), SILU_CACHE,
      label === 'i' ? INTRA_Q_CACHE : Promise.resolve(null)]);
    let device = null;
    let wasm = null;
    if (backend === 'webgpu') {
      const adapter = await navigator.gpu?.requestAdapter();
      if (adapter) {
        const bufferLimit = Math.min(1 << 30,
          adapter.limits.maxStorageBufferBindingSize,
          adapter.limits.maxBufferSize);
        device = await adapter.requestDevice({requiredLimits: {
          maxStorageBufferBindingSize: bufferLimit,
          maxBufferSize: bufferLimit,
        }});
      }
      if (!device) throw new Error('WebGPU device unavailable');
    }
    wasm = await createPortableWasmConv();
    return new PortableModelKernels(assets, siluTable, qTable,
      backend, device, wasm);
  }

  async conv(name, input, channels, height, width) {
    const started = this.profile ? performance.now() : 0;
    const layer = this.assets.layer(name);
    const [outputChannels, weightChannels, kh, kw] = layer.shape;
    if (layer.transpose || layer.dilation.some(v => v !== 1))
      throw new Error(`Portable layer shape is not implemented: ${name}`);
    let kind;
    let options = {};
    if (kh === 1 && kw === 1 && layer.groups === 1 &&
        layer.padding.every(v => v === 0) && weightChannels === channels)
      kind = 'pointwise';
    else if (kh === 3 && kw === 3 && layer.groups === channels &&
             outputChannels === channels && weightChannels === 1 &&
             layer.padding.every(v => v === 1))
      kind = 'depthwise';
    else if (layer.groups === 1 && weightChannels === channels &&
             kh === kw && layer.stride[0] === layer.stride[1] &&
             layer.padding[0] === layer.padding[1]) {
      kind = 'general';
      options = {kernel: kh, stride: layer.stride[0],
        padding: layer.padding[0]};
    } else throw new Error(`Portable layer kernel is not implemented: ${name}`);
    const outputHeight = Math.floor((height + 2 * (options.padding ??
      (kind === 'depthwise' ? 1 : 0)) - kh) / (options.stride ?? 1)) + 1;
    const outputWidth = Math.floor((width + 2 * (options.padding ??
      (kind === 'depthwise' ? 1 : 0)) - kw) / (options.stride ?? 1)) + 1;
    const quantized = this.wasm.quantize(input, layer.inputStep,
      layer.activationLimit);
    const quantizedAt = this.profile ? performance.now() : 0;
    let result, computedAt;
    if (this.backend === 'webgpu') {
      const source = this.device.createBuffer({size: quantized.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});
      this.device.queue.writeBuffer(source, 0, quantized);
      const fusedPointwise = kind === 'pointwise' && this.fusePointwise;
      const fusedSpatial = kind !== 'pointwise' && this.fuseSpatial;
      const conv = fusedPointwise
        ? fusedPointwiseGpu(this.device, source, layer.weight, layer.fixed,
          channels, height, width, outputChannels)
        : fusedSpatial
          ? fusedSpatialGpu(this.device, source, layer.weight, layer.fixed,
            channels, height, width, outputChannels, kind, options)
          : await exactConvWebGpuResident(source, layer.weight,
            channels, height, width, outputChannels, kind, this.device, options);
      const fixed = fusedPointwise || fusedSpatial ? conv.buffer
        : await fixedDequantGpuResident(this.device, conv.buffer,
          conv.count, outputHeight * outputWidth, layer.fixed);
      computedAt = this.profile ? performance.now() : 0;
      result = await readFixedGpuBuffer(this.device, fixed, conv.count);
      source.destroy(); conv.buffer.destroy();
      if (fixed !== conv.buffer) fixed.destroy();
    } else {
      result = this.wasm.convFixed(quantized, layer.weight, channels,
        height, width, outputChannels, kind, layer.fixed, options);
      computedAt = this.profile ? performance.now() : 0;
    }
    if (this.profile) this.profile.push({name, elements: result.length,
      quantMs: quantizedAt - started, computeMs: computedAt - quantizedAt,
      dequantMs: performance.now() - computedAt});
    if (this.trace) {
      const hash = Array.from(new Uint8Array(await crypto.subtle.digest(
        'SHA-256', result.buffer))).map(v => v.toString(16).padStart(2, '0')).join('');
      this.trace.push({name, hash});
    }
    return result;
  }

  async depthBlock(prefix, input, channels, height, width,
                   {shortcut = false} = {}) {
    if (this.gpuResidentBlocks && !this.trace)
      return this.depthBlockGpu(prefix, input, channels, height, width,
        shortcut);
    let x = input;
    let outputChannels = channels;
    if (this.assets.names.includes(`${prefix}.adaptor`)) {
      x = await this.conv(`${prefix}.adaptor`, input, channels, height, width);
      outputChannels = this.assets.layer(`${prefix}.adaptor`).shape[0];
    }
    let out = await this.conv(`${prefix}.dc.0`, x, outputChannels, height, width);
    out = this.wasm.silu(out, this.siluTable);
    if (this.gpuResidentPairs && !this.trace)
      out = await this.convPair(`${prefix}.dc.2`, `${prefix}.dc.3`,
        out, outputChannels, height, width);
    else {
      out = await this.conv(`${prefix}.dc.2`, out, outputChannels, height, width);
      out = await this.conv(`${prefix}.dc.3`, out, outputChannels, height, width);
    }
    for (let i = 0; i < out.length; i++) out[i] = Math.fround(out[i] + x[i]);
    const residual = out;
    const widened = await this.conv(`${prefix}.ffn.0`, out,
      outputChannels, height, width);
    const activated = this.wasm.silu(widened, this.siluTable);
    const narrow = new Float32Array(2 * outputChannels * height * width);
    for (let i = 0; i < narrow.length; i++)
      narrow[i] = Math.fround(activated[i] + activated[i + narrow.length]);
    out = await this.conv(`${prefix}.ffn.2`, narrow,
      outputChannels * 2, height, width);
    for (let i = 0; i < out.length; i++) {
      out[i] = Math.fround(out[i] + residual[i]);
      if (shortcut) out[i] = Math.fround(out[i] + x[i]);
    }
    return {data: out, channels: outputChannels, height, width};
  }

  async depthBlockGpu(prefix, input, channels, height, width, shortcut) {
    const started = this.profile ? performance.now() : 0;
    const device = this.device;
    const plane = height * width;
    const temporary = [];
    const keep = buffer => { temporary.push(buffer); return buffer; };
    const source = keep(device.createBuffer({size: input.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST}));
    device.queue.writeBuffer(source, 0, input);
    const runConv = async (name, buffer, inChannels) => {
      const layer = this.assets.layer(name);
      const [outChannels, , kh] = layer.shape;
      const kind = kh === 1 && layer.groups === 1 ? 'pointwise'
        : layer.groups === inChannels ? 'depthwise' : 'general';
      const options = kind === 'general' ? {kernel: kh,
        stride: layer.stride[0], padding: layer.padding[0]} : {};
      const quant = keep(gpuResidentQuantize(device, buffer,
        inChannels * plane, layer.inputStep, layer.activationLimit));
      const conv = await exactConvWebGpuResident(quant, layer.weight,
        inChannels, height, width, outChannels, kind, device, options);
      keep(conv.buffer);
      if (conv.height !== height || conv.width !== width)
        throw new Error(`Depth block changed spatial shape: ${name}`);
      const result = keep(await fixedDequantGpuResident(device,
        conv.buffer, conv.count, plane, layer.fixed));
      return {buffer: result, channels: outChannels};
    };
    try {
      let x = source, outputChannels = channels;
      if (this.assets.names.includes(`${prefix}.adaptor`)) {
        const adapted = await runConv(`${prefix}.adaptor`, source, channels);
        x = adapted.buffer;
        outputChannels = adapted.channels;
      }
      const count = outputChannels * plane;
      let out = (await runConv(`${prefix}.dc.0`, x, outputChannels)).buffer;
      out = keep(gpuSilu(device, out, count, this.siluTable));
      out = (await runConv(`${prefix}.dc.2`, out, outputChannels)).buffer;
      out = (await runConv(`${prefix}.dc.3`, out, outputChannels)).buffer;
      const residual = keep(gpuAdd(device, out, x, count, this.siluTable));
      out = (await runConv(`${prefix}.ffn.0`, residual,
        outputChannels)).buffer;
      out = keep(gpuSilu(device, out, count * 4, this.siluTable));
      out = keep(gpuFold(device, out, count * 2, this.siluTable));
      out = (await runConv(`${prefix}.ffn.2`, out,
        outputChannels * 2)).buffer;
      out = keep(gpuAdd(device, out, residual, count, this.siluTable));
      if (shortcut) out = keep(gpuAdd(device, out, x, count,
        this.siluTable));
      const result = await readFixedGpuBuffer(device, out, count);
      if (this.profile) this.profile.push({name: prefix, elements: count,
        quantMs: 0, computeMs: performance.now() - started, dequantMs: 0});
      return {data: result, channels: outputChannels, height, width};
    } finally {
      for (const buffer of temporary) buffer.destroy();
    }
  }

  async runConvGpuBuffer(name, input, channels, height, width, buffers) {
    const layer = this.assets.layer(name);
    const [outChannels, , kh] = layer.shape;
    const kind = kh === 1 && layer.groups === 1 ? 'pointwise'
      : layer.groups === channels ? 'depthwise' : 'general';
    const options = kind === 'general' ? {kernel: kh,
      stride: layer.stride[0], padding: layer.padding[0]} : {};
    if (this.packedPointwise && kind === 'pointwise' &&
        name.startsWith('dec.dec_1.') && name.endsWith('.ffn.0')) {
      const packed = packedPointwiseGpu(this.device, input, layer.weight,
        layer.fixed, channels, height, width, outChannels,
        layer.inputStep, layer.activationLimit);
      buffers.push(...packed.temporaries, packed.buffer);
      return {buffer: packed.buffer, channels: outChannels,
        height, width};
    }
    const quant = gpuResidentQuantize(this.device, input,
      channels * height * width, layer.inputStep, layer.activationLimit);
    buffers.push(quant);
    const fusedPointwise = kind === 'pointwise' && this.fusePointwise;
    const fusedSpatial = kind !== 'pointwise' && this.fuseSpatial;
    const conv = fusedPointwise
      ? fusedPointwiseGpu(this.device, quant, layer.weight, layer.fixed,
        channels, height, width, outChannels)
      : fusedSpatial
        ? fusedSpatialGpu(this.device, quant, layer.weight, layer.fixed,
          channels, height, width, outChannels, kind, options)
        : await exactConvWebGpuResident(quant, layer.weight,
          channels, height, width, outChannels, kind, this.device, options);
    buffers.push(conv.buffer);
    const output = fusedPointwise || fusedSpatial
      ? conv.buffer
      : await fixedDequantGpuResident(this.device, conv.buffer,
        conv.count, conv.height * conv.width, layer.fixed);
    if (output !== conv.buffer) buffers.push(output);
    return {buffer: output, channels: outChannels,
      height: conv.height, width: conv.width};
  }

  async depthBlockGpuBuffer(prefix, source, channels, height, width,
                             shortcut, buffers) {
    const device = this.device;
    const detail = this.profile && prefix === 'dec.dec_1.1' ? [] : null;
    let detailAt = detail ? performance.now() : 0;
    const checkpoint = async name => {
      if (!detail) return;
      await device.queue.onSubmittedWorkDone();
      const now = performance.now();
      detail.push({name, ms: now - detailAt});
      detailAt = now;
    };
    const count = height * width;
    const firstTemporary = buffers.length;
    const keep = buffer => { buffers.push(buffer); return buffer; };
    let x = source, outputChannels = channels;
    if (this.assets.names.includes(`${prefix}.adaptor`)) {
      const adapted = await this.runConvGpuBuffer(`${prefix}.adaptor`,
        source, channels, height, width, buffers);
      x = adapted.buffer;
      outputChannels = adapted.channels;
    }
    let out = (await this.runConvGpuBuffer(`${prefix}.dc.0`, x,
      outputChannels, height, width, buffers)).buffer;
    await checkpoint('dc_0');
    out = keep(gpuSilu(device, out, outputChannels * count, this.siluTable));
    out = (await this.runConvGpuBuffer(`${prefix}.dc.2`, out,
      outputChannels, height, width, buffers)).buffer;
    await checkpoint('dc_2');
    out = (await this.runConvGpuBuffer(`${prefix}.dc.3`, out,
      outputChannels, height, width, buffers)).buffer;
    await checkpoint('dc_3');
    const residual = keep(gpuAdd(device, out, x, outputChannels * count,
      this.siluTable));
    out = (await this.runConvGpuBuffer(`${prefix}.ffn.0`, residual,
      outputChannels, height, width, buffers)).buffer;
    await checkpoint('ffn_0');
    out = keep(gpuSilu(device, out, outputChannels * count * 4,
      this.siluTable));
    out = keep(gpuFold(device, out, outputChannels * count * 2,
      this.siluTable));
    out = (await this.runConvGpuBuffer(`${prefix}.ffn.2`, out,
      outputChannels * 2, height, width, buffers)).buffer;
    await checkpoint('ffn_2');
    out = keep(gpuAdd(device, out, residual, outputChannels * count,
      this.siluTable));
    if (shortcut) out = keep(gpuAdd(device, out, x,
      outputChannels * count, this.siluTable));
    for (let i = firstTemporary; i < buffers.length; i++)
      if (buffers[i] !== out) buffers[i].destroy();
    if (detail) this.profile.push({name: 'intraBlockDetail',
      elements: outputChannels * count, quantMs: 0, computeMs: 0,
      dequantMs: 0, details: detail});
    return {buffer: out, channels: outputChannels};
  }

  async convPair(firstName, secondName, input, channels, height, width) {
    const {gpuResidentQuantize} =
      await import('./portable_gpu_resident_quant.js');
    const first = this.assets.layer(firstName);
    const second = this.assets.layer(secondName);
    if (first.shape[0] !== channels || second.shape[0] !== channels ||
        first.shape[2] !== 3 || second.shape[2] !== 1)
      throw new Error('GPU-resident pair requires depthwise then pointwise');
    const device = this.device;
    const source = device.createBuffer({size: input.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});
    device.queue.writeBuffer(source, 0, input);
    let current = source;
    const temporary = [source];
    for (const [layer, kind] of [[first, 'depthwise'],
                                 [second, 'pointwise']]) {
      const quant = gpuResidentQuantize(device, current, input.length,
        layer.inputStep, layer.activationLimit);
      temporary.push(quant);
      const conv = await exactConvWebGpuResident(quant, layer.weight,
        channels, height, width, channels, kind, device);
      temporary.push(conv.buffer);
      const dequant = await fixedDequantGpuResident(device, conv.buffer,
        conv.count, height * width, layer.fixed);
      temporary.push(dequant);
      current = dequant;
    }
    const read = device.createBuffer({size: input.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
    const commands = device.createCommandEncoder();
    commands.copyBufferToBuffer(current, 0, read, 0, input.byteLength);
    device.queue.submit([commands.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const output = new Float32Array(new Float32Array(read.getMappedRange()));
    read.unmap();
    read.destroy();
    for (const buffer of temporary) buffer.destroy();
    return output;
  }

  async featureExtractor(input, height, width, quant,
                         quantP1, quantP2, quantP3) {
    if (this.backend === 'webgpu' && !this.trace)
      return this.featureExtractorGpu(input, height, width, quant,
        quantP1, quantP2, quantP3);
    const run = async (branch, count, source, channels, h, w) => {
      let state = {data: source, channels};
      for (let i = 0; i < count; i++)
        state = await this.depthBlock(`feature_extractor.${branch}.${i}`,
          state.data, state.channels, h, w);
      return state.data;
    };
    const x1 = await run('conv1', 2, input, 192, height, width);
    const temporal = fixedMultiply(input, quant, 20, 192);
    const ctx1 = await run('conv2', 4, x1, 192, height, width);
    const half = fixedHalf(x1, 192, height, width);
    const ctx2 = fixedDouble(await run('conv3', 4, half, 192,
      height / 2, width / 2), 192, height / 2, width / 2);
    const ctx3 = fixedDouble(await run('conv4', 4, half, 192,
      height / 2, width / 2), 192, height / 2, width / 2);
    const first = fixedMultiply(ctx1, quantP1, 20, 192);
    const second = fixedMultiply(ctx2, quantP2, 20, 192);
    const third = fixedMultiply(ctx3, quantP3, 20, 192);
    const merged = new Float32Array(first.length * 3);
    merged.set(first);
    merged.set(second, first.length);
    merged.set(third, first.length * 2);
    const context = await this.depthBlock('feature_extractor.fuse',
      merged, 576, height, width);
    return {context: context.data, temporal};
  }

  async featureExtractorGpu(input, height, width, quant,
                             quantP1, quantP2, quantP3) {
    const device = this.device;
    const plane = height * width;
    const buffers = [];
    const keep = buffer => { buffers.push(buffer); return buffer; };
    const source = keep(device.createBuffer({size: input.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST}));
    device.queue.writeBuffer(source, 0, input);
    const branch = async (name, count, initial, h, w) => {
      let current = initial, countChannels = 192;
      for (let i = 0; i < count; i++) {
        const previous = current;
        const block = await this.depthBlockGpuBuffer(
          `feature_extractor.${name}.${i}`, current,
          countChannels, h, w, false, buffers);
        current = block.buffer;
        countChannels = block.channels;
        if (previous !== initial) previous.destroy();
      }
      return current;
    };
    try {
      const x1 = await branch('conv1', 2, source, height, width);
      const temporalBuffer = keep(gpuFixedMultiplyResident(device,
        source, 192 * plane, quant, 192));
      const ctx1 = await branch('conv2', 4, x1, height, width);
      const half = keep(gpuResampleResident(device, x1, 192,
        height, width, 'half'));
      const ctx2Half = await branch('conv3', 4, half,
        height / 2, width / 2);
      const ctx3Half = await branch('conv4', 4, half,
        height / 2, width / 2);
      const ctx2 = keep(gpuResampleResident(device, ctx2Half, 192,
        height / 2, width / 2, 'double'));
      const ctx3 = keep(gpuResampleResident(device, ctx3Half, 192,
        height / 2, width / 2, 'double'));
      const first = keep(gpuFixedMultiplyResident(device, ctx1,
        192 * plane, quantP1, 192));
      const second = keep(gpuFixedMultiplyResident(device, ctx2,
        192 * plane, quantP2, 192));
      const third = keep(gpuFixedMultiplyResident(device, ctx3,
        192 * plane, quantP3, 192));
      const firstTwo = keep(gpuConcat(device, first, 192 * plane,
        second, 192 * plane));
      const merged = keep(gpuConcat(device, firstTwo, 384 * plane,
        third, 192 * plane));
      const fused = await this.depthBlockGpuBuffer(
        'feature_extractor.fuse', merged, 576, height, width,
        false, buffers);
      const context = await readFixedGpuBuffer(device, fused.buffer,
        fused.channels * plane);
      const temporal = await readFixedGpuBuffer(device,
        temporalBuffer, 192 * plane);
      return {context, temporal};
    } finally {
      for (const buffer of buffers) buffer.destroy();
    }
  }

  async runGpuChain(input, channels, height, width,
                    prefixes, finalConv = null) {
    const device = this.device;
    const buffers = [];
    const source = device.createBuffer({size: input.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});
    buffers.push(source);
    device.queue.writeBuffer(source, 0, input);
    let current = source, countChannels = channels;
    try {
      for (const prefix of prefixes) {
        const previous = current;
        const block = await this.depthBlockGpuBuffer(prefix, current,
          countChannels, height, width, false, buffers);
        current = block.buffer;
        countChannels = block.channels;
        previous.destroy();
      }
      if (finalConv) {
        const previous = current;
        const result = await this.runConvGpuBuffer(finalConv,
          current, countChannels, height, width, buffers);
        current = result.buffer;
        countChannels = result.channels;
        previous.destroy();
      }
      return await readFixedGpuBuffer(device, current,
        countChannels * height * width);
    } finally {
      for (const buffer of buffers) buffer.destroy();
    }
  }

  async priorFusion(input, height, width) {
    if (this.backend === 'webgpu' && !this.trace)
      return this.runGpuChain(input, 240, height, width,
        [0, 1, 2].map(i => `y_prior_fusion.conv.${i}`),
        'y_prior_fusion.conv.3');
    let state = input;
    for (let i = 0; i < 3; i++)
      state = (await this.depthBlock(`y_prior_fusion.conv.${i}`,
        state, 240, height, width)).data;
    return this.conv('y_prior_fusion.conv.3', state, 240, height, width);
  }

  async spatialPrior(input, height, width) {
    if (this.backend === 'webgpu' && !this.trace)
      return this.runGpuChain(input, 320, height, width,
        ['y_spatial_prior.conv.0', 'y_spatial_prior.conv.1'],
        'y_spatial_prior.conv.2');
    let state = (await this.depthBlock('y_spatial_prior.conv.0',
      input, 320, height, width)).data;
    state = (await this.depthBlock('y_spatial_prior.conv.1',
      state, 240, height, width)).data;
    return this.conv('y_spatial_prior.conv.2', state, 240, height, width);
  }

  async temporalPrior(input, height, width) {
    if (this.backend === 'webgpu' && !this.trace)
      return this.temporalPriorGpu(input, height, width);
    const down = await this.conv('temporal_prior_encoder.down',
      input, 192, height, width);
    return (await this.depthBlock('temporal_prior_encoder.conv',
      down, 160, height / 2, width / 2, {shortcut: true})).data;
  }

  async temporalPriorGpu(input, height, width) {
    const device = this.device;
    const buffers = [];
    const source = device.createBuffer({size: input.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});
    device.queue.writeBuffer(source, 0, input);
    buffers.push(source);
    try {
      const down = await this.runConvGpuBuffer('temporal_prior_encoder.down',
        source, 192, height, width, buffers);
      source.destroy();
      const block = await this.depthBlockGpuBuffer(
        'temporal_prior_encoder.conv', down.buffer, down.channels,
        down.height, down.width, true, buffers);
      down.buffer.destroy();
      return await readFixedGpuBuffer(device, block.buffer,
        block.channels * down.height * down.width);
    } finally {
      for (const buffer of buffers) buffer.destroy();
    }
  }

  async residualUpsample(prefix, input, channels, outputChannels,
                         height, width) {
    const expanded = await this.conv(`${prefix}.up.conv.0`,
      input, channels, height, width);
    const up = pixelShuffle(expanded, outputChannels, height, width, 2);
    return (await this.depthBlock(`${prefix}.conv`, up, outputChannels,
      height * 2, width * 2, {shortcut: true})).data;
  }

  async hyperDecoder(input, height, width) {
    if (this.backend === 'webgpu' && !this.trace)
      return this.hyperDecoderGpu(input, height, width);
    const isIntra = this.assets.names.includes('hyper_dec.0.up.conv.0');
    const root = isIntra ? 'hyper_dec' : 'hyper_decoder.conv';
    const channels = isIntra ? 128 : 96;
    let state = await this.residualUpsample(`${root}.0`, input,
      channels, channels, height, width);
    state = await this.residualUpsample(`${root}.1`, state,
      channels, channels, height * 2, width * 2);
    return (await this.depthBlock(`${root}.2`, state, channels,
      height * 4, width * 4)).data;
  }

  async hyperDecoderGpu(input, height, width) {
    const device = this.device;
    const buffers = [];
    const source = device.createBuffer({size: input.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});
    device.queue.writeBuffer(source, 0, input);
    buffers.push(source);
    const isIntra = this.assets.names.includes('hyper_dec.0.up.conv.0');
    const root = isIntra ? 'hyper_dec' : 'hyper_decoder.conv';
    const channels = isIntra ? 128 : 96;
    let current = source, h = height, w = width;
    try {
      for (let i = 0; i < 2; i++) {
        const conv = await this.runConvGpuBuffer(`${root}.${i}.up.conv.0`,
          current, channels, h, w, buffers);
        current.destroy();
        const up = gpuPixelShuffle(device, conv.buffer, channels, h, w, 2);
        buffers.push(up);
        conv.buffer.destroy();
        h *= 2; w *= 2;
        const block = await this.depthBlockGpuBuffer(`${root}.${i}.conv`,
          up, channels, h, w, true, buffers);
        up.destroy();
        current = block.buffer;
      }
      const final = await this.depthBlockGpuBuffer(`${root}.2`,
        current, channels, h, w, false, buffers);
      current.destroy();
      return await readFixedGpuBuffer(device, final.buffer,
        final.channels * h * w);
    } finally {
      for (const buffer of buffers) buffer.destroy();
    }
  }

  async decoder(input, context, height, width, quantLow, quantHigh) {
    if (this.backend === 'webgpu' && !this.trace)
      return this.decoderGpu(input, context, height, width,
        quantLow, quantHigh);
    const upConv = await this.conv('decoder.up.conv.0', input,
      80, height, width);
    const up = pixelShuffle(upConv, 192, height, width, 2);
    const outH = height * 2, outW = width * 2;
    if (context.length !== up.length)
      throw new Error('P decoder context shape mismatch');
    const merged = new Float32Array(up.length * 2);
    merged.set(up);
    merged.set(context, up.length);
    let state = merged, channels = 384;
    for (let i = 0; i < 3; i++) {
      const block = await this.depthBlock(`decoder.conv1.${i}`,
        state, channels, outH, outW);
      state = block.data;
      channels = block.channels;
    }
    state = await this.conv('decoder.conv2', state, 192, outH, outW);
    if (this.backend === 'webgpu' && !this.trace) {
      const result = await gpuFixedDecompose(this.device, state,
        quantLow, quantHigh, 192, outH, outW);
      if (result) {
        if (this.profile) this.profile.push({name: 'fixedDecomposeGpu',
          elements: result.length, quantMs: 0, computeMs: 0, dequantMs: 0});
        return result;
      }
      if (this.profile) this.profile.push({name: 'fixedDecomposeCpuFallback',
        elements: state.length,
        maxAbs: state.reduce((m, value) => Math.max(m, Math.abs(value)), 0),
        quantMs: 0, computeMs: 0, dequantMs: 0});
    }
    return fixedDecompose(state, quantLow, quantHigh,
      192, outH, outW);
  }

  async decoderGpu(input, context, height, width, quantLow, quantHigh) {
    const device = this.device;
    const buffers = [];
    const upload = values => {
      const buffer = device.createBuffer({size: values.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});
      device.queue.writeBuffer(buffer, 0, values);
      buffers.push(buffer);
      return buffer;
    };
    const source = upload(input);
    const outH = height * 2, outW = width * 2;
    if (context.length !== 192 * outH * outW)
      throw new Error('P decoder context shape mismatch');
    try {
      const upConv = await this.runConvGpuBuffer('decoder.up.conv.0',
        source, 80, height, width, buffers);
      source.destroy();
      const up = gpuPixelShuffle(device, upConv.buffer, 192,
        height, width, 2);
      buffers.push(up); upConv.buffer.destroy();
      const contextBuffer = upload(context);
      let current = gpuConcat(device, up, 192 * outH * outW,
        contextBuffer, context.length);
      buffers.push(current);
      up.destroy(); contextBuffer.destroy();
      let channels = 384;
      for (let i = 0; i < 3; i++) {
        const previous = current;
        const block = await this.depthBlockGpuBuffer(`decoder.conv1.${i}`,
          current, channels, outH, outW, false, buffers);
        current = block.buffer; channels = block.channels;
        previous.destroy();
      }
      const prior = await this.runConvGpuBuffer('decoder.conv2', current,
        192, outH, outW, buffers);
      current.destroy();
      const decomposed = gpuFixedDecomposeResident(device, prior.buffer,
        quantLow, quantHigh, 192, outH, outW);
      buffers.push(decomposed); prior.buffer.destroy();
      if (this.profile) this.profile.push({name: 'fixedDecomposeGpu',
        elements: 192 * outH * outW,
        quantMs: 0, computeMs: 0, dequantMs: 0});
      return await readFixedGpuBuffer(device, decomposed,
        192 * outH * outW);
    } finally {
      for (const buffer of buffers) buffer.destroy();
    }
  }

  async reconstruction(input, height, width, quant) {
    if (this.backend === 'webgpu' && !this.trace)
      return this.reconstructionGpu(input, height, width, quant);
    let state = input;
    for (let i = 0; i < 4; i++)
      state = (await this.depthBlock(`recon_generation_net_lpips.conv.${i}`,
        state, 192, height, width)).data;
    if (quant.length !== 192) throw new Error('Reconstruction quantizer shape mismatch');
    const plane = height * width;
    for (let c = 0; c < 192; c++)
      for (let p = 0; p < plane; p++)
        state[c * plane + p] = Math.fround(state[c * plane + p] * quant[c]);
    const head = await this.conv('recon_generation_net_lpips.head',
      state, 192, height, width);
    return pixelShuffle(head, 3, height, width, 16);
  }

  async reconstructionGpu(input, height, width, quant) {
    if (quant.length !== 192) throw new Error('Reconstruction quantizer shape mismatch');
    const device = this.device;
    const buffers = [];
    const source = device.createBuffer({size: input.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});
    device.queue.writeBuffer(source, 0, input);
    buffers.push(source);
    let current = source;
    try {
      for (let i = 0; i < 4; i++) {
        const previous = current;
        const block = await this.depthBlockGpuBuffer(
          `recon_generation_net_lpips.conv.${i}`,
          current, 192, height, width, false, buffers);
        current = block.buffer;
        previous.destroy();
      }
      const scaled = gpuChannelMultiply(device, current,
        192 * height * width, height * width, quant);
      buffers.push(scaled);
      current.destroy();
      const head = await this.runConvGpuBuffer(
        'recon_generation_net_lpips.head', scaled,
        192, height, width, buffers);
      scaled.destroy();
      const shuffled = gpuPixelShuffle(device, head.buffer, 3,
        height, width, 16);
      buffers.push(shuffled);
      head.buffer.destroy();
      return await readFixedGpuBuffer(device, shuffled,
        3 * height * width * 256);
    } finally {
      for (const buffer of buffers) buffer.destroy();
    }
  }

  async intraDecoder(input, height, width, quant) {
    if (this.backend === 'webgpu' && !this.trace)
      return this.intraDecoderGpu(input, height, width, quant);
    const upConv = await this.conv('dec.dec_1.0.up.conv.0',
      input, 256, height, width);
    const outH = height * 2, outW = width * 2;
    let state = pixelShuffle(upConv, 368, height, width, 2);
    state = (await this.depthBlock('dec.dec_1.0.conv',
      state, 368, outH, outW, {shortcut: true})).data;
    for (let i = 1; i <= 12; i++)
      state = (await this.depthBlock(`dec.dec_1.${i}`,
        state, 368, outH, outW)).data;
    if (quant.length !== 368) throw new Error('I decoder quantizer shape mismatch');
    const plane = outH * outW;
    for (let c = 0; c < 368; c++)
      for (let p = 0; p < plane; p++)
        state[c * plane + p] = Math.fround(state[c * plane + p] * quant[c]);
    state = (await this.depthBlock('dec.dec_2',
      state, 368, outH, outW)).data;
    return pixelShuffle(state, 3, outH, outW, 8);
  }

  async intraDecoderGpu(input, height, width, quant) {
    const started = this.profile ? performance.now() : 0;
    const device = this.device;
    const details = [];
    let checkpoint = async name => {
      if (!this.profile) return;
      await device.queue.onSubmittedWorkDone();
      const now = performance.now();
      details.push({name, ms: now - checkpoint.last});
      checkpoint.last = now;
    };
    checkpoint.last = started;
    const buffers = [];
    const keep = buffer => { buffers.push(buffer); return buffer; };
    const source = keep(device.createBuffer({size: input.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST}));
    device.queue.writeBuffer(source, 0, input);
    try {
      let state = await this.runConvGpuBuffer('dec.dec_1.0.up.conv.0',
        source, 256, height, width, buffers);
      await checkpoint('up_conv');
      const outH = height * 2, outW = width * 2;
      let current = keep(gpuPixelShuffle(device, state.buffer, 368,
        height, width, 2));
      state = await this.depthBlockGpuBuffer('dec.dec_1.0.conv',
        current, 368, outH, outW, true, buffers);
      await checkpoint('block_0');
      current.destroy();
      current = state.buffer;
      for (let i = 1; i <= 12; i++) {
        const previous = current;
        state = await this.depthBlockGpuBuffer(`dec.dec_1.${i}`,
          current, 368, outH, outW, false, buffers);
        await checkpoint(`block_${i}`);
        current = state.buffer;
        previous.destroy();
      }
      current = keep(gpuChannelMultiply(device, current,
        368 * outH * outW, outH * outW, quant));
      state = await this.depthBlockGpuBuffer('dec.dec_2', current,
        368, outH, outW, false, buffers);
      await checkpoint('output_block');
      current = keep(gpuPixelShuffle(device, state.buffer, 3,
        outH, outW, 8));
      const result = await readFixedGpuBuffer(device, current,
        3 * outH * outW * 64);
      await checkpoint('pixel_shuffle_read');
      if (this.profile) this.profile.push({name: 'intraDecoderGpu',
        elements: result.length, quantMs: 0,
        computeMs: performance.now() - started, dequantMs: 0, details});
      return result;
    } finally {
      for (const buffer of buffers) buffer.destroy();
    }
  }

  async intraPriorFusion(input, height, width) {
    if (this.backend === 'webgpu' && !this.trace)
      return this.runGpuChain(input, 256, height, width,
        [0, 1, 2].map(i => `y_prior_fusion.${i}`),
        'y_prior_fusion.3');
    let state = input, channels = 256;
    for (let i = 0; i < 3; i++) {
      const block = await this.depthBlock(`y_prior_fusion.${i}`,
        state, channels, height, width);
      state = block.data;
      channels = block.channels;
    }
    return this.conv('y_prior_fusion.3', state, 512, height, width);
  }

  async intraPriorReduction(input, height, width) {
    return this.conv('y_spatial_prior_reduction',
      input, 514, height, width);
  }

  async intraSpatialAdaptor(number, input, height, width) {
    if (![1, 2, 3].includes(number)) throw new Error('Unknown I-frame adaptor');
    return (await this.depthBlock(`y_spatial_prior_adaptor_${number}`,
      input, 512, height, width)).data;
  }

  async intraSpatialPrior(input, height, width) {
    if (this.backend === 'webgpu' && !this.trace)
      return this.runGpuChain(input, 512, height, width,
        [0, 1, 2].map(i => `y_spatial_prior.${i}`),
        'y_spatial_prior.3');
    let state = input;
    for (let i = 0; i < 3; i++)
      state = (await this.depthBlock(`y_spatial_prior.${i}`,
        state, 512, height, width)).data;
    return this.conv('y_spatial_prior.3', state, 512, height, width);
  }

  async featureAdaptorI(input, height, width) {
    return (await this.depthBlock('feature_adaptor_i',
      input, 768, height, width)).data;
  }

  async featureAdaptorP(input, height, width) {
    return this.conv('feature_adaptor_p', input, 192, height, width);
  }

  close() {
    if (this.device) {
      releaseExactConvWebGpu(this.device);
      releaseFusedPointwiseGpu(this.device);
      releasePackedPointwiseGpu(this.device);
      this.device.destroy();
    }
  }
}
