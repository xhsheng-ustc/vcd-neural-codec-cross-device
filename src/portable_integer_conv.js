// W16A16 browser port: each signed product fits in i32, with i64 accumulation.
// The active codec passes signed INT64 sum pairs to the fixed Q20 GPU epilogue.
// Legacy host readback helpers below are retained for standalone probes.

const SHADER = `
@group(0) @binding(0) var<storage, read> x: array<i32>;
@group(0) @binding(1) var<storage, read> w: array<i32>;
@group(0) @binding(2) var<storage, read_write> out64: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read> params: array<u32>;

fn product64(a: i32, b: i32) -> vec2<u32> {
  let value = a * b;
  return vec2<u32>(bitcast<u32>(value),
    select(0u, 0xffffffffu, value < 0));
}

fn add64(acc: vec2<u32>, value: vec2<u32>) -> vec2<u32> {
  let lo = acc.x + value.x;
  return vec2<u32>(lo, acc.y + value.y + u32(lo < acc.x));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let channels = params[0];
  let height = params[1];
  let width = params[2];
  let output_channels = params[3];
  let kind = params[4];
  let input_plane = height * width;
  let kernel = params[5];
  let stride = params[6];
  let padding = params[7];
  let output_height = params[8];
  let output_width = params[9];
  let plane = output_height * output_width;
  let local_index = id.x + id.y * params[10] * 64u;
  if (local_index >= params[12]) { return; }
  let index = params[11] + local_index;
  let channel = index / plane;
  let location = index % plane;
  var acc = vec2<u32>(0u, 0u);
  if (kind == 0u) {
    for (var k = 0u; k < channels; k = k + 1u) {
      acc = add64(acc, product64(x[k * input_plane + location],
                                  w[channel * channels + k]));
    }
  } else if (kind == 1u) {
    let y = i32(location / output_width);
    let xpos = i32(location % output_width);
    for (var ky = 0; ky < 3; ky = ky + 1) {
      let iy = y + ky - 1;
      if (iy < 0 || iy >= i32(height)) { continue; }
      for (var kx = 0; kx < 3; kx = kx + 1) {
        let ix = xpos + kx - 1;
        if (ix < 0 || ix >= i32(width)) { continue; }
        let input_index = channel * input_plane + u32(iy) * width + u32(ix);
        let weight_index = channel * 9u + u32(ky * 3 + kx);
        acc = add64(acc, product64(x[input_index], w[weight_index]));
      }
    }
  } else {
    let y = i32(location / output_width);
    let xpos = i32(location % output_width);
    for (var ic = 0u; ic < channels; ic = ic + 1u) {
      for (var ky = 0u; ky < kernel; ky = ky + 1u) {
        let iy = y * i32(stride) + i32(ky) - i32(padding);
        if (iy < 0 || iy >= i32(height)) { continue; }
        for (var kx = 0u; kx < kernel; kx = kx + 1u) {
          let ix = xpos * i32(stride) + i32(kx) - i32(padding);
          if (ix < 0 || ix >= i32(width)) { continue; }
          let xi = ic * input_plane + u32(iy) * width + u32(ix);
          let wi = ((channel * channels + ic) * kernel + ky) * kernel + kx;
          acc = add64(acc, product64(x[xi], w[wi]));
        }
      }
    }
  }
  out64[local_index] = acc;
}`;

const POINTWISE_SHADER = SHADER.slice(0, SHADER.indexOf('@compute')) + `
var<workgroup> tile_x: array<i32, 256>;
var<workgroup> tile_w: array<i32, 256>;

@compute @workgroup_size(8, 8)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local: vec3<u32>) {
  let channels = params[0];
  let plane = params[1];
  let base_channel = params[3];
  let chunk_channels = params[4];
  let local_channel = group.y * 8u + local.y;
  let spatial = group.x * 8u + local.x;
  let global_channel = base_channel + local_channel;
  let valid = local_channel < chunk_channels && spatial < plane;
  let lane = local.y * 8u + local.x;
  var acc = vec2<u32>(0u, 0u);
  for (var base = 0u; base < channels; base = base + 32u) {
    for (var slot = lane; slot < 256u; slot = slot + 64u) {
      let kx = base + slot / 8u;
      let px = group.x * 8u + slot % 8u;
      var xv = 0;
      if (kx < channels && px < plane) { xv = x[kx * plane + px]; }
      tile_x[slot] = xv;
      let kw = base + slot % 32u;
      let cw = base_channel + group.y * 8u + slot / 32u;
      var wv = 0;
      if (kw < channels && cw < base_channel + chunk_channels) {
        wv = w[cw * channels + kw];
      }
      tile_w[slot] = wv;
    }
    workgroupBarrier();
    if (valid) {
      for (var k = 0u; k < 32u; k = k + 1u) {
        acc = add64(acc, product64(tile_x[k * 8u + local.x],
                                   tile_w[local.y * 32u + k]));
      }
    }
    workgroupBarrier();
  }
  if (valid) {
    let output_index = local_channel * plane + spatial;
    out64[output_index] = acc;
  }
}`;

const PIPELINES = new WeakMap();
const POINTWISE_PIPELINES = new WeakMap();
const WEIGHT_BUFFERS = new WeakMap();

export function releaseExactConvWebGpu(device) {
  const cached = WEIGHT_BUFFERS.get(device);
  if (cached) for (const buffer of cached.values()) buffer.destroy();
  WEIGHT_BUFFERS.delete(device);
  PIPELINES.delete(device);
  POINTWISE_PIPELINES.delete(device);
}

export function roundEven(value) {
  const lower = Math.floor(value);
  const fraction = value - lower;
  return fraction < 0.5 ? lower : fraction > 0.5 ? lower + 1
    : lower % 2 === 0 ? lower : lower + 1;
}

export function quantizePowerTwo(input, step, limit = 32767) {
  if (!(step > 0) || !Number.isInteger(Math.log2(step)))
    throw new Error('The fixed PTQ activation step must be a power of two');
  const output = new Int32Array(input.length);
  for (let i = 0; i < input.length; i++) {
    if (!Number.isFinite(input[i])) throw new Error('Nonfinite activation');
    output[i] = Math.max(-limit, Math.min(limit, roundEven(input[i] / step)));
  }
  return output;
}

export function exactConvCpu(input, weights, channels, height, width,
                             outputChannels, kind, options = {}) {
  const plane = height * width;
  const kernel = kind === 'pointwise' ? 1 : kind === 'depthwise' ? 3 : options.kernel;
  const stride = kind === 'general' ? options.stride : 1;
  const padding = kind === 'general' ? options.padding : kind === 'depthwise' ? 1 : 0;
  if (!Number.isInteger(kernel) || !Number.isInteger(stride) ||
      !Number.isInteger(padding)) throw new Error('Invalid convolution parameters');
  const outH = Math.floor((height + 2 * padding - kernel) / stride) + 1;
  const outW = Math.floor((width + 2 * padding - kernel) / stride) + 1;
  const outputPlane = outH * outW;
  if (input.length !== channels * plane ||
      weights.length !== (kind === 'pointwise' ? outputChannels * channels
        : kind === 'depthwise' ? channels * 9 :
          outputChannels * channels * kernel * kernel) ||
      (kind === 'depthwise' && outputChannels !== channels))
    throw new Error('Invalid convolution shape');
  const output = new Float64Array(outputChannels * outputPlane);
  for (let oc = 0; oc < outputChannels; oc++) {
    for (let p = 0; p < outputPlane; p++) {
      let sum = 0;
      if (kind === 'pointwise') {
        for (let ic = 0; ic < channels; ic++)
          sum += input[ic * plane + p] * weights[oc * channels + ic];
      } else if (kind === 'depthwise') {
        const y = Math.floor(p / width), x = p % width;
        for (let ky = 0; ky < 3; ky++) for (let kx = 0; kx < 3; kx++) {
          const iy = y + ky - 1, ix = x + kx - 1;
          if (iy >= 0 && iy < height && ix >= 0 && ix < width)
            sum += input[oc * plane + iy * width + ix] *
              weights[oc * 9 + ky * 3 + kx];
        }
      } else {
        const y = Math.floor(p / outW), x = p % outW;
        for (let ic = 0; ic < channels; ic++)
          for (let ky = 0; ky < kernel; ky++)
            for (let kx = 0; kx < kernel; kx++) {
              const iy = y * stride + ky - padding;
              const ix = x * stride + kx - padding;
              if (iy >= 0 && iy < height && ix >= 0 && ix < width)
                sum += input[ic * plane + iy * width + ix] *
                  weights[((oc * channels + ic) * kernel + ky) * kernel + kx];
            }
      }
      if (!Number.isSafeInteger(sum)) throw new Error('INT64 result exceeds exact FP64 range');
      output[oc * outputPlane + p] = sum;
    }
  }
  return output;
}

export function dequantizeSums(sums, inputStep, weightSteps, bias, height, width) {
  const plane = height * width;
  const output = new Float32Array(sums.length);
  for (let i = 0; i < sums.length; i++) {
    const channel = Math.floor(i / plane);
    let value = (sums[i] * inputStep) * weightSteps[channel];
    if (bias) value += bias[channel];
    output[i] = Math.fround(value);
  }
  return output;
}

function upload(device, data, usage) {
  const buffer = device.createBuffer({size: data.byteLength, usage,
    mappedAtCreation: true});
  new data.constructor(buffer.getMappedRange()).set(data);
  buffer.unmap();
  return buffer;
}

async function tiledPointwiseWebGpu(device, inputBuffer, weightBuffer,
                                    channels, plane, outputChannels) {
  let pipeline = POINTWISE_PIPELINES.get(device);
  if (!pipeline) {
    const module = device.createShaderModule({code: POINTWISE_SHADER});
    pipeline = device.createComputePipeline({layout: 'auto',
      compute: {module, entryPoint: 'main'}});
    POINTWISE_PIPELINES.set(device, pipeline);
  }
  const maxElements = Math.min(8 * 1024 * 1024,
    Math.floor(device.limits.maxStorageBufferBindingSize / 8));
  const maxChannels = Math.max(8,
    Math.floor(maxElements / plane / 8) * 8);
  const output = new Float64Array(outputChannels * plane);
  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
  for (let base = 0; base < outputChannels; base += maxChannels) {
    const countChannels = Math.min(maxChannels, outputChannels - base);
    const count = countChannels * plane;
    if (count * 8 > device.limits.maxStorageBufferBindingSize)
      throw new Error('Pointwise output plane exceeds WebGPU storage limit');
    const groupsX = Math.ceil(plane / 8);
    const groupsY = Math.ceil(countChannels / 8);
    if (groupsX > device.limits.maxComputeWorkgroupsPerDimension ||
        groupsY > device.limits.maxComputeWorkgroupsPerDimension)
      throw new Error('Pointwise convolution exceeds WebGPU dispatch limits');
    const paramBuffer = upload(device, new Uint32Array([
      channels, plane, outputChannels, base, countChannels]),
    GPUBufferUsage.STORAGE);
    const resultBuffer = device.createBuffer({size: count * 8, usage: storage});
    const readBuffer = device.createBuffer({size: count * 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
    const group = device.createBindGroup({layout: pipeline.getBindGroupLayout(0),
      entries: [inputBuffer, weightBuffer, resultBuffer, paramBuffer].map((buffer, binding) =>
        ({binding, resource: {buffer}}))});
    device.pushErrorScope('validation');
    const commands = device.createCommandEncoder();
    const pass = commands.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(groupsX, groupsY);
    pass.end();
    commands.copyBufferToBuffer(resultBuffer, 0, readBuffer, 0, count * 8);
    device.queue.submit([commands.finish()]);
    const error = await device.popErrorScope();
    if (error) throw new Error(`Tiled WebGPU pointwise rejected: ${error.message}`);
    await readBuffer.mapAsync(GPUMapMode.READ);
    const words = new Uint32Array(readBuffer.getMappedRange());
    for (let i = 0; i < count; i++) {
      const value = (words[2 * i + 1] | 0) * 4294967296 + words[2 * i];
      if (!Number.isSafeInteger(value))
        throw new Error('Tiled pointwise sum exceeds exact FP64 range');
      output[base * plane + i] = value;
    }
    readBuffer.unmap();
    for (const buffer of [paramBuffer, resultBuffer, readBuffer])
      buffer.destroy();
  }
  return output;
}

export async function exactConvWebGpu(input, weights, channels, height, width,
                                      outputChannels, kind, existingDevice = null,
                                      options = {}) {
  if (!navigator.gpu) throw new Error('WebGPU unavailable');
  const device = existingDevice || await (await navigator.gpu.requestAdapter())?.requestDevice();
  if (!device) throw new Error('WebGPU device unavailable');
  const kernel = kind === 'pointwise' ? 1 : kind === 'depthwise' ? 3 : options.kernel;
  const stride = kind === 'general' ? options.stride : 1;
  const padding = kind === 'general' ? options.padding : kind === 'depthwise' ? 1 : 0;
  const outH = Math.floor((height + 2 * padding - kernel) / stride) + 1;
  const outW = Math.floor((width + 2 * padding - kernel) / stride) + 1;
  const elements = outputChannels * outH * outW;
  const maximumChunk = Math.min(8 * 1024 * 1024,
    Math.floor(device.limits.maxStorageBufferBindingSize / 8));
  if (maximumChunk <= 0) throw new Error('Invalid WebGPU storage buffer limit');
  let pipeline = PIPELINES.get(device);
  if (!pipeline) {
    const module = device.createShaderModule({code: SHADER});
    pipeline = device.createComputePipeline({layout: 'auto',
      compute: {module, entryPoint: 'main'}});
    PIPELINES.set(device, pipeline);
  }
  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
  const inputBuffer = upload(device, input, GPUBufferUsage.STORAGE);
  let weightsForDevice = WEIGHT_BUFFERS.get(device);
  if (!weightsForDevice) {
    weightsForDevice = new Map();
    WEIGHT_BUFFERS.set(device, weightsForDevice);
  }
  let weightBuffer = weightsForDevice.get(weights);
  if (!weightBuffer) {
    weightBuffer = upload(device, weights, GPUBufferUsage.STORAGE);
    weightsForDevice.set(weights, weightBuffer);
  }
  if (kind === 'pointwise') {
    const result = await tiledPointwiseWebGpu(device, inputBuffer,
      weightBuffer, channels, height * width, outputChannels);
    inputBuffer.destroy();
    return result;
  }
  const output = new Float64Array(elements);
  for (let base = 0; base < elements; base += maximumChunk) {
    const count = Math.min(maximumChunk, elements - base);
    const workgroupCount = Math.ceil(count / 64);
    const groupsX = Math.min(workgroupCount,
      device.limits.maxComputeWorkgroupsPerDimension);
    const groupsY = Math.ceil(workgroupCount / groupsX);
    if (groupsY > device.limits.maxComputeWorkgroupsPerDimension)
      throw new Error('Portable convolution exceeds WebGPU dispatch limits');
    const paramBuffer = upload(device, new Uint32Array([
      channels, height, width, outputChannels,
      kind === 'pointwise' ? 0 : kind === 'depthwise' ? 1 : 2,
      kernel, stride, padding, outH, outW, groupsX, base, count]),
    GPUBufferUsage.STORAGE);
    const resultBuffer = device.createBuffer({size: count * 8, usage: storage});
    const readBuffer = device.createBuffer({size: count * 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
    const group = device.createBindGroup({layout: pipeline.getBindGroupLayout(0),
      entries: [inputBuffer, weightBuffer, resultBuffer, paramBuffer].map((buffer, binding) =>
        ({binding, resource: {buffer}}))});
    device.pushErrorScope('validation');
    const commands = device.createCommandEncoder();
    const pass = commands.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(groupsX, groupsY);
    pass.end();
    commands.copyBufferToBuffer(resultBuffer, 0, readBuffer, 0, count * 8);
    device.queue.submit([commands.finish()]);
    const validationError = await device.popErrorScope();
    if (validationError)
      throw new Error(`WebGPU convolution rejected: ${validationError.message}`);
    await readBuffer.mapAsync(GPUMapMode.READ);
    const words = new Uint32Array(readBuffer.getMappedRange());
    for (let i = 0; i < count; i++) {
      // Fixed-PTQ bounds keep each sum within JavaScript's exact integer range.
      const value = (words[2 * i + 1] | 0) * 4294967296 + words[2 * i];
      if (!Number.isSafeInteger(value))
        throw new Error('INT64 result exceeds exact FP64 range');
      output[base + i] = value;
    }
    readBuffer.unmap();
    for (const buffer of [paramBuffer, resultBuffer, readBuffer])
      buffer.destroy();
  }
  inputBuffer.destroy();
  return output;
}

// Experimental GPU-resident path for one unchunked layer. The caller owns the
// returned INT64-pair buffer and must destroy it after GPU dequantization.
export async function exactConvWebGpuResident(inputBuffer, weights, channels,
                                              height, width, outputChannels,
                                              kind, device, options = {}) {
  const kernel = kind === 'pointwise' ? 1 : kind === 'depthwise' ? 3 : options.kernel;
  const stride = kind === 'general' ? options.stride : 1;
  const padding = kind === 'general' ? options.padding : kind === 'depthwise' ? 1 : 0;
  const outH = Math.floor((height + 2 * padding - kernel) / stride) + 1;
  const outW = Math.floor((width + 2 * padding - kernel) / stride) + 1;
  const count = outputChannels * outH * outW;
  if (count * 8 > device.limits.maxStorageBufferBindingSize)
    throw new Error('GPU-resident convolution needs output tiling');
  let weightBuffers = WEIGHT_BUFFERS.get(device);
  if (!weightBuffers) {
    weightBuffers = new Map();
    WEIGHT_BUFFERS.set(device, weightBuffers);
  }
  let weightBuffer = weightBuffers.get(weights);
  if (!weightBuffer) {
    weightBuffer = upload(device, weights, GPUBufferUsage.STORAGE);
    weightBuffers.set(weights, weightBuffer);
  }
  const pointwise = kind === 'pointwise';
  const cache = pointwise ? POINTWISE_PIPELINES : PIPELINES;
  let pipeline = cache.get(device);
  if (!pipeline) {
    const module = device.createShaderModule({code: pointwise ?
      POINTWISE_SHADER : SHADER});
    pipeline = device.createComputePipeline({layout: 'auto',
      compute: {module, entryPoint: 'main'}});
    cache.set(device, pipeline);
  }
  const outputBuffer = device.createBuffer({size: count * 8,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});
  let params, groupsX, groupsY;
  if (pointwise) {
    params = new Uint32Array([channels, height * width, outputChannels,
      0, outputChannels]);
    groupsX = Math.ceil(height * width / 8);
    groupsY = Math.ceil(outputChannels / 8);
  } else {
    const workgroups = Math.ceil(count / 64);
    groupsX = Math.min(workgroups,
      device.limits.maxComputeWorkgroupsPerDimension);
    groupsY = Math.ceil(workgroups / groupsX);
    params = new Uint32Array([channels, height, width, outputChannels,
      kind === 'depthwise' ? 1 : 2, kernel, stride, padding,
      outH, outW, groupsX, 0, count]);
  }
  if (groupsX > device.limits.maxComputeWorkgroupsPerDimension ||
      groupsY > device.limits.maxComputeWorkgroupsPerDimension)
    throw new Error('GPU-resident convolution exceeds dispatch limit');
  const paramsBuffer = upload(device, params, GPUBufferUsage.STORAGE);
  const group = device.createBindGroup({layout: pipeline.getBindGroupLayout(0),
    entries: [inputBuffer, weightBuffer, outputBuffer, paramsBuffer]
      .map((buffer, binding) => ({binding, resource: {buffer}}))});
  const commands = device.createCommandEncoder();
  const pass = commands.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(groupsX, groupsY);
  pass.end();
  device.queue.submit([commands.finish()]);
  paramsBuffer.destroy();
  return {buffer: outputBuffer, height: outH, width: outW,
    count};
}
