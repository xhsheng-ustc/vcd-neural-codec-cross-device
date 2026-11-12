import {FIXED_MATH_WGSL} from './portable_fixed_dequant_gpu.js';

// Exact W16 pointwise convolution and Q20 epilogue in one dispatch.
// The INT64 pair remains in registers and never reaches a storage buffer.
const COMMON = `
@group(0) @binding(0) var<storage, read> x: array<i32>;
@group(0) @binding(1) var<storage, read> w: array<i32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;
@group(0) @binding(4) var<storage, read> multiplier: array<u32>;
@group(0) @binding(5) var<storage, read> right_shift: array<u32>;
@group(0) @binding(6) var<storage, read> bias_q: array<i32>;

fn product64(a: i32, b: i32) -> vec2<u32> {
  let value = a * b;
  return vec2<u32>(bitcast<u32>(value),
    select(0u, 0xffffffffu, value < 0));
}

fn add64(acc: vec2<u32>, value: vec2<u32>) -> vec2<u32> {
  let lo = acc.x + value.x;
  return vec2<u32>(lo, acc.y + value.y + u32(lo < acc.x));
}

${FIXED_MATH_WGSL}`;

const SOURCE = `${COMMON}

var<workgroup> tile_x: array<i32, 256>;
var<workgroup> tile_w: array<i32, 512>;

@compute @workgroup_size(8, 16)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local: vec3<u32>) {
  let channels = params[0];
  let plane = params[1];
  let output_channels = params[2];
  let output_channel = group.y * 16u + local.y;
  let spatial = group.x * 8u + local.x;
  let valid = output_channel < output_channels && spatial < plane;
  let lane = local.y * 8u + local.x;
  var acc = vec2<u32>(0u, 0u);
  for (var base = 0u; base < channels; base = base + 32u) {
    for (var slot = lane; slot < 256u; slot = slot + 128u) {
      let kx = base + slot / 8u;
      let px = group.x * 8u + slot % 8u;
      var xv = 0;
      if (kx < channels && px < plane) { xv = x[kx * plane + px]; }
      tile_x[slot] = xv;
    }
    for (var slot = lane; slot < 512u; slot = slot + 128u) {
      let kw = base + slot % 32u;
      let cw = group.y * 16u + slot / 32u;
      var wv = 0;
      if (kw < channels && cw < output_channels) {
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
    output[output_channel * plane + spatial] = fixed_value(acc,
      multiplier[output_channel], right_shift[output_channel],
      bias_q[output_channel]);
  }
}`;

const SPATIAL_SOURCE = `${COMMON}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let channels = params[0];
  let height = params[1];
  let width = params[2];
  let output_channels = params[3];
  let kind = params[4];
  let kernel = params[5];
  let stride = params[6];
  let padding = params[7];
  let output_height = params[8];
  let output_width = params[9];
  let plane = output_height * output_width;
  let index = id.x + id.y * params[10] * 64u;
  if (index >= params[11]) { return; }
  let channel = index / plane;
  let location = index % plane;
  let input_plane = height * width;
  let y = i32(location / output_width);
  let xpos = i32(location % output_width);
  var acc = vec2<u32>(0u, 0u);
  if (kind == 1u) {
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
  output[index] = fixed_value(acc, multiplier[channel],
    right_shift[channel], bias_q[channel]);
}`;

const PIPELINES = new WeakMap();
const WEIGHTS = new WeakMap();
const FIXED = new WeakMap();

function upload(device, values) {
  const buffer = device.createBuffer({size: values.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});
  device.queue.writeBuffer(buffer, 0, values);
  return buffer;
}

function cached(device, cache, key, create) {
  let entries = cache.get(device);
  if (!entries) { entries = new Map(); cache.set(device, entries); }
  if (!entries.has(key)) entries.set(key, create());
  return entries.get(key);
}

export function fusedPointwiseGpu(device, input, weights, fixed,
                                  channels, height, width, outputChannels) {
  const plane = height * width;
  const count = outputChannels * plane;
  if (count * 4 > device.limits.maxStorageBufferBindingSize)
    throw new Error('Fused pointwise convolution needs output tiling');
  const pipeline = cached(device, PIPELINES, 'pointwise', () => {
    const module = device.createShaderModule({code: SOURCE});
    return device.createComputePipeline({layout: 'auto',
      compute: {module, entryPoint: 'main'}});
  });
  const weight = cached(device, WEIGHTS, weights,
    () => upload(device, weights));
  const params = cached(device, FIXED, fixed, () => ({
    multiplier: upload(device, fixed.multiplier),
    rightShift: upload(device, fixed.rightShift),
    biasQ: upload(device, fixed.biasQ),
  }));
  const output = device.createBuffer({size: count * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});
  const shape = upload(device, new Uint32Array([channels, plane,
    outputChannels, 0]));
  const groupsX = Math.ceil(plane / 8);
  const groupsY = Math.ceil(outputChannels / 16);
  if (groupsX > device.limits.maxComputeWorkgroupsPerDimension ||
      groupsY > device.limits.maxComputeWorkgroupsPerDimension)
    throw new Error('Fused pointwise convolution exceeds dispatch limit');
  const group = device.createBindGroup({layout: pipeline.getBindGroupLayout(0),
    entries: [input, weight, output, shape, params.multiplier,
      params.rightShift, params.biasQ].map((buffer, binding) =>
      ({binding, resource: {buffer}}))});
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(groupsX, groupsY);
  pass.end();
  device.queue.submit([encoder.finish()]);
  shape.destroy();
  return {buffer: output, height, width, count};
}

export function fusedSpatialGpu(device, input, weights, fixed,
                                 channels, height, width, outputChannels,
                                 kind, options = {}) {
  if (kind !== 'depthwise' && kind !== 'general')
    throw new Error('Expected a spatial convolution');
  const kernel = kind === 'depthwise' ? 3 : options.kernel;
  const stride = kind === 'depthwise' ? 1 : options.stride;
  const padding = kind === 'depthwise' ? 1 : options.padding;
  const outH = Math.floor((height + 2 * padding - kernel) / stride) + 1;
  const outW = Math.floor((width + 2 * padding - kernel) / stride) + 1;
  const count = outputChannels * outH * outW;
  if (count * 4 > device.limits.maxStorageBufferBindingSize)
    throw new Error('Fused spatial convolution needs output tiling');
  const pipeline = cached(device, PIPELINES, 'spatial', () => {
    const module = device.createShaderModule({code: SPATIAL_SOURCE});
    return device.createComputePipeline({layout: 'auto',
      compute: {module, entryPoint: 'main'}});
  });
  const weight = cached(device, WEIGHTS, weights,
    () => upload(device, weights));
  const fixedParams = cached(device, FIXED, fixed, () => ({
    multiplier: upload(device, fixed.multiplier),
    rightShift: upload(device, fixed.rightShift),
    biasQ: upload(device, fixed.biasQ),
  }));
  const output = device.createBuffer({size: count * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});
  const groupsX = Math.min(Math.ceil(count / 64),
    device.limits.maxComputeWorkgroupsPerDimension);
  const groupsY = Math.ceil(Math.ceil(count / 64) / groupsX);
  if (groupsY > device.limits.maxComputeWorkgroupsPerDimension)
    throw new Error('Fused spatial convolution exceeds dispatch limit');
  const shape = upload(device, new Uint32Array([channels, height, width,
    outputChannels, kind === 'depthwise' ? 1 : 2, kernel, stride, padding,
    outH, outW, groupsX, count]));
  const group = device.createBindGroup({layout: pipeline.getBindGroupLayout(0),
    entries: [input, weight, output, shape, fixedParams.multiplier,
      fixedParams.rightShift, fixedParams.biasQ].map((buffer, binding) =>
      ({binding, resource: {buffer}}))});
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(groupsX, groupsY);
  pass.end();
  device.queue.submit([encoder.finish()]);
  shape.destroy();
  return {buffer: output, height: outH, width: outW, count};
}

export function releaseFusedPointwiseGpu(device) {
  const weights = WEIGHTS.get(device);
  if (weights) for (const buffer of weights.values()) buffer.destroy();
  const fixed = FIXED.get(device);
  if (fixed) for (const item of fixed.values())
    for (const buffer of Object.values(item)) buffer.destroy();
  WEIGHTS.delete(device);
  FIXED.delete(device);
  PIPELINES.delete(device);
}
