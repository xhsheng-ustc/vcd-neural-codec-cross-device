import {FIXED_MATH_WGSL} from './portable_fixed_dequant_gpu.js';

// Exact W16 pointwise convolution with prepacked bytes and a fused Q20 output.
const DOT_SOURCE = `
@group(0) @binding(0) var<storage, read> x_lo: array<u32>;
@group(0) @binding(1) var<storage, read> x_hi: array<u32>;
@group(0) @binding(2) var<storage, read> w_lo: array<u32>;
@group(0) @binding(3) var<storage, read> w_hi: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<storage, read> params: array<u32>;
@group(0) @binding(6) var<storage, read> fixed_params: array<vec4<i32>>;

fn add64(acc: vec2<u32>, value: vec2<u32>) -> vec2<u32> {
  let low = acc.x + value.x;
  return vec2<u32>(low, acc.y + value.y + u32(low < acc.x));
}

fn high_sum(packed: u32) -> i32 {
  return (bitcast<i32>(packed << 24u) >> 24) +
    (bitcast<i32>(packed << 16u) >> 24) +
    (bitcast<i32>(packed << 8u) >> 24) +
    (bitcast<i32>(packed) >> 24);
}

${FIXED_MATH_WGSL}

var<workgroup> tile_x_lo: array<u32, 64>;
var<workgroup> tile_x_hi: array<u32, 64>;
var<workgroup> tile_w_lo: array<u32, 128>;
var<workgroup> tile_w_hi: array<u32, 128>;

@compute @workgroup_size(8, 16)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local: vec3<u32>) {
  let quartets = params[0];
  let plane = params[1];
  let output_channels = params[2];
  let output_channel = group.y * 16u + local.y;
  let spatial = group.x * 8u + local.x;
  let valid = output_channel < output_channels && spatial < plane;
  let lane = local.y * 8u + local.x;
  var acc = vec2<u32>(0u, 0u);
  for (var base = 0u; base < quartets; base = base + 8u) {
    if (lane < 64u) {
      let q = base + lane / 8u;
      let p = group.x * 8u + lane % 8u;
      var lo = 0u;
      var hi = 0u;
      if (q < quartets && p < plane) {
        lo = x_lo[q * plane + p];
        hi = x_hi[q * plane + p];
      }
      tile_x_lo[lane] = lo;
      tile_x_hi[lane] = hi;
    }
    let q = base + lane % 8u;
    let c = group.y * 16u + lane / 8u;
    var lo = 0u;
    var hi = 0u;
    if (q < quartets && c < output_channels) {
      lo = w_lo[c * quartets + q];
      hi = w_hi[c * quartets + q];
    }
    tile_w_lo[lane] = lo;
    tile_w_hi[lane] = hi;
    workgroupBarrier();
    if (valid) {
      for (var k = 0u; k < 8u; k = k + 1u) {
        let xl = tile_x_lo[k * 8u + local.x];
        let xh = tile_x_hi[k * 8u + local.x];
        let wl = tile_w_lo[local.y * 8u + k];
        let wh = tile_w_hi[local.y * 8u + k];
        let ll = dot4U8Packed(xl, wl);
        let hh = dot4I8Packed(xh, wh);
        let cross = dot4I8Packed(xh, wl ^ 0x80808080u) +
          dot4I8Packed(xl ^ 0x80808080u, wh) +
          128 * (high_sum(xh) + high_sum(wh));
        let small = i32(ll) + cross * 256;
        let small_pair = vec2<u32>(bitcast<u32>(small),
          select(0u, 0xffffffffu, small < 0));
        let high_pair = vec2<u32>(bitcast<u32>(hh) << 16u,
          bitcast<u32>(hh >> 16));
        acc = add64(acc, add64(small_pair, high_pair));
      }
    }
    workgroupBarrier();
  }
  if (valid) {
    let fixed = fixed_params[output_channel];
    output[output_channel * plane + spatial] = fixed_value(acc,
      bitcast<u32>(fixed.x), bitcast<u32>(fixed.y), fixed.z);
  }
}`;

const QUANT_PACK_SOURCE = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> low: array<u32>;
@group(0) @binding(2) var<storage, read_write> high: array<u32>;
@group(0) @binding(3) var<uniform> params: array<u32, 8>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * params[5] * 64u;
  if (index >= params[2] * params[1]) { return; }
  let quartet = index / params[1];
  let spatial = index % params[1];
  let scale = bitcast<f32>(params[3]);
  let limit = f32(params[4]);
  var lo = 0u;
  var hi = 0u;
  for (var j = 0u; j < 4u; j = j + 1u) {
    let channel = quartet * 4u + j;
    var value = 0;
    if (channel < params[0]) {
      let rounded = round(input[channel * params[1] + spatial] * scale);
      value = i32(clamp(rounded, -limit, limit));
    }
    lo = lo | ((bitcast<u32>(value) & 255u) << (8u * j));
    hi = hi | ((bitcast<u32>(value >> 8) & 255u) << (8u * j));
  }
  low[index] = lo;
  high[index] = hi;
}`;

const DOT_PIPELINES = new WeakMap();
const QUANT_PIPELINES = new WeakMap();
const WEIGHT_BUFFERS = new WeakMap();
const FIXED_BUFFERS = new WeakMap();

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

function pipeline(device, cache, code) {
  let value = cache.get(device);
  if (!value) {
    const module = device.createShaderModule({code});
    value = device.createComputePipeline({layout: 'auto',
      compute: {module, entryPoint: 'main'}});
    cache.set(device, value);
  }
  return value;
}

function packedWeights(device, weights, channels, outputChannels) {
  return cached(device, WEIGHT_BUFFERS, weights, () => {
    const quartets = Math.ceil(channels / 4);
    const low = new Uint32Array(outputChannels * quartets);
    const high = new Uint32Array(outputChannels * quartets);
    for (let c = 0; c < outputChannels; c++)
      for (let q = 0; q < quartets; q++) {
        let lo = 0, hi = 0;
        for (let j = 0; j < 4; j++) {
          const value = 4 * q + j < channels
            ? weights[c * channels + 4 * q + j] : 0;
          lo |= (value & 255) << (8 * j);
          hi |= ((value >> 8) & 255) << (8 * j);
        }
        low[c * quartets + q] = lo;
        high[c * quartets + q] = hi;
      }
    return {low: upload(device, low), high: upload(device, high)};
  });
}

function packedFixed(device, fixed, outputChannels) {
  return cached(device, FIXED_BUFFERS, fixed, () => {
    const values = new Int32Array(outputChannels * 4);
    for (let c = 0; c < outputChannels; c++) {
      values[c * 4] = fixed.multiplier[c];
      values[c * 4 + 1] = fixed.rightShift[c];
      values[c * 4 + 2] = fixed.biasQ[c];
    }
    return upload(device, values);
  });
}

export function packedPointwiseGpu(device, input, weights, fixed,
                                   channels, height, width, outputChannels,
                                   step, limit) {
  const plane = height * width;
  const quartets = Math.ceil(channels / 4);
  const packedCount = quartets * plane;
  const count = outputChannels * plane;
  if (count * 4 > device.limits.maxStorageBufferBindingSize)
    throw new Error('Packed pointwise output needs tiling');
  const quant = pipeline(device, QUANT_PIPELINES, QUANT_PACK_SOURCE);
  const dot = pipeline(device, DOT_PIPELINES, DOT_SOURCE);
  const makePacked = () => device.createBuffer({size: packedCount * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});
  const low = makePacked(), high = makePacked();
  const groupsX = Math.min(Math.ceil(packedCount / 64),
    device.limits.maxComputeWorkgroupsPerDimension);
  const groupsY = Math.ceil(Math.ceil(packedCount / 64) / groupsX);
  const scaleBits = new Uint32Array(new Float32Array([1 / step]).buffer)[0];
  const quantParams = device.createBuffer({size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});
  device.queue.writeBuffer(quantParams, 0, new Uint32Array([
    channels, plane, quartets, scaleBits, limit, groupsX, 0, 0]));
  const quantGroup = device.createBindGroup({layout: quant.getBindGroupLayout(0),
    entries: [input, low, high, quantParams].map((buffer, binding) =>
      ({binding, resource: {buffer}}))});
  const quantCommands = device.createCommandEncoder();
  const quantPass = quantCommands.beginComputePass();
  quantPass.setPipeline(quant); quantPass.setBindGroup(0, quantGroup);
  quantPass.dispatchWorkgroups(groupsX, groupsY); quantPass.end();
  device.queue.submit([quantCommands.finish()]);
  quantParams.destroy();

  const weight = packedWeights(device, weights, channels, outputChannels);
  const fixedBuffer = packedFixed(device, fixed, outputChannels);
  const output = device.createBuffer({size: count * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});
  const params = upload(device, new Uint32Array([
    quartets, plane, outputChannels, 0]));
  const group = device.createBindGroup({layout: dot.getBindGroupLayout(0),
    entries: [low, high, weight.low, weight.high, output, params, fixedBuffer]
      .map((buffer, binding) => ({binding, resource: {buffer}}))});
  const commands = device.createCommandEncoder();
  const pass = commands.beginComputePass();
  pass.setPipeline(dot); pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(Math.ceil(plane / 8),
    Math.ceil(outputChannels / 16));
  pass.end();
  device.queue.submit([commands.finish()]);
  params.destroy();
  return {buffer: output, height, width, count,
    temporaries: [low, high]};
}

export function releasePackedPointwiseGpu(device) {
  for (const entry of WEIGHT_BUFFERS.get(device)?.values() ?? []) {
    entry.low.destroy(); entry.high.destroy();
  }
  for (const buffer of FIXED_BUFFERS.get(device)?.values() ?? [])
    buffer.destroy();
  WEIGHT_BUFFERS.delete(device); FIXED_BUFFERS.delete(device);
  DOT_PIPELINES.delete(device); QUANT_PIPELINES.delete(device);
}
