// Experimental prepacked W16 dot-product kernel. Not used by the codec.
const SOURCE = `
@group(0) @binding(0) var<storage, read> x_lo: array<u32>;
@group(0) @binding(1) var<storage, read> x_hi: array<u32>;
@group(0) @binding(2) var<storage, read> w_lo: array<u32>;
@group(0) @binding(3) var<storage, read> w_hi: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<vec2<u32>>;
@group(0) @binding(5) var<storage, read> params: array<u32>;

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
  if (valid) { output[output_channel * plane + spatial] = acc; }
}`;

const PIPELINES = new WeakMap();
const PACK_PIPELINES = new WeakMap();

const PACK_SOURCE = `
@group(0) @binding(0) var<storage, read> input: array<i32>;
@group(0) @binding(1) var<storage, read_write> low: array<u32>;
@group(0) @binding(2) var<storage, read_write> high: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * params.w * 64u;
  if (index >= params.z * params.y) { return; }
  let quartet = index / params.y;
  let spatial = index % params.y;
  var lo = 0u;
  var hi = 0u;
  for (var j = 0u; j < 4u; j = j + 1u) {
    let channel = quartet * 4u + j;
    var value = 0;
    if (channel < params.x) {
      value = input[channel * params.y + spatial];
    }
    lo = lo | ((bitcast<u32>(value) & 255u) << (8u * j));
    hi = hi | ((bitcast<u32>(value >> 8) & 255u) << (8u * j));
  }
  low[index] = lo;
  high[index] = hi;
}`;

export function packW16Gpu(device, input, channels, plane) {
  let pipeline = PACK_PIPELINES.get(device);
  if (!pipeline) {
    const module = device.createShaderModule({code: PACK_SOURCE});
    pipeline = device.createComputePipeline({layout: 'auto',
      compute: {module, entryPoint: 'main'}});
    PACK_PIPELINES.set(device, pipeline);
  }
  const quartets = Math.ceil(channels / 4);
  const count = quartets * plane;
  const makeOutput = () => device.createBuffer({size: count * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});
  const low = makeOutput(), high = makeOutput();
  const groupsX = Math.min(Math.ceil(count / 64),
    device.limits.maxComputeWorkgroupsPerDimension);
  const groupsY = Math.ceil(Math.ceil(count / 64) / groupsX);
  const params = device.createBuffer({size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});
  device.queue.writeBuffer(params, 0,
    new Uint32Array([channels, plane, quartets, groupsX]));
  const group = device.createBindGroup({layout: pipeline.getBindGroupLayout(0),
    entries: [input, low, high, params].map((buffer, binding) =>
      ({binding, resource: {buffer}}))});
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline); pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(groupsX, groupsY); pass.end();
  device.queue.submit([encoder.finish()]);
  params.destroy();
  return {low, high};
}

export function packedDotGpu(device, buffers, quartets, plane, outputChannels) {
  let pipeline = PIPELINES.get(device);
  if (!pipeline) {
    const module = device.createShaderModule({code: SOURCE});
    pipeline = device.createComputePipeline({layout: 'auto',
      compute: {module, entryPoint: 'main'}});
    PIPELINES.set(device, pipeline);
  }
  const params = device.createBuffer({size: 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});
  device.queue.writeBuffer(params, 0,
    new Uint32Array([quartets, plane, outputChannels, 0]));
  const output = device.createBuffer({size: outputChannels * plane * 8,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});
  const group = device.createBindGroup({layout: pipeline.getBindGroupLayout(0),
    entries: [...buffers, output, params].map((buffer, binding) =>
      ({binding, resource: {buffer}}))});
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(Math.ceil(plane / 8),
    Math.ceil(outputChannels / 16));
  pass.end();
  device.queue.submit([encoder.finish()]);
  params.destroy();
  return output;
}
