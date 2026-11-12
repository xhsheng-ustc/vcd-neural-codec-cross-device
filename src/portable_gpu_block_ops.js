const SILU_SOURCE = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> table: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x + id.y * params.y * 64u;
  if (i >= params.x) { return; }
  let scaled = input[i] * 16384.0;
  let lower = floor(scaled);
  let fraction = scaled - lower;
  let integer = i32(lower);
  let increase = fraction > 0.5 ||
    (fraction == 0.5 && (integer & 1) != 0);
  let q = clamp(integer + i32(increase), -131072, 131071);
  output[i] = table[u32(q + 131072)];
}`;
const ADD_SOURCE = `
@group(0) @binding(0) var<storage, read> first: array<f32>;
@group(0) @binding(1) var<storage, read> second: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x + id.y * params.y * 64u;
  if (i >= params.x) { return; }
  output[i] = first[i] + second[i];
}`;
const FOLD_SOURCE = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x + id.y * params.y * 64u;
  if (i >= params.x) { return; }
  output[i] = input[i] + input[i + params.x];
}`;
const PIXEL_SOURCE = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: array<u32, 8>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x + id.y * params[5] * 64u;
  if (i >= params[4]) { return; }
  let height = params[1];
  let width = params[2];
  let factor = params[3];
  let out_plane = height * width * factor * factor;
  let channel = i / out_plane;
  let spatial = i % out_plane;
  let oy = spatial / (width * factor);
  let ox = spatial % (width * factor);
  let sub_y = oy % factor;
  let sub_x = ox % factor;
  let source_channel = (channel * factor + sub_y) * factor + sub_x;
  let source_spatial = (oy / factor) * width + ox / factor;
  output[i] = input[source_channel * height * width + source_spatial];
}`;
const MULTIPLY_SOURCE = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> quant: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x + id.y * params.z * 64u;
  if (i >= params.x) { return; }
  output[i] = input[i] * quant[i / params.y];
}`;

const CONCAT_SOURCE = `
@group(0) @binding(0) var<storage, read> first: array<f32>;
@group(0) @binding(1) var<storage, read> second: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x + id.y * params.z * 64u;
  if (i >= params.y) { return; }
  if (i < params.x) { output[i] = first[i]; }
  else { output[i] = second[i - params.x]; }
}`;

const CACHE = new WeakMap();
const EXTRA_PIPELINES = new WeakMap();
function resources(device, siluTable) {
  let value = CACHE.get(device);
  if (!value) {
    const pipelines = [SILU_SOURCE, ADD_SOURCE, FOLD_SOURCE].map(code =>
      device.createComputePipeline({layout: 'auto', compute: {
        module: device.createShaderModule({code}), entryPoint: 'main'}}));
    const table = device.createBuffer({size: siluTable.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});
    device.queue.writeBuffer(table, 0, siluTable);
    value = {pipelines, table};
    CACHE.set(device, value);
  }
  return value;
}

function dispatch(device, pipeline, buffers, count) {
  const groupsX = Math.min(Math.ceil(count / 64),
    device.limits.maxComputeWorkgroupsPerDimension);
  const groupsY = Math.ceil(Math.ceil(count / 64) / groupsX);
  if (groupsY > device.limits.maxComputeWorkgroupsPerDimension)
    throw new Error('GPU block operation exceeds dispatch limit');
  const params = device.createBuffer({size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});
  device.queue.writeBuffer(params, 0, new Uint32Array([count, groupsX, 0, 0]));
  const output = device.createBuffer({size: count * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});
  const bind = device.createBindGroup({layout: pipeline.getBindGroupLayout(0),
    entries: [...buffers, output, params].map((buffer, binding) =>
      ({binding, resource: {buffer}}))});
  const commands = device.createCommandEncoder();
  const pass = commands.beginComputePass();
  pass.setPipeline(pipeline); pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(groupsX, groupsY);
  pass.end();
  device.queue.submit([commands.finish()]);
  params.destroy();
  return output;
}

export function gpuSilu(device, input, count, siluTable) {
  const {pipelines, table} = resources(device, siluTable);
  return dispatch(device, pipelines[0], [input, table], count);
}
export function gpuAdd(device, first, second, count, siluTable) {
  const {pipelines} = resources(device, siluTable);
  return dispatch(device, pipelines[1], [first, second], count);
}
export function gpuFold(device, input, count, siluTable) {
  const {pipelines} = resources(device, siluTable);
  return dispatch(device, pipelines[2], [input], count);
}

export function gpuPixelShuffle(device, input, outputChannels, height,
                                width, factor) {
  let pair = EXTRA_PIPELINES.get(device);
  if (!pair) {
    pair = [PIXEL_SOURCE, MULTIPLY_SOURCE, CONCAT_SOURCE].map(code =>
      device.createComputePipeline({layout: 'auto', compute: {
        module: device.createShaderModule({code}), entryPoint: 'main'}}));
    EXTRA_PIPELINES.set(device, pair);
  }
  const count = outputChannels * height * width * factor * factor;
  const groupsX = Math.min(Math.ceil(count / 64),
    device.limits.maxComputeWorkgroupsPerDimension);
  const groupsY = Math.ceil(Math.ceil(count / 64) / groupsX);
  const params = uploadParams(device, new Uint32Array([
    outputChannels, height, width, factor, count, groupsX, 0, 0]));
  const output = submitExtra(device, pair[0], [input, params], count,
    groupsX, groupsY);
  params.destroy();
  return output;
}

export function gpuChannelMultiply(device, input, count, plane, quant) {
  let pair = EXTRA_PIPELINES.get(device);
  if (!pair) {
    pair = [PIXEL_SOURCE, MULTIPLY_SOURCE, CONCAT_SOURCE].map(code =>
      device.createComputePipeline({layout: 'auto', compute: {
        module: device.createShaderModule({code}), entryPoint: 'main'}}));
    EXTRA_PIPELINES.set(device, pair);
  }
  const groupsX = Math.min(Math.ceil(count / 64),
    device.limits.maxComputeWorkgroupsPerDimension);
  const groupsY = Math.ceil(Math.ceil(count / 64) / groupsX);
  const q = device.createBuffer({size: quant.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});
  device.queue.writeBuffer(q, 0, quant);
  const params = uploadParams(device,
    new Uint32Array([count, plane, groupsX, 0]));
  const output = submitExtra(device, pair[1], [input, q, params], count,
    groupsX, groupsY);
  q.destroy(); params.destroy();
  return output;
}

export function gpuConcat(device, first, firstCount, second, secondCount) {
  let pair = EXTRA_PIPELINES.get(device);
  if (!pair) {
    pair = [PIXEL_SOURCE, MULTIPLY_SOURCE, CONCAT_SOURCE].map(code =>
      device.createComputePipeline({layout: 'auto', compute: {
        module: device.createShaderModule({code}), entryPoint: 'main'}}));
    EXTRA_PIPELINES.set(device, pair);
  }
  const count = firstCount + secondCount;
  const groupsX = Math.min(Math.ceil(count / 64),
    device.limits.maxComputeWorkgroupsPerDimension);
  const groupsY = Math.ceil(Math.ceil(count / 64) / groupsX);
  const params = uploadParams(device,
    new Uint32Array([firstCount, count, groupsX, 0]));
  const output = submitExtra(device, pair[2], [first, second, params],
    count, groupsX, groupsY);
  params.destroy();
  return output;
}

function uploadParams(device, values) {
  const params = device.createBuffer({size: values.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});
  device.queue.writeBuffer(params, 0, values);
  return params;
}

function submitExtra(device, pipeline, inputs, count, groupsX, groupsY) {
  if (groupsY > device.limits.maxComputeWorkgroupsPerDimension)
    throw new Error('GPU block operation exceeds dispatch limit');
  const output = device.createBuffer({size: count * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});
  const bind = device.createBindGroup({layout: pipeline.getBindGroupLayout(0),
    entries: [...inputs.slice(0, -1), output, inputs.at(-1)]
      .map((buffer, binding) => ({binding, resource: {buffer}}))});
  const commands = device.createCommandEncoder();
  const pass = commands.beginComputePass();
  pass.setPipeline(pipeline); pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(groupsX, groupsY);
  pass.end();
  device.queue.submit([commands.finish()]);
  return output;
}
