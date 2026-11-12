const HEADER = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: array<u32, 8>;
fn grid(value: f32) -> i32 {
  let scaled = value * 1048576.0;
  let lower = floor(scaled);
  let fraction = scaled - lower;
  let base = i32(lower);
  return base + i32(fraction > 0.5 ||
    (fraction == 0.5 && (base & 1) != 0));
}
fn divide_even(value: i32, shift: u32) -> i32 {
  let negative = value < 0;
  let positive = u32(select(value, -value, negative));
  let quotient = positive >> shift;
  let remainder = positive & ((1u << shift) - 1u);
  let half = 1u << (shift - 1u);
  let rounded = i32(quotient + u32(remainder > half ||
    (remainder == half && (quotient & 1u) != 0u)));
  return select(rounded, -rounded, negative);
}
`;
const HALF_SOURCE = HEADER + `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x + id.y * params[5] * 64u;
  if (i >= params[0]) { return; }
  let height = params[2]; let width = params[3];
  let out_width = width / 2u;
  let out_plane = (height / 2u) * out_width;
  let channel = i / out_plane;
  let location = i % out_plane;
  let row = (location / out_width) * 2u;
  let col = (location % out_width) * 2u;
  let base = channel * height * width + row * width + col;
  let sum = grid(input[base]) + grid(input[base + 1u]) +
    grid(input[base + width]) + grid(input[base + width + 1u]);
  output[i] = f32(divide_even(sum, 2u)) * 0.00000095367431640625;
}`;
const DOUBLE_SOURCE = HEADER + `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x + id.y * params[5] * 64u;
  if (i >= params[0]) { return; }
  let height = params[2]; let width = params[3];
  let out_width = width * 2u;
  let out_plane = height * width * 4u;
  let channel = i / out_plane;
  let location = i % out_plane;
  let row = location / out_width;
  let col = location % out_width;
  let base_y = row / 2u;
  let base_x = col / 2u;
  let y0 = select(base_y, u32(max(i32(base_y) - 1, 0)), row % 2u == 0u);
  let y1 = select(min(base_y + 1u, height - 1u), base_y, row % 2u == 0u);
  let x0 = select(base_x, u32(max(i32(base_x) - 1, 0)), col % 2u == 0u);
  let x1 = select(min(base_x + 1u, width - 1u), base_x, col % 2u == 0u);
  let wy0 = select(3, 1, row % 2u == 0u);
  let wx0 = select(3, 1, col % 2u == 0u);
  let wy1 = 4 - wy0; let wx1 = 4 - wx0;
  let plane_base = channel * height * width;
  let sum = grid(input[plane_base + u32(y0) * width + u32(x0)]) * wy0 * wx0 +
    grid(input[plane_base + u32(y0) * width + u32(x1)]) * wy0 * wx1 +
    grid(input[plane_base + u32(y1) * width + u32(x0)]) * wy1 * wx0 +
    grid(input[plane_base + u32(y1) * width + u32(x1)]) * wy1 * wx1;
  output[i] = f32(divide_even(sum, 4u)) * 0.00000095367431640625;
}`;
const CACHE = new WeakMap();

export async function gpuResample(device, data, channels, height,
                                   width, mode) {
  let pipelines = CACHE.get(device);
  if (!pipelines) {
    pipelines = [HALF_SOURCE, DOUBLE_SOURCE].map(code =>
      device.createComputePipeline({layout: 'auto', compute: {
        module: device.createShaderModule({code}), entryPoint: 'main'}}));
    CACHE.set(device, pipelines);
  }
  const index = mode === 'half' ? 0 : mode === 'double' ? 1 : -1;
  if (index < 0) throw new Error('Unknown fixed resample mode');
  if (index === 0 && (height % 2 || width % 2))
    throw new Error('Half resampling needs even dimensions');
  const count = channels * height * width * (index === 0 ? 1 / 4 : 4);
  const source = device.createBuffer({size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});
  device.queue.writeBuffer(source, 0, data);
  const output = device.createBuffer({size: count * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});
  const groupsX = Math.min(Math.ceil(count / 64),
    device.limits.maxComputeWorkgroupsPerDimension);
  const groupsY = Math.ceil(Math.ceil(count / 64) / groupsX);
  const params = device.createBuffer({size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});
  device.queue.writeBuffer(params, 0, new Uint32Array([
    count, channels, height, width, 0, groupsX, 0, 0]));
  const pipeline = pipelines[index];
  const bind = device.createBindGroup({layout: pipeline.getBindGroupLayout(0),
    entries: [source, output, params].map((buffer, binding) =>
      ({binding, resource: {buffer}}))});
  const commands = device.createCommandEncoder();
  const pass = commands.beginComputePass();
  pass.setPipeline(pipeline); pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(groupsX, groupsY); pass.end();
  device.queue.submit([commands.finish()]);
  const read = device.createBuffer({size: count * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
  const copy = device.createCommandEncoder();
  copy.copyBufferToBuffer(output, 0, read, 0, count * 4);
  device.queue.submit([copy.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(new Float32Array(read.getMappedRange()));
  read.unmap();
  for (const buffer of [source, output, params, read]) buffer.destroy();
  return result;
}

export function gpuResampleResident(device, inputBuffer, channels,
                                     height, width, mode) {
  let pipelines = CACHE.get(device);
  if (!pipelines) {
    pipelines = [HALF_SOURCE, DOUBLE_SOURCE].map(code =>
      device.createComputePipeline({layout: 'auto', compute: {
        module: device.createShaderModule({code}), entryPoint: 'main'}}));
    CACHE.set(device, pipelines);
  }
  const index = mode === 'half' ? 0 : mode === 'double' ? 1 : -1;
  if (index < 0) throw new Error('Unknown fixed resample mode');
  if (index === 0 && (height % 2 || width % 2))
    throw new Error('Half resampling needs even dimensions');
  const count = channels * height * width * (index === 0 ? 1 / 4 : 4);
  const groupsX = Math.min(Math.ceil(count / 64),
    device.limits.maxComputeWorkgroupsPerDimension);
  const groupsY = Math.ceil(Math.ceil(count / 64) / groupsX);
  const output = device.createBuffer({size: count * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});
  const params = device.createBuffer({size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});
  device.queue.writeBuffer(params, 0, new Uint32Array([
    count, channels, height, width, 0, groupsX, 0, 0]));
  const bind = device.createBindGroup({layout: pipelines[index].getBindGroupLayout(0),
    entries: [inputBuffer, output, params].map((buffer, binding) =>
      ({binding, resource: {buffer}}))});
  const commands = device.createCommandEncoder();
  const pass = commands.beginComputePass();
  pass.setPipeline(pipelines[index]); pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(groupsX, groupsY); pass.end();
  device.queue.submit([commands.finish()]);
  params.destroy();
  return output;
}
