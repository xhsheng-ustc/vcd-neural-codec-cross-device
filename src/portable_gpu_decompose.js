const SOURCE = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> low_quant: array<f32>;
@group(0) @binding(2) var<storage, read> high_quant: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: array<u32, 8>;

fn grid(value: f32) -> i32 {
  let scaled = value * 1048576.0;
  let low = floor(scaled);
  let fraction = scaled - low;
  let base = i32(low);
  return base + i32(fraction > 0.5 ||
    (fraction == 0.5 && (base & 1) != 0));
}
fn signed64(value: i32) -> vec2<u32> {
  return vec2<u32>(bitcast<u32>(value),
    select(0u, 0xffffffffu, value < 0));
}
fn add64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  let lo = a.x + b.x;
  return vec2<u32>(lo, a.y + b.y + u32(lo < a.x));
}
fn neg64(value: vec2<u32>) -> vec2<u32> {
  let lo = (~value.x) + 1u;
  return vec2<u32>(lo, (~value.y) + u32(lo == 0u));
}
fn mul64x16(value: vec2<u32>, m: u32) -> vec2<u32> {
  let p0 = (value.x & 65535u) * m;
  let p1 = (value.x >> 16u) * m;
  let lo = p0 + (p1 << 16u);
  let hi = (p1 >> 16u) + u32(lo < p0) + value.y * m;
  return vec2<u32>(lo, hi);
}
fn mul64x32(value: vec2<u32>, coefficient: i32) -> vec2<u32> {
  let negative = (bitcast<i32>(value.y) < 0) != (coefficient < 0);
  let magnitude = select(value, neg64(value), bitcast<i32>(value.y) < 0);
  let m = u32(abs(coefficient));
  let lower = mul64x16(magnitude, m & 65535u);
  let upper = mul64x16(magnitude, m >> 16u);
  let shifted = vec2<u32>(upper.x << 16u,
    (upper.y << 16u) | (upper.x >> 16u));
  let result = add64(lower, shifted);
  return select(result, neg64(result), negative);
}
fn down(channel: u32, row: u32, col: u32,
        height: u32, width: u32) -> vec2<u32> {
  let base = channel * height * width + row * 2u * width + col * 2u;
  return add64(add64(signed64(grid(input[base])),
    signed64(grid(input[base + 1u]))),
    add64(signed64(grid(input[base + width])),
      signed64(grid(input[base + width + 1u]))));
}
fn round_shift26(words: vec2<u32>) -> i32 {
  let negative = bitcast<i32>(words.y) < 0;
  var magnitude = words;
  if (negative) {
    let lo = (~words.x) + 1u;
    magnitude = vec2<u32>(lo, (~words.y) + u32(lo == 0u));
  }
  let quotient = (magnitude.x >> 26u) | (magnitude.y << 6u);
  let remainder = magnitude.x & 67108863u;
  let rounded = i32(quotient + u32(remainder > 33554432u ||
    (remainder == 33554432u && (quotient & 1u) != 0u)));
  return select(rounded, -rounded, negative);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x + id.y * params[5] * 64u;
  if (i >= params[0]) { return; }
  let height = params[2]; let width = params[3];
  let plane = height * width;
  let channel = i / plane;
  let y = (i % plane) / width;
  let x = i % width;
  let half_height = height / 2u;
  let half_width = width / 2u;
  let base_y = y / 2u; let base_x = x / 2u;
  let y0 = select(base_y, u32(max(i32(base_y) - 1, 0)), y % 2u == 0u);
  let y1 = select(min(base_y + 1u, half_height - 1u), base_y, y % 2u == 0u);
  let x0 = select(base_x, u32(max(i32(base_x) - 1, 0)), x % 2u == 0u);
  let x1 = select(min(base_x + 1u, half_width - 1u), base_x, x % 2u == 0u);
  let wy0 = select(3, 1, y % 2u == 0u);
  let wx0 = select(3, 1, x % 2u == 0u);
  let wy1 = 4 - wy0; let wx1 = 4 - wx0;
  let low = add64(add64(
    mul64x32(down(channel, y0, x0, height, width), wy0 * wx0),
    mul64x32(down(channel, y0, x1, height, width), wy0 * wx1)),
    add64(
      mul64x32(down(channel, y1, x0, height, width), wy1 * wx0),
      mul64x32(down(channel, y1, x1, height, width), wy1 * wx1)));
  let q = signed64(grid(input[i]));
  let sixty_four_q = vec2<u32>(q.x << 6u,
    (q.y << 6u) | (q.x >> 26u));
  let high = add64(sixty_four_q, neg64(low));
  let quant_channel = select(channel, 0u, params[4] == 1u);
  let ql = grid(low_quant[quant_channel]);
  let qh = grid(high_quant[quant_channel]);
  let total = add64(mul64x32(low, ql), mul64x32(high, qh));
  output[i] = f32(round_shift26(total)) * 0.00000095367431640625;
}`;
const PIPELINES = new WeakMap();

export async function gpuFixedDecompose(device, input, lowQuant,
                                         highQuant, channels, height, width) {
  if (height % 2 || width % 2) throw new Error('Even shape required');
  if (lowQuant.length !== highQuant.length ||
      ![1, channels].includes(lowQuant.length))
    throw new Error('Invalid decompose quantizer shape');
  let pipeline = PIPELINES.get(device);
  if (!pipeline) {
    pipeline = device.createComputePipeline({layout: 'auto', compute: {
      module: device.createShaderModule({code: SOURCE}), entryPoint: 'main'}});
    PIPELINES.set(device, pipeline);
  }
  const upload = (values, usage) => {
    const buffer = device.createBuffer({size: values.byteLength,
      usage: usage | GPUBufferUsage.COPY_DST});
    device.queue.writeBuffer(buffer, 0, values);
    return buffer;
  };
  const count = channels * height * width;
  const groupsX = Math.min(Math.ceil(count / 64),
    device.limits.maxComputeWorkgroupsPerDimension);
  const groupsY = Math.ceil(Math.ceil(count / 64) / groupsX);
  const buffers = [upload(input, GPUBufferUsage.STORAGE),
    upload(lowQuant, GPUBufferUsage.STORAGE),
    upload(highQuant, GPUBufferUsage.STORAGE),
    device.createBuffer({size: count * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC}),
    upload(new Uint32Array([count, channels, height, width,
      lowQuant.length, groupsX, 0, 0]), GPUBufferUsage.UNIFORM)];
  const bind = device.createBindGroup({layout: pipeline.getBindGroupLayout(0),
    entries: buffers.map((buffer, binding) =>
      ({binding, resource: {buffer}}))});
  const commands = device.createCommandEncoder();
  const pass = commands.beginComputePass();
  pass.setPipeline(pipeline); pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(groupsX, groupsY); pass.end();
  device.queue.submit([commands.finish()]);
  const read = device.createBuffer({size: count * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
  const copy = device.createCommandEncoder();
  copy.copyBufferToBuffer(buffers[3], 0, read, 0, count * 4);
  device.queue.submit([copy.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(new Float32Array(read.getMappedRange()));
  read.unmap();
  for (const buffer of [...buffers, read]) buffer.destroy();
  return result;
}

export function gpuFixedDecomposeResident(device, inputBuffer, lowQuant,
                                           highQuant, channels, height, width) {
  if (height % 2 || width % 2) throw new Error('Even shape required');
  let pipeline = PIPELINES.get(device);
  if (!pipeline) {
    pipeline = device.createComputePipeline({layout: 'auto', compute: {
      module: device.createShaderModule({code: SOURCE}), entryPoint: 'main'}});
    PIPELINES.set(device, pipeline);
  }
  const upload = (values, usage) => {
    const buffer = device.createBuffer({size: values.byteLength,
      usage: usage | GPUBufferUsage.COPY_DST});
    device.queue.writeBuffer(buffer, 0, values);
    return buffer;
  };
  const count = channels * height * width;
  const groupsX = Math.min(Math.ceil(count / 64),
    device.limits.maxComputeWorkgroupsPerDimension);
  const groupsY = Math.ceil(Math.ceil(count / 64) / groupsX);
  const low = upload(lowQuant, GPUBufferUsage.STORAGE);
  const high = upload(highQuant, GPUBufferUsage.STORAGE);
  const output = device.createBuffer({size: count * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});
  const params = upload(new Uint32Array([count, channels, height, width,
    lowQuant.length, groupsX, 0, 0]), GPUBufferUsage.UNIFORM);
  const bind = device.createBindGroup({layout: pipeline.getBindGroupLayout(0),
    entries: [inputBuffer, low, high, output, params]
      .map((buffer, binding) => ({binding, resource: {buffer}}))});
  const commands = device.createCommandEncoder();
  const pass = commands.beginComputePass();
  pass.setPipeline(pipeline); pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(groupsX, groupsY); pass.end();
  device.queue.submit([commands.finish()]);
  low.destroy(); high.destroy(); params.destroy();
  return output;
}
