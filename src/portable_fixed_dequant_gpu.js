// Exact Q20 epilogue for signed 64-bit integer convolution sums.
// Per-channel multiplier is at most 65535; shifts are 15..35 in this asset.
export const FIXED_MATH_WGSL = `
fn abs64(words: vec2<u32>, negative: bool) -> vec2<u32> {
  if (!negative) { return words; }
  let low = (~words.x) + 1u;
  return vec2<u32>(low, (~words.y) + u32(low == 0u));
}

fn mul64x16(value: vec2<u32>, m: u32) -> vec2<u32> {
  let p0 = (value.x & 65535u) * m;
  let p1 = (value.x >> 16u) * m;
  let low = p0 + (p1 << 16u);
  let high = (p1 >> 16u) + u32(low < p0) + value.y * m;
  return vec2<u32>(low, high);
}

fn rounded_shift(value: vec2<u32>, shift: u32) -> u32 {
  if (shift < 32u) {
    let quotient = (value.x >> shift) | (value.y << (32u - shift));
    let remainder = value.x & ((1u << shift) - 1u);
    let half = 1u << (shift - 1u);
    return quotient + u32(remainder > half ||
      (remainder == half && (quotient & 1u) != 0u));
  }
  if (shift == 32u) {
    let quotient = value.y;
    return quotient + u32(value.x > 0x80000000u ||
      (value.x == 0x80000000u && (quotient & 1u) != 0u));
  }
  let upper_shift = shift - 32u;
  let quotient = value.y >> upper_shift;
  let remainder_high = value.y & ((1u << upper_shift) - 1u);
  let half_high = 1u << (upper_shift - 1u);
  return quotient + u32(remainder_high > half_high ||
    (remainder_high == half_high &&
      (value.x != 0u || (quotient & 1u) != 0u)));
}

fn fixed_value(words: vec2<u32>, m: u32, shift: u32, bias: i32) -> f32 {
  let negative = bitcast<i32>(words.y) < 0;
  let magnitude = abs64(words, negative);
  let product = mul64x16(magnitude, m);
  let rounded = rounded_shift(product, shift);
  let signed = select(i32(rounded), -i32(rounded), negative);
  return f32(signed + bias) * 0.00000095367431640625;
}`;

const SOURCE = `
@group(0) @binding(0) var<storage, read> sums: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> multiplier: array<u32>;
@group(0) @binding(2) var<storage, read> right_shift: array<u32>;
@group(0) @binding(3) var<storage, read> bias_q: array<i32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<uniform> params: vec4<u32>;

${FIXED_MATH_WGSL}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * params.z * 64u;
  if (index >= params.x) { return; }
  let channel = index / params.y;
  output[index] = fixed_value(sums[index], multiplier[channel],
    right_shift[channel], bias_q[channel]);
}`;

const PIPELINES = new WeakMap();

function upload(device, values, usage) {
  const buffer = device.createBuffer({size: values.byteLength,
    usage: usage | GPUBufferUsage.COPY_DST});
  device.queue.writeBuffer(buffer, 0, values);
  return buffer;
}

export async function fixedDequantGpuResident(device, sumBuffer, count,
                                               plane, fixed) {
  let pipeline = PIPELINES.get(device);
  if (!pipeline) {
    const module = device.createShaderModule({code: SOURCE});
    pipeline = device.createComputePipeline({layout: 'auto',
      compute: {module, entryPoint: 'main'}});
    PIPELINES.set(device, pipeline);
  }
  const multiplier = upload(device, fixed.multiplier, GPUBufferUsage.STORAGE);
  const rightShift = upload(device, fixed.rightShift, GPUBufferUsage.STORAGE);
  const biasQ = upload(device, fixed.biasQ, GPUBufferUsage.STORAGE);
  const groupsX = Math.min(Math.ceil(count / 64),
    device.limits.maxComputeWorkgroupsPerDimension);
  const groupsY = Math.ceil(Math.ceil(count / 64) / groupsX);
  if (groupsY > device.limits.maxComputeWorkgroupsPerDimension)
    throw new Error('Fixed dequantization exceeds WebGPU dispatch limit');
  const params = upload(device, new Uint32Array([count, plane, groupsX, 0]),
    GPUBufferUsage.UNIFORM);
  const output = device.createBuffer({size: count * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});
  const group = device.createBindGroup({layout: pipeline.getBindGroupLayout(0),
    entries: [sumBuffer, multiplier, rightShift, biasQ, output, params]
      .map((buffer, binding) => ({binding, resource: {buffer}}))});
  const commands = device.createCommandEncoder();
  const pass = commands.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(groupsX, groupsY);
  pass.end();
  device.queue.submit([commands.finish()]);
  for (const buffer of [multiplier, rightShift, biasQ, params])
    buffer.destroy();
  return output;
}

export async function readFixedGpuBuffer(device, buffer, count) {
  const read = device.createBuffer({size: count * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
  const commands = device.createCommandEncoder();
  commands.copyBufferToBuffer(buffer, 0, read, 0, count * 4);
  device.queue.submit([commands.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(new Float32Array(read.getMappedRange()));
  read.unmap();
  read.destroy();
  return result;
}
