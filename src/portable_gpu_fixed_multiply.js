const SOURCE = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> quant: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

fn round_even(value: f32) -> i32 {
  let low = floor(value);
  let fraction = value - low;
  let base = i32(low);
  return base + i32(fraction > 0.5 ||
    (fraction == 0.5 && (base & 1) != 0));
}

fn product64(a: i32, b: i32) -> vec2<u32> {
  let aa = u32(abs(a));
  let bb = u32(abs(b));
  let al = aa & 65535u;
  let ah = aa >> 16u;
  let bl = bb & 65535u;
  let bh = bb >> 16u;
  let p0 = al * bl;
  let cross = al * bh + ah * bl;
  let low = p0 + (cross << 16u);
  let high = (cross >> 16u) + ah * bh + u32(low < p0);
  return vec2<u32>(low, high);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x + id.y * params.w * 64u;
  if (i >= params.x) { return; }
  let qi = select(select(i / params.y, i, params.z == params.x),
    0u, params.z == 1u);
  let a = round_even(input[i] * 1048576.0);
  let b = round_even(quant[qi] * 1048576.0);
  let product = product64(a, b);
  let quotient = (product.x >> 20u) | (product.y << 12u);
  let remainder = product.x & 1048575u;
  let increase = remainder > 524288u ||
    (remainder == 524288u && (quotient & 1u) != 0u);
  let magnitude = i32(quotient + u32(increase));
  let signed = select(magnitude, -magnitude, (a < 0) != (b < 0));
  output[i] = f32(signed) * 0.00000095367431640625;
}`;
const PIPELINES = new WeakMap();

export function gpuFixedMultiplyResident(device, input, count,
                                          quantValues, channels) {
  let pipeline = PIPELINES.get(device);
  if (!pipeline) {
    pipeline = device.createComputePipeline({layout: 'auto', compute: {
      module: device.createShaderModule({code: SOURCE}), entryPoint: 'main'}});
    PIPELINES.set(device, pipeline);
  }
  const q = device.createBuffer({size: quantValues.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});
  device.queue.writeBuffer(q, 0, quantValues);
  const groupsX = Math.min(Math.ceil(count / 64),
    device.limits.maxComputeWorkgroupsPerDimension);
  const groupsY = Math.ceil(Math.ceil(count / 64) / groupsX);
  if (groupsY > device.limits.maxComputeWorkgroupsPerDimension)
    throw new Error('Fixed multiply exceeds WebGPU dispatch limit');
  const params = device.createBuffer({size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});
  device.queue.writeBuffer(params, 0, new Uint32Array([
    count, count / channels, quantValues.length, groupsX]));
  const output = device.createBuffer({size: count * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});
  const bind = device.createBindGroup({layout: pipeline.getBindGroupLayout(0),
    entries: [input, q, output, params].map((buffer, binding) =>
      ({binding, resource: {buffer}}))});
  const commands = device.createCommandEncoder();
  const pass = commands.beginComputePass();
  pass.setPipeline(pipeline); pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(groupsX, groupsY);
  pass.end();
  device.queue.submit([commands.finish()]);
  q.destroy(); params.destroy();
  return output;
}
