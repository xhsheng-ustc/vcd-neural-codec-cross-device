// Experimental GPU dequantization. This module is intentionally isolated from
// the codec until it matches the FP64 reference on real model activations.
export const DEQUANT_SOURCE = `
@group(0) @binding(0) var<storage, read> sums: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> steps: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> biases: array<f32>;
@group(0) @binding(3) var<storage, read_write> result: array<f32>;
@group(0) @binding(4) var<uniform> params: vec4<u32>;
@group(0) @binding(5) var<storage, read_write> debug: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> records: array<vec4<u32>>;
@group(0) @binding(7) var<storage, read_write> counter: atomic<u32>;

fn two_sum(a: f32, b: f32) -> vec2<f32> {
  let s = a + b;
  let bb = s - a;
  return vec2<f32>(s, (a - (s - bb)) + (b - bb));
}

fn two_prod(a: f32, b: f32) -> vec2<f32> {
  let p = a * b;
  let ca = 4097.0 * a;
  let ah = ca - (ca - a);
  let al = a - ah;
  let cb = 4097.0 * b;
  let bh = cb - (cb - b);
  let bl = b - bh;
  let e = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
  return vec2<f32>(p, e);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= params.x) { return; }
  let words = sums[i];
  let high = f32(bitcast<i32>(words.y)) * 4294967296.0;
  let middle = f32(words.x >> 16u) * 65536.0;
  let low = f32(words.x & 65535u);
  let partial = two_sum(high, middle);
  let total = two_sum(partial.x, low);
  let sum_lo = (partial.y + total.y);
  let st = steps[i / params.y];
  let pair = two_prod(total.x, st.x);
  let product = pair.x;
  let error = pair.y +
    total.x * st.y + sum_lo * st.x + sum_lo * st.y;
  let corrected = two_sum(product, error);
  let added = two_sum(corrected.x, biases[i / params.y]);
  let residue = added.y + corrected.y;
  result[i] = added.x + residue;
  let ulp = abs(bitcast<f32>(bitcast<u32>(added.x) + 1u) - added.x);
  if (abs(result[i]) < abs(product) * 0.0001 ||
      abs(abs(residue) - 0.5 * ulp) < 0.001 * ulp) {
    let slot = atomicAdd(&counter, 1u);
    if (slot < params.z) {
      records[slot] = vec4<u32>(i, words.x, words.y, 0u);
    }
  }
  if (i < 4u) { debug[i] = vec4<f32>(total.x, sum_lo, product, error); }
}`;

function upload(device, data, usage) {
  const buffer = device.createBuffer({size: data.byteLength,
    usage: usage | GPUBufferUsage.COPY_DST});
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

export async function probeGpuDequant(device, sums, step, inputStep,
                                      bias, plane) {
  const count = sums.length;
  const sumWords = new Uint32Array(count * 2);
  for (let i = 0; i < count; i++) {
    const whole = BigInt(sums[i]);
    sumWords[2 * i] = Number(whole & 0xffffffffn);
    sumWords[2 * i + 1] = Number(BigInt.asUintN(32, whole >> 32n));
  }
  const splitSteps = new Float32Array(step.length * 2);
  for (let i = 0; i < step.length; i++) {
    const full = step[i] * inputStep;
    splitSteps[2 * i] = Math.fround(full);
    splitSteps[2 * i + 1] = Math.fround(full - splitSteps[2 * i]);
  }
  const bias32 = new Float32Array(bias);
  const shader = device.createShaderModule({code: DEQUANT_SOURCE});
  const pipeline = device.createComputePipeline({layout: 'auto',
    compute: {module: shader, entryPoint: 'main'}});
  const capacity = Math.max(4, Math.ceil(count * 0.02));
  const buffers = [
    upload(device, sumWords, GPUBufferUsage.STORAGE),
    upload(device, splitSteps, GPUBufferUsage.STORAGE),
    upload(device, bias32, GPUBufferUsage.STORAGE),
    device.createBuffer({size: count * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC}),
    upload(device, new Uint32Array([count, plane, capacity, 0]), GPUBufferUsage.UNIFORM),
    device.createBuffer({size: count * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ}),
    device.createBuffer({size: 64,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC}),
    device.createBuffer({size: 64,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ}),
    device.createBuffer({size: capacity * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC}),
    device.createBuffer({size: capacity * 16,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ}),
    device.createBuffer({size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC}),
    device.createBuffer({size: 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ}),
  ];
  const group = device.createBindGroup({layout: pipeline.getBindGroupLayout(0),
    entries: buffers.slice(0, 5).concat(buffers[6], buffers[8], buffers[10]).map((buffer, binding) =>
      ({binding, resource: {buffer}}))});
  device.pushErrorScope('validation');
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(Math.ceil(count / 64));
  pass.end();
  encoder.copyBufferToBuffer(buffers[3], 0, buffers[5], 0, count * 4);
  encoder.copyBufferToBuffer(buffers[6], 0, buffers[7], 0, 64);
  encoder.copyBufferToBuffer(buffers[8], 0, buffers[9], 0, capacity * 16);
  encoder.copyBufferToBuffer(buffers[10], 0, buffers[11], 0, 4);
  device.queue.submit([encoder.finish()]);
  const error = await device.popErrorScope();
  if (error) throw new Error(error.message);
  await Promise.all([buffers[5].mapAsync(GPUMapMode.READ),
    buffers[7].mapAsync(GPUMapMode.READ),
    buffers[9].mapAsync(GPUMapMode.READ),
    buffers[11].mapAsync(GPUMapMode.READ)]);
  const output = new Float32Array(new Float32Array(buffers[5].getMappedRange()));
  const debugValues = new Float32Array(new Float32Array(buffers[7].getMappedRange()));
  output.debugValues = Array.from(debugValues);
  const countFlagged = new Uint32Array(buffers[11].getMappedRange())[0];
  if (countFlagged > capacity) throw new Error('GPU correction capacity exceeded');
  const flagged = new Uint32Array(count);
  const correctionRecords = new Uint32Array(buffers[9].getMappedRange());
  for (let i = 0; i < countFlagged; i++) flagged[correctionRecords[4 * i]] = 1;
  output.uncertain = flagged;
  buffers[5].unmap();
  buffers[7].unmap();
  buffers[9].unmap();
  buffers[11].unmap();
  for (const buffer of buffers) buffer.destroy();
  return output;
}
