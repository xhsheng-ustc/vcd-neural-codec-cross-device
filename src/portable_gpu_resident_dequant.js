import {DEQUANT_SOURCE} from './portable_gpu_dequant_probe.js';

const PIPELINES = new WeakMap();
const PATCH_SOURCE = `
@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<storage, read> patches: array<vec2<u32>>;
@group(0) @binding(2) var<uniform> count: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= count.x) { return; }
  let item = patches[id.x];
  output[item.x] = bitcast<f32>(item.y);
}`;

function pipelinePair(device) {
  let pair = PIPELINES.get(device);
  if (pair) return pair;
  pair = [DEQUANT_SOURCE, PATCH_SOURCE].map(source => {
    const module = device.createShaderModule({code: source});
    return device.createComputePipeline({layout: 'auto',
      compute: {module, entryPoint: 'main'}});
  });
  PIPELINES.set(device, pair);
  return pair;
}

function upload(device, data, usage) {
  const buffer = device.createBuffer({size: data.byteLength,
    usage: usage | GPUBufferUsage.COPY_DST});
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

// Keeps all ordinary outputs on the GPU. Only the rare rounding-risk sums are
// transferred to CPU for the reference FP64 operation, then patched on GPU.
// The risk detector remains experimental; callers must validate exact parity.
export async function gpuResidentDequantize(device, sumBuffer, count, plane,
                                             weightSteps, inputStep, bias) {
  const [dequantPipeline, patchPipeline] = pipelinePair(device);
  const capacity = Math.max(4, Math.ceil(count * 0.02));
  const splitSteps = new Float32Array(weightSteps.length * 2);
  for (let i = 0; i < weightSteps.length; i++) {
    const full = weightSteps[i] * inputStep;
    splitSteps[2 * i] = Math.fround(full);
    splitSteps[2 * i + 1] = Math.fround(full - splitSteps[2 * i]);
  }
  const stepsBuffer = upload(device, splitSteps, GPUBufferUsage.STORAGE);
  const biasBuffer = upload(device, new Float32Array(bias), GPUBufferUsage.STORAGE);
  const outputBuffer = device.createBuffer({size: count * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});
  const paramsBuffer = upload(device,
    new Uint32Array([count, plane, capacity, 0]), GPUBufferUsage.UNIFORM);
  const debugBuffer = device.createBuffer({size: 64, usage: GPUBufferUsage.STORAGE});
  const recordBuffer = device.createBuffer({size: capacity * 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});
  const counterBuffer = device.createBuffer({size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});
  const recordRead = device.createBuffer({size: capacity * 16,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
  const counterRead = device.createBuffer({size: 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
  const inputs = [sumBuffer, stepsBuffer, biasBuffer, outputBuffer,
    paramsBuffer, debugBuffer, recordBuffer, counterBuffer];
  const group = device.createBindGroup({layout: dequantPipeline.getBindGroupLayout(0),
    entries: inputs.map((buffer, binding) => ({binding, resource: {buffer}}))});
  device.pushErrorScope('validation');
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(dequantPipeline);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(Math.ceil(count / 64));
  pass.end();
  encoder.copyBufferToBuffer(recordBuffer, 0, recordRead, 0, capacity * 16);
  encoder.copyBufferToBuffer(counterBuffer, 0, counterRead, 0, 4);
  device.queue.submit([encoder.finish()]);
  const error = await device.popErrorScope();
  if (error) throw new Error(`GPU dequantization rejected: ${error.message}`);
  await Promise.all([recordRead.mapAsync(GPUMapMode.READ),
    counterRead.mapAsync(GPUMapMode.READ)]);
  const uncertainCount = new Uint32Array(counterRead.getMappedRange())[0];
  if (uncertainCount > capacity)
    throw new Error(`GPU dequantization correction capacity exceeded: ${uncertainCount}/${capacity}`);
  const records = new Uint32Array(recordRead.getMappedRange());
  let patchBuffer = null, patchParams = null;
  if (uncertainCount) {
    const patches = new Uint32Array(uncertainCount * 2);
    const bits = new Uint32Array(new Float32Array(1).buffer);
    const value = new Float32Array(bits.buffer);
    for (let i = 0; i < uncertainCount; i++) {
      const index = records[4 * i];
      const lo = records[4 * i + 1];
      const hi = records[4 * i + 2];
      const sum = Number(BigInt.asIntN(64,
        (BigInt(hi) << 32n) | BigInt(lo)));
      if (!Number.isSafeInteger(sum))
        throw new Error('GPU sum outside exact JavaScript integer range');
      const channel = Math.floor(index / plane);
      value[0] = Math.fround((sum * inputStep) * weightSteps[channel] + bias[channel]);
      patches[2 * i] = index;
      patches[2 * i + 1] = bits[0];
    }
    patchBuffer = upload(device, patches, GPUBufferUsage.STORAGE);
    patchParams = upload(device,
      new Uint32Array([uncertainCount, 0, 0, 0]), GPUBufferUsage.UNIFORM);
    const patchGroup = device.createBindGroup({layout: patchPipeline.getBindGroupLayout(0),
      entries: [outputBuffer, patchBuffer, patchParams].map((buffer, binding) =>
        ({binding, resource: {buffer}}))});
    device.pushErrorScope('validation');
    const commands = device.createCommandEncoder();
    const patchPass = commands.beginComputePass();
    patchPass.setPipeline(patchPipeline);
    patchPass.setBindGroup(0, patchGroup);
    patchPass.dispatchWorkgroups(Math.ceil(uncertainCount / 64));
    patchPass.end();
    device.queue.submit([commands.finish()]);
    const patchError = await device.popErrorScope();
    if (patchError) throw new Error(`GPU correction rejected: ${patchError.message}`);
  }
  recordRead.unmap();
  counterRead.unmap();
  for (const buffer of [stepsBuffer, biasBuffer, paramsBuffer, debugBuffer,
    recordBuffer, counterBuffer, recordRead, counterRead,
    patchBuffer, patchParams]) buffer?.destroy();
  return {buffer: outputBuffer, uncertainCount};
}
