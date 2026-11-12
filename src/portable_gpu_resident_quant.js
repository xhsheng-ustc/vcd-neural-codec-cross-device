const SOURCE = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<i32>;
@group(0) @binding(2) var<uniform> params: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * params.w * 64u;
  if (index >= params.x) { return; }
  let scale = bitcast<f32>(params.y);
  let rounded = round(input[index] * scale);
  output[index] = i32(clamp(rounded, -f32(params.z), f32(params.z)));
}`;
const PIPELINES = new WeakMap();

export function gpuResidentQuantize(device, inputBuffer, count, step,
                                    limit = 524287) {
  if (!(step > 0) || !Number.isInteger(Math.log2(step)))
    throw new Error('GPU activation step must be a power of two');
  let pipeline = PIPELINES.get(device);
  if (!pipeline) {
    const module = device.createShaderModule({code: SOURCE});
    pipeline = device.createComputePipeline({layout: 'auto',
      compute: {module, entryPoint: 'main'}});
    PIPELINES.set(device, pipeline);
  }
  const outputBuffer = device.createBuffer({size: count * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});
  const paramsBuffer = device.createBuffer({size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});
  const groupsX = Math.min(Math.ceil(count / 64),
    device.limits.maxComputeWorkgroupsPerDimension);
  const groupsY = Math.ceil(Math.ceil(count / 64) / groupsX);
  if (groupsY > device.limits.maxComputeWorkgroupsPerDimension)
    throw new Error('GPU quantization exceeds dispatch limit');
  const scaleBytes = new Uint32Array(new Float32Array([1 / step]).buffer)[0];
  device.queue.writeBuffer(paramsBuffer, 0,
    new Uint32Array([count, scaleBytes, limit, groupsX]));
  const group = device.createBindGroup({layout: pipeline.getBindGroupLayout(0),
    entries: [inputBuffer, outputBuffer, paramsBuffer].map((buffer, binding) =>
      ({binding, resource: {buffer}}))});
  const commands = device.createCommandEncoder();
  const pass = commands.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(groupsX, groupsY);
  pass.end();
  device.queue.submit([commands.finish()]);
  paramsBuffer.destroy();
  return outputBuffer;
}
