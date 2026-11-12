// Browser reader for the desktop fixed-PTQ weight export. No float ONNX
// convolution weights are used by the exact kernels.

function typedValues(view, offset, count) {
  const values = new Int32Array(count);
  for (let i = 0; i < count; i++)
    values[i] = view.getInt32(offset + 4 * i, true);
  return values;
}

export async function loadPortableAssets(
  label, base = new URL('../models/portable', import.meta.url).href) {
  if (label !== 'i' && label !== 'p') throw new Error('Unknown codec model');
  const manifest = await (await fetch(`${base}/manifest.json`)).json();
  if (manifest.version !== 1 || manifest.settings?.weight_bits !== 16 ||
      manifest.settings?.activation_bits !== 16)
    throw new Error('Unexpected portable model format');
  const info = manifest[label];
  const response = await fetch(`${base}/${info.file}`);
  if (!response.ok) throw new Error(`Missing portable ${label} weights`);
  const bytes = await new Response(response.body.pipeThrough(
    new DecompressionStream('gzip'))).arrayBuffer();
  if (bytes.byteLength !== info.uncompressedBytes)
    throw new Error('Portable weight file size mismatch');
  const view = new DataView(bytes);
  const records = new Map(info.layers.map(layer => [layer.name, layer]));
  const decoded = new Map();
  return {
    manifest,
    names: [...records.keys()],
    layer(name) {
      if (decoded.has(name)) return decoded.get(name);
      const record = records.get(name);
      if (!record) throw new Error(`Missing portable layer: ${name}`);
      const result = {
        ...record,
        weight: typedValues(view, record.weight.offset, record.weight.count),
        fixed: {
          multiplier: typedValues(view, record.multiplier.offset,
            record.multiplier.count),
          rightShift: typedValues(view, record.rightShift.offset,
            record.rightShift.count),
          biasQ: typedValues(view, record.biasQ.offset,
            record.biasQ.count),
        },
      };
      decoded.set(name, result);
      return result;
    },
  };
}

export async function loadPortableSiluLut(
  base = new URL('../models/portable', import.meta.url).href) {
  const manifest = await (await fetch(`${base}/manifest.json`)).json();
  const info = manifest.silu;
  if (!info || info.stepBits !== manifest.settings.silu_step_bits ||
      info.count !== 16 * (1 << info.stepBits))
    throw new Error('Unexpected portable SiLU table');
  const response = await fetch(`${base}/${info.file}`);
  if (!response.ok) throw new Error('Missing portable SiLU table');
  const bytes = await new Response(response.body.pipeThrough(
    new DecompressionStream('gzip'))).arrayBuffer();
  if (bytes.byteLength !== info.count * 4)
    throw new Error('Portable SiLU table size mismatch');
  return new Float32Array(bytes);
}

export async function loadPortableIntraQTable(
  base = new URL('../models/portable', import.meta.url).href) {
  const manifest = await (await fetch(`${base}/manifest.json`)).json();
  const info = manifest.intraQ;
  if (!info || info.inputBits !== 14 || info.outputBits !== 20 ||
      info.count !== 16 * (1 << info.inputBits))
    throw new Error('Unexpected portable I-frame Q table');
  const response = await fetch(`${base}/${info.file}`);
  if (!response.ok) throw new Error('Missing portable I-frame Q table');
  const bytes = await new Response(response.body.pipeThrough(
    new DecompressionStream('gzip'))).arrayBuffer();
  if (bytes.byteLength !== info.count * 4)
    throw new Error('Portable I-frame Q table size mismatch');
  return new Float32Array(bytes);
}
