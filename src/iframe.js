import {RansDecoder, RansEncoder, RansTables} from './rans.js';
import {PortableModelKernels} from './portable_model_kernels.js';
import {add, asTensor, clamp, concatChannels, cropSpatial, flatNumbers,
        multiplySpatial, padToMultiple, roundEven, sliceChannels, tensor} from './tensors.js';

const I_MODULES = ['enc', 'hyper_enc', 'hyper_dec', 'y_prior_fusion',
                   'y_spatial_prior_reduction', 'y_spatial_prior_adaptor_1',
                   'y_spatial_prior_adaptor_2', 'y_spatial_prior_adaptor_3',
                   'y_spatial_prior', 'dec'];
const MICRO = [[0, 1, 2, 3], [3, 2, 1, 0],
               [2, 3, 0, 1], [1, 0, 3, 2]];
const F = Math.fround;

export function quantize(input) {
  const values = new Int8Array(input.data.length);
  const floats = new Float32Array(input.data.length);
  for (let i = 0; i < values.length; i++) {
    values[i] = clamp(roundEven(input.data[i]), -128, 127);
    floats[i] = values[i];
  }
  return {values, tensor: tensor(input.dims, floats)};
}

function separatePrior(params, qTable = null) {
  const [, channels, h, w] = params.dims;
  const hw = h * w;
  if (channels !== 514) throw new Error(`unexpected I prior channels: ${channels}`);
  const qEnc = new Float32Array(hw);
  const qDec = new Float32Array(hw);
  for (let i = 0; i < hw; i++) {
    if (qTable) {
      const offset = 8 * (1 << 14);
      qEnc[i] = qTable[clamp(roundEven(F(params.data[i] * (1 << 14))),
        -offset, offset - 1) + offset];
      qDec[i] = qTable[clamp(roundEven(F(params.data[hw + i] * (1 << 14))),
        -offset, offset - 1) + offset];
    } else {
      qEnc[i] = F(F(F(1 / F(1 + Math.exp(-params.data[i]))) * 1.5) + 0.5);
      qDec[i] = F(F(F(1 / F(1 + Math.exp(-params.data[hw + i]))) * 1.5) + 0.5);
    }
  }
  return {qEnc: tensor([1, 1, h, w], qEnc),
          qDec: tensor([1, 1, h, w], qDec),
          scales: sliceChannels(params, 2, 258),
          means: sliceChannels(params, 258, 514)};
}

function active(channel, row, column, quarter, step) {
  const micro = MICRO[step][Math.floor(channel / quarter)];
  return (row & 1) * 2 + (column & 1) === micro;
}

function processMasked(y, scales, means, step) {
  const [, channels, h, w] = y.dims;
  const hw = h * w;
  const quarter = channels / 4;
  const yhat = new Float32Array(channels * hw);
  const symbols = new Int8Array(quarter * hw);
  const scaleWrite = new Float32Array(quarter * hw);
  for (let c = 0; c < channels; c++) {
    const cBase = c * hw;
    const writeBase = (c % quarter) * hw;
    for (let row = 0; row < h; row++) {
      for (let col = 0; col < w; col++) {
        if (!active(c, row, col, quarter, step)) continue;
        const p = row * w + col;
        const mean = means.data[cBase + p];
        const symbol = clamp(roundEven(F(y.data[cBase + p] - mean)), -128, 127);
        symbols[writeBase + p] = symbol;
        scaleWrite[writeBase + p] = scales.data[cBase + p];
        yhat[cBase + p] = F(symbol + mean);
      }
    }
  }
  return {yhat: tensor(y.dims, yhat), symbols,
          scales: tensor([1, quarter, h, w], scaleWrite)};
}

function restoreMasked(symbols, means, step) {
  const [, channels, h, w] = means.dims;
  const hw = h * w;
  const quarter = channels / 4;
  const yhat = new Float32Array(channels * hw);
  for (let c = 0; c < channels; c++) {
    const cBase = c * hw;
    const readBase = (c % quarter) * hw;
    for (let row = 0; row < h; row++) {
      for (let col = 0; col < w; col++) {
        if (!active(c, row, col, quarter, step)) continue;
        const p = row * w + col;
        yhat[cBase + p] = F(symbols[readBase + p] + means.data[cBase + p]);
      }
    }
  }
  return tensor(means.dims, yhat);
}

export function scaleIndexes(scales, constants) {
  const result = new Uint8Array(scales.data.length);
  for (let i = 0; i < result.length; i++) {
    const scale = F(clamp(scales.data[i], constants.scale_min, constants.scale_max));
    result[i] = Math.trunc(F(F(F(Math.log(scale)) - F(constants.log_scale_min)) *
                             F(constants.log_step_recip)));
  }
  return result;
}

export function packSymbols(symbols, indexes) {
  const result = new Int16Array(symbols.length);
  for (let i = 0; i < result.length; i++) result[i] = (symbols[i] << 8) + indexes[i];
  return result;
}

export class IFrameCodec {
  constructor(ort, sessions, runtime, qp, portable = null) {
    this.ort = ort;
    this.sessions = sessions;
    this.portable = portable;
    this.runtime = runtime;
    this.qp = qp;
    this.qEnc = tensor([1, 368, 1, 1], flatNumbers(runtime.i.q_scale_enc[qp]));
    this.qDec = tensor([1, 368, 1, 1], flatNumbers(runtime.i.q_scale_dec[qp]));
    this.groups = [runtime.i.gaussian, runtime.i.z].map(group =>
      new RansTables(group.cdfs, group.lengths, group.offsets));
  }

  static async load(ort, modelBase, qp = 32, backend = 'wasm', log = () => {},
                    role = 'all', portableBackend = null) {
    const format = await (await fetch(`${modelBase}/asset_format.json`)).json();
    const get = async filename => {
      const response = await fetch(`${modelBase}/${filename}${format.gzip ? '.gz' : ''}`);
      if (!response.ok) throw new Error(`missing ${filename}`);
      return format.gzip
        ? await new Response(response.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer()
        : await response.arrayBuffer();
    };
    const runtime = JSON.parse(new TextDecoder().decode(await get('runtime.json')));
    const chunks = await (await fetch(`${modelBase}/model_chunks.json`)).json();
    const sessions = {};
    let names = role === 'decoder'
      ? I_MODULES.filter(name => name !== 'enc' && name !== 'hyper_enc') : I_MODULES;
    if (portableBackend) names = names.filter(name =>
      name === 'enc' || name === 'hyper_enc');
    const modelBytes = await Promise.all(names.map(async name => {
      const files = chunks[`i_${name}`];
      let bytes;
      if (files) {
        const parts = await Promise.all(files.map(async file => {
          return new Uint8Array(await get(file));
        }));
        bytes = new Uint8Array(parts.reduce((sum, part) => sum + part.length, 0));
        let offset = 0;
        for (const part of parts) { bytes.set(part, offset); offset += part.length; }
      } else {
        bytes = await get(`i_${name}.onnx`);
      }
      return bytes;
    }));
    for (let index = 0; index < names.length; index++) {
      const name = names[index];
      sessions[name] = await ort.InferenceSession.create(modelBytes[index],
        {executionProviders: [backend]});
      modelBytes[index] = null;
      log(`loaded i_${name}`);
    }
    const portable = portableBackend
      ? await PortableModelKernels.create('i', portableBackend) : null;
    return new IFrameCodec(ort, sessions, runtime, qp, portable);
  }

  async run(name, ...inputs) {
    const trace = async output => {
      if (this.trace) {
        const digest = async data => Array.from(new Uint8Array(await crypto.subtle.digest(
          'SHA-256', data.buffer.slice(data.byteOffset,
            data.byteOffset + data.byteLength)))).map(v =>
          v.toString(16).padStart(2, '0')).join('');
        this.trace.push({name,
          inputs: await Promise.all(inputs.map(x => digest(x.data))),
          output: await digest(output.data)});
      }
      return output;
    };
    if (this.portable && name !== 'enc' && name !== 'hyper_enc') {
      const x = inputs[0];
      const [, , h, w] = x.dims;
      let values, channels, outH = h, outW = w;
      if (name === 'hyper_dec') {
        values = await this.portable.hyperDecoder(x.data, h, w);
        channels = 256; outH *= 4; outW *= 4;
      } else if (name === 'y_prior_fusion') {
        values = await this.portable.intraPriorFusion(x.data, h, w);
        channels = 514;
      } else if (name === 'y_spatial_prior_reduction') {
        values = await this.portable.intraPriorReduction(x.data, h, w);
        channels = 256;
      } else if (name.startsWith('y_spatial_prior_adaptor_')) {
        values = await this.portable.intraSpatialAdaptor(Number(name.at(-1)),
          x.data, h, w);
        channels = 512;
      } else if (name === 'y_spatial_prior') {
        values = await this.portable.intraSpatialPrior(x.data, h, w);
        channels = 512;
      } else if (name === 'dec') {
        values = await this.portable.intraDecoder(x.data, h, w,
          inputs[1].data);
        channels = 3; outH *= 16; outW *= 16;
      } else throw new Error(`Unsupported portable I module: ${name}`);
      return trace(tensor([1, channels, outH, outW], values));
    }
    const feeds = {};
    inputs.forEach((input, index) => {
      feeds[`input_${index}`] = new this.ort.Tensor('float32', input.data, input.dims);
    });
    const results = await this.sessions[name].run(feeds);
    return trace(asTensor(results.output_0));
  }

  async prior(z, yH, yW) {
    let params = await this.run('hyper_dec', z);
    params = await this.run('y_prior_fusion', params);
    params = cropSpatial(params, yH, yW);
    const separated = separatePrior(params, this.portable?.qTable);
    const reduced = await this.run('y_spatial_prior_reduction', params);
    return {...separated, reduced};
  }

  async nextPrior(yhatSoFar, reduced, step) {
    const combined = concatChannels(yhatSoFar, reduced);
    const adapted = await this.run(`y_spatial_prior_adaptor_${step}`, combined);
    const prior = await this.run('y_spatial_prior', adapted);
    const half = prior.dims[1] / 2;
    return {scales: sliceChannels(prior, 0, half),
            means: sliceChannels(prior, half, prior.dims[1])};
  }

  async encode(frame) {
    const y = await this.run('enc', frame, this.qEnc);
    const z = await this.run('hyper_enc', padToMultiple(y, 4));
    const quantizedZ = quantize(z);
    const prior = await this.prior(quantizedZ.tensor, y.dims[2], y.dims[3]);
    const yScaled = multiplySpatial(y, prior.qEnc);
    const entropy = new RansEncoder(this.groups);
    const [, zChannels, zH, zW] = z.dims;
    entropy.encodeZ(quantizedZ.values, 1, this.qp * zChannels, zH * zW);
    let {scales, means} = prior;
    let yhatSoFar = null;
    for (let step = 0; step < 4; step++) {
      const part = processMasked(yScaled, scales, means, step);
      const indexes = scaleIndexes(part.scales, this.runtime.gaussian_index);
      entropy.encodeY(packSymbols(part.symbols, indexes), 0);
      yhatSoFar = yhatSoFar ? add(yhatSoFar, part.yhat) : part.yhat;
      if (step < 3) ({scales, means} = await this.nextPrior(yhatSoFar, prior.reduced, step + 1));
    }
    const reconstructed = await this.run('dec', multiplySpatial(yhatSoFar, prior.qDec), this.qDec);
    for (let i = 0; i < reconstructed.data.length; i++) {
      reconstructed.data[i] = clamp(reconstructed.data[i], 0, 1);
    }
    return {stream: entropy.flush(), reconstructed};
  }

  async decode(stream, height, width) {
    const decoder = new RansDecoder(this.groups);
    decoder.setStream(stream);
    const zH = Math.ceil(height / 64);
    const zW = Math.ceil(width / 64);
    const zChannels = 128;
    const zBytes = decoder.decodeZ(zChannels * zH * zW, 1,
                                    this.qp * zChannels, zH * zW);
    const z = tensor([1, zChannels, zH, zW], Float32Array.from(zBytes));
    const yH = Math.ceil(height / 16);
    const yW = Math.ceil(width / 16);
    const prior = await this.prior(z, yH, yW);
    let {scales, means} = prior;
    let yhatSoFar = null;
    for (let step = 0; step < 4; step++) {
      const [, channels, h, w] = scales.dims;
      const quarter = channels / 4;
      const scaleWrite = new Float32Array(quarter * h * w);
      const hw = h * w;
      for (let c = 0; c < channels; c++) {
        for (let row = 0; row < h; row++) {
          for (let col = 0; col < w; col++) {
            if (!active(c, row, col, quarter, step)) continue;
            const p = row * w + col;
            scaleWrite[(c % quarter) * hw + p] = scales.data[c * hw + p];
          }
        }
      }
      const indexes = scaleIndexes(tensor([1, quarter, h, w], scaleWrite),
                                   this.runtime.gaussian_index);
      const symbols = decoder.decodeY(indexes, 0);
      const part = restoreMasked(symbols, means, step);
      yhatSoFar = yhatSoFar ? add(yhatSoFar, part) : part;
      if (step < 3) ({scales, means} = await this.nextPrior(yhatSoFar, prior.reduced, step + 1));
    }
    const reconstructed = await this.run('dec', multiplySpatial(yhatSoFar, prior.qDec), this.qDec);
    for (let i = 0; i < reconstructed.data.length; i++) {
      reconstructed.data[i] = clamp(reconstructed.data[i], 0, 1);
    }
    return {reconstructed};
  }
}
