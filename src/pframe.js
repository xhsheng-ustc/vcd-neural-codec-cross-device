import {RansDecoder, RansEncoder, RansTables} from './rans.js';

let stateGrid = 0;
let featureGrid = 0;
export function setStateGrid(grid = 0) {
  if (grid && (!Number.isInteger(grid) || grid < 1))
    throw new RangeError('Invalid state grid');
  stateGrid = grid;
}

export function setFeatureGrid(grid = 0) {
  if (grid && (!Number.isInteger(grid) || grid < 1))
    throw new RangeError('Invalid feature grid');
  featureGrid = grid;
}

function snapState(tensor) {
  if (!stateGrid) return tensor;
  const values = tensor.data;
  for (let i = 0; i < values.length; i++)
    values[i] = Math.fround(Math.round(values[i] * stateGrid) / stateGrid);
  return tensor;
}


function snapFeature(tensor) {
  if (!featureGrid) return snapState(tensor);
  const values = tensor.data;
  for (let i = 0; i < values.length; i++)
    values[i] = Math.fround(Math.round(values[i] * featureGrid) / featureGrid);
  return tensor;
}
import {packSymbols, quantize, scaleIndexes} from './iframe.js';
import {add, asTensor, clamp, concatChannels, cropSpatial, flatNumbers,
        padToMultiple, pixelUnshuffle, roundEven, sliceChannels, tensor} from './tensors.js';

const P_MODULES = ['feature_adaptor_i', 'feature_adaptor_p', 'feature_extractor',
                   'encoder', 'hyper_encoder', 'hyper_decoder',
                   'temporal_prior_encoder', 'y_prior_fusion', 'y_spatial_prior',
                   'decoder', 'recon_generation_net_lpips'];
const QUANT_NAMES = ['q_encoder_perception', 'q_encoder_high_perception',
                     'q_decoder_perception', 'q_decoder_high_perception',
                     'q_feature_perception', 'q_feature_p1_perception',
                     'q_feature_p2_perception', 'q_feature_p3_perception',
                     'q_recon_perception'];
const F = Math.fround;

function active(channel, row, column, half, step) {
  const parity = (row + column) & 1;
  return parity === ((Math.floor(channel / half) + step) & 1);
}

function separatePrior(params) {
  if (params.dims[1] !== 240) throw new Error('unexpected P prior channels');
  const qDec = sliceChannels(params, 0, 80);
  for (let i = 0; i < qDec.data.length; i++) qDec.data[i] = Math.max(qDec.data[i], 0.5);
  return {qDec, scales: sliceChannels(params, 80, 160),
          means: sliceChannels(params, 160, 240)};
}

function divideByQuant(input, quant) {
  const output = new Float32Array(input.data.length);
  for (let i = 0; i < output.length; i++) output[i] = F(input.data[i] * F(1 / quant.data[i]));
  return tensor(input.dims, output);
}

function multiplyByQuant(input, quant) {
  const output = new Float32Array(input.data.length);
  for (let i = 0; i < output.length; i++) output[i] = input.data[i] * quant.data[i];
  return tensor(input.dims, output);
}

function maskedEncode(y, scales, means, step) {
  const [, channels, h, w] = y.dims;
  const hw = h * w;
  const half = channels / 2;
  const yhat = new Float32Array(channels * hw);
  const symbols = new Int8Array(half * hw);
  const scaleWrite = new Float32Array(half * hw);
  for (let c = 0; c < channels; c++) {
    const cBase = c * hw;
    const writeBase = (c % half) * hw;
    for (let row = 0; row < h; row++) {
      for (let col = 0; col < w; col++) {
        if (!active(c, row, col, half, step)) continue;
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
          scales: tensor([1, half, h, w], scaleWrite)};
}

function maskedScales(scales, step) {
  const [, channels, h, w] = scales.dims;
  const hw = h * w;
  const half = channels / 2;
  const output = new Float32Array(half * hw);
  for (let c = 0; c < channels; c++) {
    for (let row = 0; row < h; row++) {
      for (let col = 0; col < w; col++) {
        if (!active(c, row, col, half, step)) continue;
        const p = row * w + col;
        output[(c % half) * hw + p] = scales.data[c * hw + p];
      }
    }
  }
  return tensor([1, half, h, w], output);
}

function maskedRestore(symbols, means, step) {
  const [, channels, h, w] = means.dims;
  const hw = h * w;
  const half = channels / 2;
  const output = new Float32Array(channels * hw);
  for (let c = 0; c < channels; c++) {
    for (let row = 0; row < h; row++) {
      for (let col = 0; col < w; col++) {
        if (!active(c, row, col, half, step)) continue;
        const p = row * w + col;
        output[c * hw + p] = F(symbols[(c % half) * hw + p] + means.data[c * hw + p]);
      }
    }
  }
  return tensor(means.dims, output);
}

export class PFrameCodec {
  constructor(ort, sessions, runtime, qp, portable = null) {
    this.ort = ort;
    this.sessions = sessions;
    this.portable = portable;
    this.runtime = runtime;
    this.qp = qp;
    this.quant = {};
    for (const name of QUANT_NAMES) {
      this.quant[name] = tensor([1, 192, 1, 1],
                                flatNumbers(runtime.p.quantizers[name][qp]));
    }
    this.groups = [runtime.p.gaussian, runtime.p.z].map(group =>
      new RansTables(group.cdfs, group.lengths, group.offsets));
    this.refFrame = null;
    this.refFeature = null;
  }

  setQp(qp) {
    if (this.qp === qp) return;
    this.qp = qp;
    for (const name of QUANT_NAMES) {
      this.quant[name] = tensor([1, 192, 1, 1],
                                flatNumbers(this.runtime.p.quantizers[name][qp]));
    }
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
    const sessions = {};
    let names = role === 'decoder'
      ? P_MODULES.filter(name => name !== 'encoder' && name !== 'hyper_encoder') : P_MODULES;
    if (portableBackend) names = names.filter(name =>
      name === 'encoder' || name === 'hyper_encoder');
    const modelBytes = await Promise.all(names.map(name => get(`p_${name}.onnx`)));
    for (let index = 0; index < names.length; index++) {
      const name = names[index];
      sessions[name] = await ort.InferenceSession.create(modelBytes[index],
        {executionProviders: [backend]});
      modelBytes[index] = null;
      log(`loaded p_${name}`);
    }
    const portable = portableBackend
      ? await (await import('./portable_model_kernels.js')).PortableModelKernels
        .create('p', portableBackend) : null;
    return new PFrameCodec(ort, sessions, runtime, qp, portable);
  }

  setReference(frame) { this.refFrame = snapState(frame); this.refFeature = null; }

  async run(name, ...inputs) {
    const started = this.portable?.profile ? performance.now() : 0;
    const record = value => {
      if (this.portable?.profile)
        this.portable.profile.push({name: `module:${name}`, elements: 0,
          quantMs: 0, computeMs: performance.now() - started, dequantMs: 0});
      return value;
    };
    if (this.portable && name !== 'encoder' && name !== 'hyper_encoder') {
      const x = inputs[0];
      const [, , h, w] = x.dims;
      let values, channels, outH = h, outW = w;
      if (name === 'feature_adaptor_i') {
        values = await this.portable.featureAdaptorI(x.data, h, w);
        channels = 192;
      } else if (name === 'feature_adaptor_p') {
        values = await this.portable.featureAdaptorP(x.data, h, w);
        channels = 192;
      } else if (name === 'feature_extractor') {
        const result = await this.portable.featureExtractor(x.data, h, w,
          ...inputs.slice(1).map(item => item.data));
        return record([tensor([1, 192, h, w], result.context),
          tensor([1, 192, h, w], result.temporal)]);
      } else if (name === 'hyper_decoder') {
        values = await this.portable.hyperDecoder(x.data, h, w);
        channels = 80; outH *= 4; outW *= 4;
      } else if (name === 'temporal_prior_encoder') {
        values = await this.portable.temporalPrior(x.data, h, w);
        channels = 160; outH /= 2; outW /= 2;
      } else if (name === 'y_prior_fusion') {
        values = await this.portable.priorFusion(x.data, h, w);
        channels = 240;
      } else if (name === 'y_spatial_prior') {
        values = await this.portable.spatialPrior(x.data, h, w);
        channels = 160;
      } else if (name === 'decoder') {
        values = await this.portable.decoder(x.data, inputs[1].data,
          h, w, inputs[2].data, inputs[3].data);
        channels = 192; outH *= 2; outW *= 2;
      } else if (name === 'recon_generation_net_lpips') {
        values = await this.portable.reconstruction(x.data, h, w,
          inputs[1].data);
        channels = 3; outH *= 16; outW *= 16;
      } else throw new Error(`Unsupported portable P module: ${name}`);
      return record([tensor([1, channels, outH, outW], values)]);
    }
    const feeds = {};
    inputs.forEach((input, index) => {
      feeds[`input_${index}`] = new this.ort.Tensor('float32', input.data, input.dims);
    });
    const results = await this.sessions[name].run(feeds);
    return record(Object.keys(results).sort().map(key => asTensor(results[key])));
  }

  async one(name, ...inputs) { return (await this.run(name, ...inputs))[0]; }

  q(name) { return this.quant[`q_${name}_perception`]; }

  async featureContext() {
    if (!this.refFrame && !this.refFeature) throw new Error('P frame needs an I reference');
    const feature = this.refFeature
      ? await this.one('feature_adaptor_p', this.refFeature)
      : await this.one('feature_adaptor_i', pixelUnshuffle(this.refFrame, 16));
    return this.run('feature_extractor', feature, this.q('feature'),
                    this.q('feature_p1'), this.q('feature_p2'), this.q('feature_p3'));
  }

  async prior(z, ctxT, yH, yW) {
    const hierarchical = cropSpatial(await this.one('hyper_decoder', z), yH, yW);
    const temporal = await this.one('temporal_prior_encoder', ctxT);
    const params = await this.one('y_prior_fusion', concatChannels(hierarchical, temporal));
    return {params, ...separatePrior(params)};
  }

  async nextPrior(yhat0, params) {
    const result = await this.one('y_spatial_prior', concatChannels(yhat0, params));
    return {scales: sliceChannels(result, 0, 80), means: sliceChannels(result, 80, 160)};
  }

  async reconstruct(yhat, ctx) {
    const feature = await this.one('decoder', yhat, ctx, this.q('decoder'),
                                   this.q('decoder_high'));
    const reconstructed = await this.one('recon_generation_net_lpips',
                                         feature, this.q('recon'));
    this.rawFeature = featureGrid ? Float32Array.from(feature.data) : null;
    this.refFeature = snapFeature(feature);
    this.refFrame = snapState(reconstructed);
    return this.refFrame;
  }

  async encode(frame) {
    const [ctx, ctxT] = await this.featureContext();
    const y = await this.one('encoder', frame, ctx, this.q('encoder'),
                             this.q('encoder_high'));
    const z = await this.one('hyper_encoder', padToMultiple(y, 4));
    const quantizedZ = quantize(z);
    const prior = await this.prior(quantizedZ.tensor, ctxT, y.dims[2], y.dims[3]);
    const yScaled = divideByQuant(y, prior.qDec);
    const entropy = new RansEncoder(this.groups);
    const [, zChannels, zH, zW] = z.dims;
    entropy.encodeZ(quantizedZ.values, 1, this.qp * zChannels, zH * zW);
    const part0 = maskedEncode(yScaled, prior.scales, prior.means, 0);
    entropy.encodeY(packSymbols(part0.symbols,
      scaleIndexes(part0.scales, this.runtime.gaussian_index)), 0);
    const next = await this.nextPrior(part0.yhat, prior.params);
    const part1 = maskedEncode(yScaled, next.scales, next.means, 1);
    entropy.encodeY(packSymbols(part1.symbols,
      scaleIndexes(part1.scales, this.runtime.gaussian_index)), 0);
    const yhat = multiplyByQuant(add(part0.yhat, part1.yhat), prior.qDec);
    const reconstructed = await this.reconstruct(yhat, ctx);
    return {stream: entropy.flush(), reconstructed};
  }

  async decode(stream, height, width) {
    const decoder = new RansDecoder(this.groups);
    decoder.setStream(stream);
    const zH = Math.ceil(height / 128);
    const zW = Math.ceil(width / 128);
    const zChannels = 96;
    const zBytes = decoder.decodeZ(zChannels * zH * zW, 1,
                                    this.qp * zChannels, zH * zW);
    const z = tensor([1, zChannels, zH, zW], Float32Array.from(zBytes));
    const [ctx, ctxT] = await this.featureContext();
    const prior = await this.prior(z, ctxT, Math.ceil(height / 32), Math.ceil(width / 32));
    const indexes0 = scaleIndexes(maskedScales(prior.scales, 0),
                                  this.runtime.gaussian_index);
    const symbols0 = decoder.decodeY(indexes0, 0);
    const yhat0 = maskedRestore(symbols0, prior.means, 0);
    const next = await this.nextPrior(yhat0, prior.params);
    const indexes1 = scaleIndexes(maskedScales(next.scales, 1),
                                  this.runtime.gaussian_index);
    const symbols1 = decoder.decodeY(indexes1, 0);
    const yhat1 = maskedRestore(symbols1, next.means, 1);
    const yhat = multiplyByQuant(add(yhat0, yhat1), prior.qDec);
    return {reconstructed: await this.reconstruct(yhat, ctx)};
  }
}
