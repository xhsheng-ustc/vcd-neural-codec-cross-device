import {rgbaToYuv444, yuv420Psnr} from './color.js';

const CANDIDATES = {
  h264: ['avc1.640028', 'avc1.4d4028', 'avc1.42e028'],
  hevc: ['hvc1.1.6.L120.B0', 'hvc1.1.6.L93.B0'],
  av1: ['av01.0.08M.08', 'av01.0.04M.08'],
  vp8: ['vp8'],
};
const QUANTIZER = {
  h264: {option: 'avc', max: 51, initial: 40, gain: 4},
  hevc: {option: 'hevc', max: 51, initial: 40, gain: 4},
  av1: {option: 'av1', max: 255, initial: 180, gain: 16},
};
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

class AdaptiveQuantizer {
  constructor(targetKbps, settings) {
    this.targetKbps = targetKbps;
    this.settings = settings;
    this.qp = settings.initial;
    this.average = {key: null, delta: null};
    this.bufferBits = 0;
  }

  update(bits, isKey) {
    const normal = this.targetKbps * 1000 / 30 * 30 / 37;
    const share = isKey ? normal * 8 : normal;
    const target = clamp(share - this.bufferBits / 20, share * 0.25, share * 4);
    const type = isKey ? 'key' : 'delta';
    this.average[type] = this.average[type] === null
      ? bits : 0.25 * bits + 0.75 * this.average[type];
    const ratio = Math.max(1, this.average[type]) / Math.max(1, target);
    const step = clamp(Math.round(this.settings.gain * Math.log2(ratio)),
                       -this.settings.gain * 2, this.settings.gain * 2);
    // Conventional codec QP has the opposite direction from our neural QP.
    this.qp = clamp(this.qp + step, 0, this.settings.max);
    const nominal = this.targetKbps * 1000 / 30;
    this.bufferBits = clamp(this.bufferBits + bits - nominal,
                            -nominal * 30, nominal * 30);
  }
}

let encoder = null;
let decoder = null;
let codecName = '';
let codecId = '';
let width = 0;
let height = 0;
let targetKbps = 100;
let rateMode = 'constant';
let quantizer = null;
let sourceFrames = new Map();
let qpByTimestamp = new Map();
let frameCount = 0;
let totalBits = 0;
let decodedCount = 0;
let decodedBits = 0;
let bitsByTimestamp = new Map();
let canvas;
let context;
let sourceCanvas;
let sourceContext;

function close() {
  try { encoder?.close(); } catch {}
  try { decoder?.close(); } catch {}
  encoder = decoder = null;
  sourceFrames.clear();
  bitsByTimestamp.clear();
  qpByTimestamp.clear();
  quantizer = null;
}

async function chooseCodec(name, kbps) {
  if (!globalThis.VideoEncoder || !globalThis.VideoDecoder) {
    throw new Error('WebCodecs video encoding is unavailable in this browser');
  }
  for (const codec of CANDIDATES[name] || []) {
    try {
      const dec = await VideoDecoder.isConfigSupported({codec, codedWidth: width,
                                                       codedHeight: height});
      if (!dec.supported) continue;
      if (QUANTIZER[name]) {
        const config = {codec, width, height, framerate: 30,
          bitrateMode: 'quantizer', latencyMode: 'realtime'};
        if (name === 'h264') config.avc = {format: 'annexb'};
        const enc = await VideoEncoder.isConfigSupported(config);
        if (enc.supported) return {codec, config: enc.config, bitrateMode: 'quantizer'};
      }
    } catch {}
    for (const bitrateMode of ['constant', 'variable']) {
      const config = {codec, width, height, bitrate: kbps * 1000,
                      framerate: 30, bitrateMode, latencyMode: 'realtime'};
      if (name === 'h264') config.avc = {format: 'annexb'};
      try {
        const enc = await VideoEncoder.isConfigSupported(config);
        const dec = await VideoDecoder.isConfigSupported({codec, codedWidth: width,
                                                         codedHeight: height});
        if (enc.supported && dec.supported) {
          return {codec, config: enc.config, bitrateMode};
        }
      } catch {}
    }
  }
  throw new Error(`${name.toUpperCase()} encoding/decoding is not supported here`);
}

async function outputFrame(frame) {
  try {
    const original = sourceFrames.get(frame.timestamp);
    if (!original) return;
    sourceFrames.delete(frame.timestamp);
    const qp = qpByTimestamp.get(frame.timestamp);
    qpByTimestamp.delete(frame.timestamp);
    decodedBits += bitsByTimestamp.get(frame.timestamp) || 0;
    bitsByTimestamp.delete(frame.timestamp);
    context.drawImage(frame, 0, 0, width, height);
    const rgba = context.getImageData(0, 0, width, height).data;
    const a = rgbaToYuv444(original, width, height, height);
    const b = rgbaToYuv444(rgba, width, height, height);
    const [psnrY, psnrU, psnrV] = yuv420Psnr(a, b, width, height);
    decodedCount++;
    const result = new Uint8ClampedArray(rgba);
    self.postMessage({type: 'frame', codec: codecName, rgba: result.buffer,
      bits: decodedBits, frames: decodedCount, targetKbps,
      actualKbps: decodedBits * 30 / decodedCount / 1000,
      psnrY, psnrU, psnrV, qp, rateMode}, [result.buffer]);
  } catch (error) {
    self.postMessage({type: 'error', message: error.message || String(error)});
  } finally { frame.close(); }
}

self.onmessage = async event => {
  const message = event.data;
  try {
    if (message.type === 'stop') { close(); return; }
    if (message.type === 'init') {
      close();
      codecName = message.codec;
      width = message.width;
      height = message.height;
      targetKbps = message.targetKbps;
      totalBits = decodedBits = decodedCount = frameCount = 0;
      canvas = new OffscreenCanvas(width, height);
      context = canvas.getContext('2d', {willReadFrequently: true});
      sourceCanvas = new OffscreenCanvas(width, height);
      sourceContext = sourceCanvas.getContext('2d');
      const selected = await chooseCodec(codecName, targetKbps);
      codecId = selected.codec;
      rateMode = selected.bitrateMode;
      quantizer = rateMode === 'quantizer'
        ? new AdaptiveQuantizer(targetKbps, QUANTIZER[codecName]) : null;
      decoder = new VideoDecoder({output: outputFrame,
        error: error => self.postMessage({type: 'error', message: String(error)})});
      encoder = new VideoEncoder({output: (chunk, metadata) => {
        try {
          if (metadata?.decoderConfig) decoder.configure(metadata.decoderConfig);
          else if (decoder.state === 'unconfigured') decoder.configure({codec: codecId,
            codedWidth: width, codedHeight: height});
          totalBits += chunk.byteLength * 8;
          quantizer?.update(chunk.byteLength * 8, chunk.type === 'key');
          bitsByTimestamp.set(chunk.timestamp, (bitsByTimestamp.get(chunk.timestamp) || 0) +
            chunk.byteLength * 8);
          decoder.decode(chunk);
        } catch (error) {
          self.postMessage({type: 'error', message: error.message || String(error)});
        }
      }, error: error => self.postMessage({type: 'error', message: String(error)})});
      encoder.configure(selected.config);
      self.postMessage({type: 'ready', codec: codecName, codecId,
        note: rateMode === 'quantizer'
          ? 'Adaptive per-frame QP is active.'
          : codecName === 'vp8'
            ? 'WebCodecs does not expose per-frame VP8 QP; browser CBR may miss the target.'
            : rateMode === 'variable'
              ? 'This browser provides variable-bitrate encoding for this codec.' : ''});
      return;
    }
    if (message.type === 'target' && encoder) {
      const next = Math.min(20000, Math.max(25, Number(message.targetKbps) || 100));
      if (rateMode === 'quantizer') {
        quantizer.targetKbps = next;
        targetKbps = next;
      } else if (Math.abs(next - targetKbps) / targetKbps > 0.15) {
        const selected = await chooseCodec(codecName, next);
        encoder.configure(selected.config);
        targetKbps = next;
      }
      return;
    }
    if (message.type === 'frame' && encoder) {
      if (encoder.encodeQueueSize > 2) {
        self.postMessage({type: 'skipped'});
        return;
      }
      const rgba = new Uint8ClampedArray(message.rgba);
      const timestamp = message.frameId * 33333;
      sourceFrames.set(timestamp, rgba);
      if (sourceFrames.size > 32) {
        const oldest = sourceFrames.keys().next().value;
        sourceFrames.delete(oldest);
        qpByTimestamp.delete(oldest);
        bitsByTimestamp.delete(oldest);
      }
      const image = new ImageData(rgba, width, height);
      sourceContext.putImageData(image, 0, 0);
      const frame = new VideoFrame(sourceCanvas, {timestamp});
      const qp = quantizer?.qp;
      if (qp !== undefined) qpByTimestamp.set(timestamp, qp);
      const options = {keyFrame: frameCount % 30 === 0};
      if (quantizer) options[QUANTIZER[codecName].option] = {quantizer: qp};
      encoder.encode(frame, options);
      frame.close();
      frameCount++;
    }
  } catch (error) {
    self.postMessage({type: 'error', message: error.message || String(error)});
  }
};
