import createRans from '../vendor/mlvc/wasm/mlvc_rans.mjs';
import * as ort from '../vendor/ort/ort.all.mjs';
import {initRansWasm, FrameCodec} from '../vendor/mlvc/js/wasmEntropy.js';
import {MlvcFrameLoop} from '../vendor/mlvc/js/frameLoop.js';
import {RateController} from '../vendor/mlvc/js/rateController.js';
import {createSessions, setOrt} from '../vendor/mlvc/js/session.js';
import {rgbaToYuv420, yuv420ToRgba} from '../vendor/mlvc/js/yuv.js';
import {rgbaToYuv444, yuv420Psnr} from './color.js';

const GEOMETRY = {
  360: {dir: '640x368', modelW: 640, modelH: 368, videoW: 640, videoH: 360},
  540: {dir: '960x544', modelW: 960, modelH: 544, videoW: 960, videoH: 540},
  720: {dir: '1280x720', modelW: 1280, modelH: 720, videoW: 1280, videoH: 720},
  1080: {dir: '1920x1088', modelW: 1920, modelH: 1088, videoW: 1920, videoH: 1080},
};
const PARAMS = {featureChannels: 96, latentChannels: 48,
  frameIndexMap: [0, 1, 0, 2, 0, 2, 0, 2], qpShift: [0, 8, 4],
  iframePeriod: 150, resetPeriod: null, ltrStartIdx: 8, ltrPeriod: 64, fps: 30};

setOrt(ort);
ort.env.wasm.wasmPaths = new URL('../vendor/ort/', import.meta.url).href;
ort.env.wasm.numThreads = 1;

let loop = null;
let controller = null;
let geometry = null;
let requestedResolution = 360;
let modelFlavor = 'psnr';
let comparisonNote = '';
let actualBackend = '';
let actualDecoderBackend = '';
let targetKbps = 100;
let totalBits = 0;
let frameCount = 0;
let inputCanvas, inputContext, modelCanvas, modelContext;

function scaleRgba(rgba, fromW, fromH, toW, toH) {
  if (fromW === toW && fromH === toH) return rgba;
  inputCanvas ??= new OffscreenCanvas(fromW, fromH);
  if (inputCanvas.width !== fromW || inputCanvas.height !== fromH) {
    inputCanvas.width = fromW;
    inputCanvas.height = fromH;
  }
  inputContext ??= inputCanvas.getContext('2d', {willReadFrequently: true});
  modelCanvas ??= new OffscreenCanvas(toW, toH);
  if (modelCanvas.width !== toW || modelCanvas.height !== toH) {
    modelCanvas.width = toW;
    modelCanvas.height = toH;
  }
  modelContext ??= modelCanvas.getContext('2d', {willReadFrequently: true});
  inputContext.putImageData(new ImageData(rgba, fromW, fromH), 0, 0);
  modelContext.drawImage(inputCanvas, 0, 0, toW, toH);
  return modelContext.getImageData(0, 0, toW, toH).data;
}

self.onmessage = async event => {
  const message = event.data;
  try {
    if (message.type === 'init') {
      requestedResolution = message.resolution;
      modelFlavor = message.codec === 'mlvc-perceptual' ? 'perceptual' : 'psnr';
      geometry = GEOMETRY[message.resolution];
      if (!geometry) throw new Error('MLVC resolution unavailable');
      comparisonNote = [
        modelFlavor === 'perceptual'
          ? 'Perceptual weights target visual quality; PSNR can be lower at the same bitrate.' : '',
        requestedResolution === 540
          ? 'MLVC encodes 960 × 540 directly with four rows of bottom edge padding.' : '',
        requestedResolution === 1080 &&
          (message.encoderBackend === 'webgpu' || message.decoderBackend === 'webgpu')
          ? '1080p GPU sessions may exceed available memory.' : '',
      ].filter(Boolean).join(' ');
      targetKbps = message.targetKbps;
      const wasmResponse = await fetch(new URL('../vendor/ort/ort-wasm-simd-threaded.jsep.wasm.gz', import.meta.url));
      if (wasmResponse.ok) {
        ort.env.wasm.wasmBinary = new Uint8Array(await new Response(
          wasmResponse.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
      }
      const rans = await initRansWasm(createRans);
      const modelRoot = new URL('../vendor/mlvc/models/', import.meta.url);
      const profileRoot = modelFlavor === 'perceptual'
        ? new URL('perceptual/', modelRoot) : modelRoot;
      const shared = new URL('shared/', profileRoot);
      const [gaussian, z] = await Promise.all([
        fetch(new URL('gaussian_pmf.json', shared)).then(r => r.json()),
        fetch(new URL('bit_estimator_pmf.json', shared)).then(r => r.json()),
      ]);
      const sessions = await createSessions({
        modelDir: new URL(`${geometry.dir}/`, profileRoot).href.replace(/\/$/, ''),
        encoderDevice: message.encoderBackend === 'webgpu' ? 'gpu' : 'cpu',
        decoderDevice: message.decoderBackend === 'webgpu' ? 'gpu' : 'cpu',
        log: () => {},
      });
      actualBackend = sessions.encoderEp;
      actualDecoderBackend = sessions.decoderEp;
      const codec = new FrameCodec(rans, gaussian, z, PARAMS.latentChannels);
      controller = new RateController(geometry.videoW, geometry.videoH,
                                      targetKbps * 1000, 30);
      loop = new MlvcFrameLoop({encoderSession: sessions.encoder,
        decoderSession: sessions.decoder, frameCodec: codec,
        params: {...PARAMS, ...geometry, iframePeriod: message.intraPeriod || 150},
        rate: {mode: 'cbr', controller}});
      totalBits = frameCount = 0;
      self.postMessage({type: 'ready', codec: 'mlvc', backend: actualBackend,
        encoderBackend: actualBackend, decoderBackend: actualDecoderBackend,
        note: comparisonNote});
      return;
    }
    if (message.type === 'target' && controller) {
      targetKbps = Math.min(20000, Math.max(25, Number(message.targetKbps) || 100));
      controller.configure(targetKbps * 1000, 30);
      return;
    }
    if (message.type === 'reset' && loop) {
      controller = new RateController(geometry.videoW, geometry.videoH,
                                      targetKbps * 1000, 30);
      loop._rate.controller = controller;
      loop.reset();
      totalBits = frameCount = 0;
      self.postMessage({type: 'ready', codec: 'mlvc', backend: actualBackend,
        encoderBackend: actualBackend, decoderBackend: actualDecoderBackend,
        note: comparisonNote});
      return;
    }
    if (message.type === 'period' && loop) {
      loop._params.iframePeriod = Math.max(1, Number(message.intraPeriod) || 150);
      return;
    }
    if (message.type === 'stop') { loop?.reset(); return; }
    if (message.type === 'frame' && loop) {
      const original = new Uint8ClampedArray(message.rgba);
      const source = scaleRgba(original, message.width, message.height,
                               geometry.videoW, geometry.videoH);
      const result = await loop.processFrame(rgbaToYuv420(source,
        geometry.videoW, geometry.videoH));
      const recovered = yuv420ToRgba(result.reconstructed,
        geometry.videoW, geometry.videoH);
      const display = scaleRgba(recovered, geometry.videoW, geometry.videoH,
                                message.width, message.height);
      const a = rgbaToYuv444(original, message.width, message.height, message.height);
      const b = rgbaToYuv444(display, message.width, message.height, message.height);
      const [psnrY, psnrU, psnrV] = yuv420Psnr(a, b, message.width, message.height);
      totalBits += result.bits;
      frameCount++;
      const rgba = new Uint8ClampedArray(display);
      self.postMessage({type: 'frame', codec: 'mlvc', rgba: rgba.buffer,
        bits: totalBits, frames: frameCount, targetKbps,
        actualKbps: totalBits * 30 / frameCount / 1000,
        psnrY, psnrU, psnrV, qp: result.qIndex, rateMode: 'mlvc-cbr',
        dropped: result.dropped}, [rgba.buffer]);
    }
  } catch (error) {
    self.postMessage({type: 'error', message: error.stack || String(error)});
  }
};
