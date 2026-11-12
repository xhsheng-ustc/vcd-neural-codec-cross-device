import * as ort from '../vendor/ort/ort.all.mjs';
import {IFrameCodec} from './iframe.js';
import {PFrameCodec} from './pframe.js';
import {rgbaToYuv444, yuv444ToRgba, yuv420Psnr} from './color.js';
import {RateController} from './rate_control.js';

const SIZES = {360: [640, 360, 384], 540: [960, 540, 576],
  720: [1280, 720, 768], 1080: [1920, 1080, 1088]};
let width = 640;
let height = 360;
let modelHeight = 384;
const BASE = new URL('../models/dynamic', import.meta.url).href;
ort.env.wasm.wasmPaths = new URL('../vendor/ort/', import.meta.url).href;
ort.env.wasm.numThreads = 1;

let iEncoder, iDecoder, pEncoder, pDecoder;
let encodeBackend = 'wasm';
let decodeBackend = 'wasm';
let frameNumber = 0;
let previousQpi = 63;
let previousQpp = 63;
let forceI = true;
let rateController = null;
let previousMode = 'cq';

function log(message) { self.postMessage({type: 'progress', message}); }

async function initialize(requestedEncode, requestedDecode, resolution,
                          allowExperimentalMixed = false,
                          portable = false, fusedPointwise = true,
                          fusedPointwiseP = true, fusedSpatial = true,
                          packedPointwise = false) {
  if (!SIZES[resolution]) throw new Error('Unsupported resolution');
  if (!['webgpu', 'wasm'].includes(requestedEncode) ||
      !['webgpu', 'wasm'].includes(requestedDecode)) throw new Error('Unsupported compute device');
  if (requestedEncode !== requestedDecode && !portable && !allowExperimentalMixed) {
    throw new Error('Mixed GPU/CPU execution requires the portable fixed-point browser codec; the current floating-point ONNX models diverge after P frames.');
  }
  [width, height, modelHeight] = SIZES[resolution];
  const runtimeFile = 'ort-wasm-simd-threaded.jsep.wasm';
  const runtimeUrl = new URL(`../vendor/ort/${runtimeFile}`, import.meta.url);
  const compressed = await fetch(`${runtimeUrl}.gz`);
  if (compressed.ok) {
    ort.env.wasm.wasmBinary = new Uint8Array(await new Response(
      compressed.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
  }
  const loadSide = async (requested, role) => {
    try {
      const portableBackend = portable ? (requested === 'webgpu' ? 'webgpu' : 'cpu') : null;
      return {i: await IFrameCodec.load(ort, BASE, 63, requested, log, role, portableBackend),
        p: await PFrameCodec.load(ort, BASE, 63, requested, log, role, portableBackend),
        backend: requested};
    } catch (error) {
      if (requested !== 'webgpu') throw error;
      log(`WebGPU ${role} failed; switching to CPU: ${error.message}`);
      return {i: await IFrameCodec.load(ort, BASE, 63, 'wasm', log, role,
        portable ? 'cpu' : null),
        p: await PFrameCodec.load(ort, BASE, 63, 'wasm', log, role,
          portable ? 'cpu' : null),
        backend: 'wasm'};
    }
  };
  const encoder = await loadSide(requestedEncode, 'all');
  const decoder = requestedEncode === requestedDecode
    ? encoder : await loadSide(requestedDecode, 'decoder');
  for (const side of [encoder, decoder]) {
    if (side.i.portable) side.i.portable.fusePointwise = fusedPointwise;
    if (side.p.portable) side.p.portable.fusePointwise = fusedPointwiseP;
    if (side.i.portable) side.i.portable.fuseSpatial = fusedSpatial;
    if (side.p.portable) side.p.portable.fuseSpatial = fusedSpatial;
    if (side.i.portable) side.i.portable.packedPointwise = packedPointwise;
  }
  iEncoder = encoder.i;
  pEncoder = encoder.p;
  iDecoder = new IFrameCodec(ort, decoder.i.sessions, decoder.i.runtime, 63,
    decoder.i.portable);
  pDecoder = new PFrameCodec(ort, decoder.p.sessions, decoder.p.runtime, 63,
    decoder.p.portable);
  encodeBackend = encoder.backend;
  decodeBackend = decoder.backend;
  frameNumber = 0;
  forceI = true;
  rateController = null;
  previousMode = 'cq';
  self.postMessage({type: 'ready', encodeBackend, decodeBackend,
    width, height, resolution});
}

async function processFrame(message) {
  const period = Math.max(1, Math.floor(Number(message.intraPeriod) || 150));
  const mode = message.mode === 'cbr' ? 'cbr' : 'cq';
  if (mode !== previousMode) {
    rateController = null;
    previousMode = mode;
    forceI = true;
  }
  if (mode === 'cbr') {
    const target = Math.min(20000, Math.max(25, Number(message.targetKbps) || 100));
    if (!rateController || rateController.targetKbps !== target) {
      rateController = new RateController(target, previousQpi, previousQpp);
      forceI = true;
    }
  }
  const qpi = mode === 'cbr' ? rateController.choose(true)
    : Math.min(63, Math.max(0, Number(message.qpi) || 0));
  const qpp = mode === 'cbr' ? rateController.choose(false)
    : Math.min(63, Math.max(0, Number(message.qpp) || 0));
  if (qpi !== previousQpi || qpp !== previousQpp) {
    if (qpi !== previousQpi) {
      iEncoder = new IFrameCodec(ort, iEncoder.sessions, iEncoder.runtime, qpi,
        iEncoder.portable);
      iDecoder = new IFrameCodec(ort, iDecoder.sessions, iDecoder.runtime, qpi,
        iDecoder.portable);
    }
    pEncoder.setQp(qpp);
    pDecoder.setQp(qpp);
    previousQpi = qpi;
    previousQpp = qpp;
    if (mode === 'cq') forceI = true;
  }
  const isI = forceI || frameNumber % period === 0;
  const input = rgbaToYuv444(new Uint8ClampedArray(message.rgba), width, height, modelHeight);
  const encoderRuntime = isI ? iEncoder.portable : pEncoder.portable;
  const decoderRuntime = isI ? iDecoder.portable : pDecoder.portable;
  if (message.profile) {
    if (encoderRuntime) encoderRuntime.profile = [];
    if (decoderRuntime) decoderRuntime.profile = [];
  }
  if (message.traceStages && isI) {
    iEncoder.trace = [];
    iDecoder.trace = [];
    if (iEncoder.portable) iEncoder.portable.trace = [];
    if (iDecoder.portable) iDecoder.portable.trace = [];
  }
  const start = performance.now();
  const encoded = isI ? await iEncoder.encode(input) : await pEncoder.encode(input);
  const encodeProfile = message.profile ? encoderRuntime?.profile?.slice() : null;
  if (message.profile && decoderRuntime) decoderRuntime.profile = [];
  const encodedAt = performance.now();
  // The Uint8Array is the real rANS bitstream. Decode only from these bytes.
  const decoded = isI
    ? await iDecoder.decode(encoded.stream, modelHeight, width)
    : await pDecoder.decode(encoded.stream, modelHeight, width);
  const decodeProfile = message.profile ? decoderRuntime?.profile?.slice() : null;
  const decodedAt = performance.now();
  let probe = null;
  if (message.probeHash) {
    const digest = async bytes => Array.from(new Uint8Array(
      await crypto.subtle.digest('SHA-256', bytes))).map(x => x.toString(16).padStart(2, '0')).join('');
    probe = {streamHash: await digest(encoded.stream),
      reconstructedHash: await digest(decoded.reconstructed.data.buffer)};
    if (message.traceStages && isI)
      probe.stages = {encode: iEncoder.trace, decode: iDecoder.trace};
    if (message.traceStages && isI && iEncoder.portable && iDecoder.portable)
      probe.layers = {encode: iEncoder.portable.trace,
        decode: iDecoder.portable.trace};
    if (message.profile) probe.profile = {encode: encodeProfile,
      decode: decodeProfile};
  }
  if (encoderRuntime) encoderRuntime.profile = null;
  if (decoderRuntime) decoderRuntime.profile = null;
  iEncoder.trace = null;
  iDecoder.trace = null;
  if (iEncoder.portable) iEncoder.portable.trace = null;
  if (iDecoder.portable) iDecoder.portable.trace = null;
  if (isI) {
    pEncoder.setReference(decoded.reconstructed);
    pDecoder.setReference(decoded.reconstructed);
  }
  const rgba = yuv444ToRgba(decoded.reconstructed, width, height);
  const psnr = yuv420Psnr(input, decoded.reconstructed, width, height);
  if (rateController) rateController.update(isI, period, encoded.stream.length * 8);
  frameNumber++;
  forceI = false;
  self.postMessage({type: 'frame', rgba: rgba.buffer, frameType: isI ? 'I' : 'P',
    bytes: encoded.stream.length, encodeMs: encodedAt - start,
    decodeMs: decodedAt - encodedAt, psnrY: psnr[0], psnrU: psnr[1], psnrV: psnr[2],
    frameNumber, encodeBackend, decodeBackend, width, height,
    qp: isI ? qpi : qpp, probe}, [rgba.buffer]);
}

self.onmessage = async event => {
  try {
    const message = event.data;
    if (message.type === 'init') await initialize(
      message.encodeBackend, message.decodeBackend, message.resolution,
      message.allowExperimentalMixed, message.portable,
      message.fusedPointwise !== false, message.fusedPointwiseP !== false,
      message.fusedSpatial !== false, message.packedPointwise === true);
    if (message.type === 'frame') await processFrame(message);
    if (message.type === 'forceI') forceI = true;
  } catch (error) {
    self.postMessage({type: 'error', message: error.stack || String(error)});
  }
};
