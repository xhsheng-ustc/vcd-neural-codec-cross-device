import * as ort from '../vendor/ort/ort.all.mjs';
import {IFrameCodec, setFixedCdfIndex, setFixedCdfTables,
  setCdfFrame, getCdfStatistics, getCdfGuardPasses, getCdfMaps,
  getCdfSyndromeEvents, getCdfSyndromeDecodeMs} from './iframe.js';
import {PFrameCodec, setStateGrid, setFeatureGrid} from './pframe.js';
import {rgbaToYuv444, yuv444ToRgba, yuv420Psnr} from './color.js';
import {RateController} from './rate_control.js';
import {packCdfGuard, unpackCdfGuard} from './cdf_guard.js';
import {cdfSyndrome, splitCdfIndexes, packCdfSyndromes, unpackCdfSyndromes} from './cdf_syndrome.js';
import {packFeatureSyndrome, unpackFeatureSyndrome,
        repairFeatureSyndrome} from './feature_syndrome.js';

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
let fastEntropyPolicy = null;
let guardThreshold = 0.001;
let referenceFeatureGrid = 0;
let referenceFeatureGridInterval = 1;
let referenceFeatureSyndrome = false;
let referenceFeatureSyndromeBlocks = 64;

function log(message) { self.postMessage({type: 'progress', message}); }

async function initialize(requestedEncode, requestedDecode, resolution,
                          allowExperimentalMixed = false,
                          portable = false, fusedPointwise = true,
                          fusedPointwiseP = true, fusedSpatial = true,
                          packedPointwise = false, fastPortable = false,
                          staticCdfIndex = 64, staticCdfTables = null,
                          entropyPolicy = 'static', requestedGuardThreshold = 0.001,
                          requestedStateGrid = 0, requestedFeatureGrid = 0,
                          requestedFeatureGridInterval = 1,
                          requestedFeatureSyndrome = false,
                          requestedFeatureSyndromeBlocks = 64) {
  if (!SIZES[resolution]) throw new Error('Unsupported resolution');
  if (!['webgpu', 'wasm'].includes(requestedEncode) ||
      !['webgpu', 'wasm'].includes(requestedDecode)) throw new Error('Unsupported compute device');
  if (requestedEncode !== requestedDecode && !portable && !fastPortable && !allowExperimentalMixed) {
    throw new Error('Mixed GPU/CPU execution requires fast cross-device mode.');
  }
  [width, height, modelHeight] = SIZES[resolution];
  fastEntropyPolicy = fastPortable ? entropyPolicy : null;
  setStateGrid(fastPortable ? requestedStateGrid : 0);
  referenceFeatureGrid = fastPortable ? requestedFeatureGrid : 0;
  referenceFeatureGridInterval = requestedFeatureGridInterval || 1;
  referenceFeatureSyndrome = fastPortable && requestedFeatureSyndrome;
  referenceFeatureSyndromeBlocks = requestedFeatureSyndromeBlocks;
  if (![8, 16, 32, 64].includes(referenceFeatureSyndromeBlocks))
    throw new Error('Feature syndrome blocks must be 8, 16, 32 or 64');
  setFeatureGrid(0);
  guardThreshold = requestedGuardThreshold;
  setFixedCdfIndex(fastEntropyPolicy === 'static' ? staticCdfIndex : null);
  setFixedCdfTables(fastEntropyPolicy === 'static' ? staticCdfTables : null);
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
  const featureCheckpoint = referenceFeatureGrid > 0 &&
    (frameNumber + 1) % referenceFeatureGridInterval === 0;
  setFeatureGrid(featureCheckpoint ? referenceFeatureGrid : 0);
  const featureResetInterval = Number(message.featureResetInterval) || 0;
  if (!isI && featureResetInterval > 0 && frameNumber % featureResetInterval === 0) {
    pEncoder.refFeature = null;
    pDecoder.refFeature = null;
  }
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
  const useGuard = fastEntropyPolicy?.includes('guard');
  const useSyndrome = fastEntropyPolicy?.includes('syndrome');
  const syndromeStripes = Number(fastEntropyPolicy?.match(/^syndrome([248])h?$/)?.[1] ?? 1);
  const syndromeHashed = fastEntropyPolicy?.endsWith('h') ?? false;
  setCdfFrame(isI ? 'i' : 'p', message.probeIndex === true,
    useGuard ? 'encode' : null, [], guardThreshold,
    message.probeIndexMaps === true || useSyndrome);
  const encoded = isI ? await iEncoder.encode(input) : await pEncoder.encode(input);
  const encodeIndexStats = message.probeIndex ? getCdfStatistics() : null;
  const encodeIndexMaps = message.probeIndexMaps || useSyndrome ? getCdfMaps() : null;
  let encodedStream = useGuard
    ? packCdfGuard(getCdfGuardPasses(), encoded.stream).stream : encoded.stream;
  if (referenceFeatureSyndrome && featureCheckpoint && !isI)
    encodedStream = packFeatureSyndrome(pEncoder.refFeature,
      referenceFeatureGrid, encodedStream, referenceFeatureSyndromeBlocks).stream;
  const syndromeStarted = performance.now();
  if (useSyndrome) encodedStream = packCdfSyndromes(
    encodeIndexMaps.flatMap(indexes =>
      (syndromeStripes === 1 ? [indexes] : splitCdfIndexes(indexes, syndromeStripes, syndromeHashed))
        .map(cdfSyndrome)), encodedStream).stream;
  const syndromeEncodeMs = performance.now() - syndromeStarted;
  const encodeProfile = message.profile ? encoderRuntime?.profile?.slice() : null;
  if (message.profile && decoderRuntime) decoderRuntime.profile = [];
  const encodedAt = performance.now();
  // The Uint8Array is the real rANS bitstream. Decode only from these bytes.
  const syndrome = useSyndrome
    ? unpackCdfSyndromes(encodedStream, (isI ? 4 : 2) * syndromeStripes) : null;
  const featurePacket = referenceFeatureSyndrome && featureCheckpoint && !isI
    ? unpackFeatureSyndrome(syndrome?.payload ?? encodedStream,
      referenceFeatureSyndromeBlocks) : null;
  const guarded = useGuard
    ? unpackCdfGuard(featurePacket?.payload ?? syndrome?.payload ?? encodedStream,
      isI ? 4 : 2) : null;
  setCdfFrame(isI ? 'i' : 'p', message.probeIndex === true,
    guarded ? 'decode' : null, guarded?.passes ?? [], 0,
    message.probeIndexMaps === true, useSyndrome ? 'decode' : null,
    syndrome?.checksums ?? [], message.probeIndexMaps ? encodeIndexMaps : null,
    syndromeStripes, syndromeHashed);
  const decoded = isI
    ? await iDecoder.decode(guarded?.payload ?? featurePacket?.payload ??
        syndrome?.payload ?? encodedStream, modelHeight, width)
    : await pDecoder.decode(guarded?.payload ?? featurePacket?.payload ??
        syndrome?.payload ?? encodedStream, modelHeight, width);
  const featureRepairStarted = performance.now();
  const featureEvents = featurePacket ? repairFeatureSyndrome(
    pDecoder.refFeature, pDecoder.rawFeature, referenceFeatureGrid,
    featurePacket.checksums, referenceFeatureSyndromeBlocks) : [];
  const featureRepairMs = performance.now() - featureRepairStarted;
  const decodeProfile = message.profile ? decoderRuntime?.profile?.slice() : null;
  const decodeIndexStats = message.probeIndex ? getCdfStatistics() : null;
  const decodeIndexMaps = message.probeIndexMaps ? getCdfMaps() : null;
  const syndromeEvents = useSyndrome ? getCdfSyndromeEvents() : null;
  const syndromeDecodeMs = useSyndrome ? getCdfSyndromeDecodeMs() : 0;
  const decodedAt = performance.now();
  let probe = null;
  if (message.probeHash) {
    const digest = async bytes => Array.from(new Uint8Array(
      await crypto.subtle.digest('SHA-256', bytes))).map(x => x.toString(16).padStart(2, '0')).join('');
    probe = {streamHash: await digest(encodedStream),
      reconstructedHash: await digest(decoded.reconstructed.data.buffer)};
    if (message.traceStages && isI)
      probe.stages = {encode: iEncoder.trace, decode: iDecoder.trace};
    if (message.traceStages && isI && iEncoder.portable && iDecoder.portable)
      probe.layers = {encode: iEncoder.portable.trace,
        decode: iDecoder.portable.trace};
    if (message.profile) probe.profile = {encode: encodeProfile,
      decode: decodeProfile};
    if (message.probeIndex) probe.indexStats = {encode: encodeIndexStats,
      decode: decodeIndexStats};
    if (useSyndrome) {
      probe.syndromeEvents = syndromeEvents;
      probe.syndromeMs = {encode: syndromeEncodeMs, decode: syndromeDecodeMs};
    }
    if (featurePacket) probe.featureSyndrome = {
      headerBytes: featurePacket.headerBytes, events: featureEvents,
      repairMs: featureRepairMs};
    if (message.probeIndexMaps) probe.indexMismatches = encodeIndexMaps.map(
      (indexes, pass) => {
        const other = decodeIndexMaps[pass];
        let count = 0, first = -1;
        for (let i = 0; i < indexes.length; i++) if (indexes[i] !== other[i]) {
          count++;
          if (first < 0) first = i;
        }
        return {pass, count, first};
      });
    if (message.probeState) {
      const difference = (a, b) => {
        if (!a || !b || a.length !== b.length) return null;
        let max = 0, sum = 0, aboveGrid = 0;
        const grids = [32, 64, 128, 256, 512, 1024];
        const snappedDifferences = new Uint32Array(grids.length);
        for (let i = 0; i < a.length; i++) {
          const delta = Math.abs(a[i] - b[i]);
          max = Math.max(max, delta);
          sum += delta;
          if (delta > 1 / 256) aboveGrid++;
          for (let g = 0; g < grids.length; g++)
            if (Math.round(a[i] * grids[g]) !== Math.round(b[i] * grids[g]))
              snappedDifferences[g]++;
        }
        return {max, mean: sum / a.length, aboveGrid, elements: a.length,
          snappedDifferences: Object.fromEntries(grids.map((grid, g) =>
            [grid, snappedDifferences[g]]))};
      };
      probe.state = {
        feature: difference(pEncoder.refFeature?.data, pDecoder.refFeature?.data),
        frame: difference(encoded.reconstructed.data, decoded.reconstructed.data),
      };
    }
  }
  if (message.oracleFeatureSync && featureCheckpoint && !isI) {
    const sender = pEncoder.refFeature?.data;
    const receiver = pDecoder.refFeature?.data;
    if (!sender || !receiver || sender.length !== receiver.length)
      throw new Error('Feature synchronization state missing');
    let corrections = 0, estimatedBytes = 0, previous = -1;
    const thresholds = [0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002];
    const candidates = new Uint32Array(thresholds.length);
    const covered = new Uint32Array(thresholds.length);
    let farthestMismatch = 0;
    const raw = pEncoder.rawFeature;
    const varintLength = value => {
      let bytes = 1;
      while (value >= 128) {value = Math.floor(value / 128); bytes++;}
      return bytes;
    };
    for (let i = 0; i < sender.length; i++) {
      const coordinate = raw[i] * referenceFeatureGrid;
      const distance = Math.abs(coordinate - Math.floor(coordinate) - 0.5);
      const differing = sender[i] !== receiver[i];
      for (let t = 0; t < thresholds.length; t++)
        if (distance <= thresholds[t]) {
          candidates[t]++;
          if (differing) covered[t]++;
        }
      if (differing) {
        corrections++;
        farthestMismatch = Math.max(farthestMismatch, distance);
        const integer = Math.round(coordinate);
        const zigzag = integer >= 0 ? integer * 2 : -integer * 2 - 1;
        estimatedBytes += varintLength(i - previous - 1) + varintLength(zigzag);
        previous = i;
        receiver[i] = sender[i];
      }
    }
    if (probe) probe.oracleFeatureSync = {corrections,
      estimatedBytes: estimatedBytes + 2, elements: sender.length,
      farthestMismatch, thresholds: Object.fromEntries(thresholds.map((threshold, t) =>
        [threshold, {candidates: candidates[t], covered: covered[t]}]))};
  }
  if (encoderRuntime) encoderRuntime.profile = null;
  if (decoderRuntime) decoderRuntime.profile = null;
  iEncoder.trace = null;
  iDecoder.trace = null;
  if (iEncoder.portable) iEncoder.portable.trace = null;
  if (iDecoder.portable) iDecoder.portable.trace = null;
  if (isI) {
    pEncoder.setReference(encoded.reconstructed);
    pDecoder.setReference(decoded.reconstructed);
  }
  const rgba = yuv444ToRgba(decoded.reconstructed, width, height);
  const psnr = yuv420Psnr(input, decoded.reconstructed, width, height);
  if (rateController) rateController.update(isI, period, encodedStream.length * 8);
  frameNumber++;
  forceI = false;
  self.postMessage({type: 'frame', rgba: rgba.buffer, frameType: isI ? 'I' : 'P',
    bytes: encodedStream.length, encodeMs: encodedAt - start,
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
      message.fusedSpatial !== false, message.packedPointwise === true,
      message.fastPortable === true, message.staticCdfIndex ?? 64,
      message.staticCdfTables ?? null, message.entropyPolicy ?? 'static',
      message.guardThreshold ?? 0.001, message.stateGrid ?? 0,
      message.featureGrid ?? 0, message.featureGridInterval ?? 1,
      message.featureSyndrome === true,
      message.featureSyndromeBlocks ?? 64);
    if (message.type === 'frame') await processFrame(message);
    if (message.type === 'forceI') forceI = true;
  } catch (error) {
    self.postMessage({type: 'error', message: error.stack || String(error)});
  }
};
