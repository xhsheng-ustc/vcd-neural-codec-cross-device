const SIZES = {360: [640, 360], 540: [960, 540],
  720: [1280, 720], 1080: [1920, 1080]};
const $ = id => document.getElementById(id);
const isMlvcCodec = codec => codec === 'mlvc' || codec === 'mlvc-perceptual';
const params = new URL(location.href).searchParams;
const resolution = SIZES[params.get('resolution')] ? Number(params.get('resolution')) : 540;
const [WIDTH, HEIGHT] = SIZES[resolution];
const input = $('input-view'), output = $('output-view');
input.width = output.width = WIDTH;
input.height = output.height = HEIGHT;
const inputCtx = input.getContext('2d', {willReadFrequently: true});
const outputCtx = output.getContext('2d');
const comparisonView = $('comparison-view');
comparisonView.width = WIDTH;
comparisonView.height = HEIGHT;
const comparisonCtx = comparisonView.getContext('2d');
const camera = $('camera');
const worker = new Worker('./src/codec.worker.js', {type: 'module'});
let ready = false, busy = false, cameraStream = null, cameraTimer = null;
let totalBytes = 0, totalFrames = 0, captureStartedAt = 0, lastFrameAt = 0;
let compareWorker = null, compareReady = false, compareBusy = false;
let compareFrameId = 0, compareTarget = 300;
let compareBackend = '';
let pendingOwnFrame = null;
const backendLabel = backend => backend === 'wasm' ? 'CPU / WASM'
  : backend === 'webgpu' ? 'GPU / WebGPU' : (backend || 'unknown').toUpperCase();

$('resolution').value = String(resolution);
$('source-size').textContent = `SOURCE · ${WIDTH} × ${HEIGHT}`;
$('decoded-size').textContent = `DECODED · ${WIDTH} × ${HEIGHT}`;
$('comparison-size').textContent = `COMPARISON · ${WIDTH} × ${HEIGHT}`;

function rateTarget() {
  if ($('rate-mode').value === 'cbr') {
    return Math.min(20000, Math.max(25, Number($('target-kbps').value) || 100));
  }
  return Math.min(20000, Math.max(25, Number($('bitrate-30').textContent) || 300));
}

function formatPsnr(value) {
  return Number.isFinite(value) ? value.toFixed(2) : '∞';
}

function resetStatistics() {
  totalBytes = totalFrames = lastFrameAt = 0;
  captureStartedAt = performance.now();
  for (const id of ['bitrate', 'bitrate-30', 'fps', 'encode-time', 'decode-time',
                     'frame-size', 'psnr-y', 'psnr-u', 'psnr-v', 'frame-qp',
                     'target-actual']) $(id).textContent = '—';
  $('rate-limit-note').textContent = '';
}

function setComparison() {
  if (pendingOwnFrame) flushPendingOwnFrame();
  compareWorker?.terminate();
  compareWorker = null;
  compareReady = compareBusy = false;
  compareBackend = '';
  compareFrameId = 0;
  const codec = $('compare-codec').value;
  const active = Boolean(codec);
  const mlvc = isMlvcCodec(codec);
  $('mlvc-encoder-setting').classList.toggle('hidden', !mlvc);
  $('mlvc-decoder-setting').classList.toggle('hidden', !mlvc);
  $('comparison-card').classList.toggle('hidden', !active);
  $('comparison-metrics').classList.toggle('hidden', !active);
  document.querySelector('.video-grid').classList.toggle('comparing', active);
  if (!active) return;
  const label = $('compare-codec').selectedOptions[0].textContent;
  $('comparison-title').textContent = label;
  $('comparison-metrics-title').textContent = `${label} · live statistics`;
  $('compare-status').textContent = 'Loading';
  $('comparison-note').textContent = '';
  for (const id of ['compare-target-actual', 'compare-psnr-y',
                     'compare-psnr-u', 'compare-psnr-v', 'compare-qp']) $(id).textContent = '—';
  comparisonCtx.clearRect(0, 0, WIDTH, HEIGHT);
  compareTarget = rateTarget();
  const workerPath = isMlvcCodec(codec) ? './src/mlvc_compare.worker.js'
    : './src/webcodecs_compare.worker.js';
  compareWorker = new Worker(workerPath, {type: 'module'});
  const thisWorker = compareWorker;
  thisWorker.onmessage = event => {
    if (thisWorker !== compareWorker) return;
    const message = event.data;
    if (message.type === 'ready') {
      compareReady = true;
      compareBusy = false;
      compareBackend = mlvc
        ? `${backendLabel(message.encoderBackend)} → ${backendLabel(message.decoderBackend)}`
        : 'WEBCODECS';
      $('compare-status').textContent = 'Ready';
      $('comparison-note').textContent = message.note || '';
    } else if (message.type === 'frame') {
      compareBusy = false;
      comparisonCtx.putImageData(new ImageData(
        new Uint8ClampedArray(message.rgba), WIDTH, HEIGHT), 0, 0);
      $('compare-status').textContent = `Frame ${message.frames} · ${compareBackend}`;
      $('compare-target-actual').textContent =
        `${message.targetKbps.toFixed(0)} / ${message.actualKbps.toFixed(1)}`;
      $('compare-psnr-y').textContent = formatPsnr(message.psnrY);
      $('compare-psnr-u').textContent = formatPsnr(message.psnrU);
      $('compare-psnr-v').textContent = formatPsnr(message.psnrV);
      $('compare-qp').textContent = Number.isInteger(message.qp)
        ? String(message.qp) : (message.rateMode === 'constant' ? 'Browser CBR' : 'Browser VBR');
      $('compare-qp-label').textContent = message.codec === 'mlvc'
        ? 'MLVC q-index' : 'Encoder QP / mode';
      flushPendingOwnFrame();
    } else if (message.type === 'skipped') {
      compareBusy = false;
      flushPendingOwnFrame();
    } else if (message.type === 'error') {
      compareBusy = false;
      compareReady = false;
      $('compare-status').textContent = 'Unavailable';
      $('comparison-note').textContent = message.message.split('\n')[0];
      console.error(message.message);
      flushPendingOwnFrame();
    }
  };
  thisWorker.onerror = event => {
    if (thisWorker === compareWorker) {
      compareBusy = false;
      compareReady = false;
      $('compare-status').textContent = 'Unavailable';
      $('comparison-note').textContent = event.message;
      flushPendingOwnFrame();
    }
  };
  thisWorker.postMessage({type: 'init', codec, width: WIDTH, height: HEIGHT,
    resolution, encoderBackend: $('mlvc-encode-backend').value,
    decoderBackend: $('mlvc-decode-backend').value,
    intraPeriod: Number($('period').value),
    targetKbps: compareTarget});
}

function resetComparison() {
  if (!compareWorker || !compareReady) return;
  compareReady = compareBusy = false;
  compareFrameId = 0;
  compareTarget = rateTarget();
  for (const id of ['compare-target-actual', 'compare-psnr-y',
                     'compare-psnr-u', 'compare-psnr-v', 'compare-qp']) $(id).textContent = '—';
  compareWorker.postMessage(isMlvcCodec($('compare-codec').value)
    ? {type: 'reset'} : {type: 'init', codec: $('compare-codec').value,
      width: WIDTH, height: HEIGHT, targetKbps: compareTarget});
}

function status(message, kind = '') {
  $('status').textContent = message;
  $('dot').className = `dot ${kind}`;
}

function setButtons() {
  $('camera-start').disabled = !ready || Boolean(cameraStream) ||
    ($('neural-runtime').value === 'legacy' &&
     $('encode-backend').value !== $('decode-backend').value);
  $('camera-stop').disabled = !cameraStream;
  $('encode-backend').disabled = Boolean(cameraStream);
  $('decode-backend').disabled = Boolean(cameraStream);
  $('neural-runtime').disabled = Boolean(cameraStream);
  $('resolution').disabled = Boolean(cameraStream);
}

function postOwnFrame(data) {
  worker.postMessage({type: 'frame', rgba: data, qpi: $('qpi').value,
    qpp: $('qpp').value, intraPeriod: $('period').value,
    mode: $('rate-mode').value, targetKbps: rateTarget()}, [data]);
}

function flushPendingOwnFrame() {
  if (!pendingOwnFrame) return;
  const data = pendingOwnFrame;
  pendingOwnFrame = null;
  postOwnFrame(data);
}

function sendFrame(rgba) {
  if (!ready || busy) return;
  busy = true;
  const data = rgba.buffer.slice(rgba.byteOffset, rgba.byteOffset + rgba.byteLength);
  const mlvcCompare = isMlvcCodec($('compare-codec').value);
  if (compareWorker && compareReady && (!mlvcCompare || !compareBusy)) {
    const comparisonData = data.slice(0);
    if (mlvcCompare) compareBusy = true;
    if (mlvcCompare && resolution === 1080) pendingOwnFrame = data;
    compareWorker.postMessage({type: 'frame', rgba: comparisonData,
      width: WIDTH, height: HEIGHT, frameId: compareFrameId++}, [comparisonData]);
    if (pendingOwnFrame) return;
  }
  postOwnFrame(data);
}

function captureCamera() {
  if (!cameraStream || busy || !ready || camera.readyState < 2) return;
  inputCtx.drawImage(camera, 0, 0, WIDTH, HEIGHT);
  sendFrame(inputCtx.getImageData(0, 0, WIDTH, HEIGHT).data);
}

function updateMetrics(message) {
  const now = performance.now();
  totalBytes += message.bytes;
  totalFrames++;
  const elapsedSeconds = Math.max((now - captureStartedAt) / 1000, 0.001);
  $('bitrate').textContent = (totalBytes * 8 / elapsedSeconds / 1000).toFixed(1);
  $('bitrate-30').textContent = (totalBytes * 8 * 30 / totalFrames / 1000).toFixed(1);
  $('target-actual').textContent = `${$('rate-mode').value === 'cbr'
    ? rateTarget().toFixed(0) : '—'} / ${$('bitrate-30').textContent}`;
  $('fps').textContent = lastFrameAt ? (1000 / (now - lastFrameAt)).toFixed(2) : '—';
  lastFrameAt = now;
  $('encode-time').textContent = message.encodeMs.toFixed(0);
  $('decode-time').textContent = message.decodeMs.toFixed(0);
  $('frame-size').textContent = String(message.bytes);
  $('psnr-y').textContent = formatPsnr(message.psnrY);
  $('psnr-u').textContent = formatPsnr(message.psnrU);
  $('psnr-v').textContent = formatPsnr(message.psnrV);
  $('frame-qp').textContent = `${message.frameType} · ${message.qp}`;
  const actual = Number($('bitrate-30').textContent);
  const target = rateTarget();
  $('rate-limit-note').textContent = $('rate-mode').value === 'cbr' &&
    totalFrames >= 10 && message.frameType === 'P' &&
    ((message.qp === 63 && actual < target * 0.7) ||
     (message.qp === 0 && actual > target * 1.3))
    ? 'Target cannot be reached for this scene within the available QP range.' : '';
  $('frame-tag').textContent = `${message.frameType} frame · #${message.frameNumber} · ${message.encodeBackend.toUpperCase()} → ${message.decodeBackend.toUpperCase()}`;
  if (compareWorker && compareReady && $('rate-mode').value === 'cq') {
    const next = rateTarget();
    if (Math.abs(next - compareTarget) / compareTarget > 0.15) {
      compareTarget = next;
      compareWorker.postMessage({type: 'target', targetKbps: compareTarget});
    }
  }
}

worker.onmessage = event => {
  const message = event.data;
  if (message.type === 'progress') { status(`Loading model: ${message.message}`); return; }
  if (message.type === 'ready') {
    ready = true; busy = false;
    $('encode-backend').value = message.encodeBackend;
    $('decode-backend').value = message.decodeBackend;
    status(`Ready · encode ${message.encodeBackend === 'webgpu' ? 'GPU' : 'CPU'} / decode ${message.decodeBackend === 'webgpu' ? 'GPU' : 'CPU'}`, 'ready');
    setButtons();
    return;
  }
  if (message.type === 'error') {
    busy = false;
    status(`Error: ${message.message.split('\n')[0]}`, 'error');
    console.error(message.message);
    setButtons();
    return;
  }
  if (message.type === 'frame') {
    outputCtx.putImageData(new ImageData(new Uint8ClampedArray(message.rgba), WIDTH, HEIGHT), 0, 0);
    updateMetrics(message);
    busy = false;
    status(cameraStream ? 'Encoding and decoding camera video' : 'Camera stopped', 'ready');
  }
};

$('camera-start').onclick = async () => {
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({audio: false,
      video: {width: {ideal: 1920}, height: {ideal: 1080}, frameRate: {ideal: 30}}});
    camera.srcObject = cameraStream;
    await camera.play();
    const settings = cameraStream.getVideoTracks()[0].getSettings();
    $('source-tag').textContent = `Camera ${settings.width || ''} × ${settings.height || ''}`;
    worker.postMessage({type: 'forceI'});
    resetStatistics();
    if ($('compare-codec').value) {
      if (compareWorker) resetComparison();
      else setComparison();
    }
    cameraTimer = setInterval(captureCamera, 33);
    status('Camera active', 'ready');
    setButtons();
  } catch (error) {
    cameraStream?.getTracks().forEach(track => track.stop());
    cameraStream = null;
    status(`Camera unavailable: ${error.message}`, 'error');
    setButtons();
  }
};

$('camera-stop').onclick = () => {
  clearInterval(cameraTimer);
  cameraStream?.getTracks().forEach(track => track.stop());
  cameraStream = null;
  camera.srcObject = null;
  $('source-tag').textContent = 'Camera off';
  status('Camera stopped', 'ready');
  setButtons();
};

function reloadWith(name, value) {
  const url = new URL(location.href);
  url.searchParams.set(name, value);
  location.href = url.href;
}
$('encode-backend').onchange = () => reloadWith('encode', $('encode-backend').value);
$('decode-backend').onchange = () => reloadWith('decode', $('decode-backend').value);
$('neural-runtime').onchange = () => reloadWith('runtime', $('neural-runtime').value);
$('resolution').onchange = () => reloadWith('resolution', $('resolution').value);
function updateRateControls() {
  const cbr = $('rate-mode').value === 'cbr';
  $('target-setting').classList.toggle('hidden', !cbr);
  $('qpi-setting').classList.toggle('hidden', cbr);
  $('qpp-setting').classList.toggle('hidden', cbr);
  resetStatistics();
  if (compareWorker) {
    compareTarget = rateTarget();
    compareWorker.postMessage({type: 'target', targetKbps: compareTarget});
  }
}
$('rate-mode').onchange = updateRateControls;
$('target-kbps').onchange = updateRateControls;
$('compare-codec').onchange = setComparison;
$('mlvc-encode-backend').onchange = setComparison;
$('mlvc-decode-backend').onchange = setComparison;
$('period').onchange = () => compareWorker?.postMessage({
  type: 'period', intraPeriod: Number($('period').value)});
for (const name of ['qpi', 'qpp']) {
  $(name).oninput = () => $(`${name}-value`).textContent = $(name).value;
}

setButtons();
const initialEncode = params.get('encode') || params.get('backend') || 'webgpu';
const initialDecode = params.get('decode') || params.get('backend') || 'webgpu';
const portable = params.get('runtime') !== 'legacy';
$('mlvc-encode-backend').value = params.get('mlvcEncode') === 'wasm' ? 'wasm' : 'webgpu';
$('mlvc-decode-backend').value = params.get('mlvcDecode') === 'wasm' ? 'wasm' : 'webgpu';
$('neural-runtime').value = portable ? 'portable' : 'legacy';
if (portable) {
  $('encode-backend').querySelector('option[value="wasm"]').textContent =
    'CPU · WebAssembly exact';
  $('decode-backend').querySelector('option[value="wasm"]').textContent =
    'CPU · WebAssembly exact';
}
$('encode-backend').value = initialEncode;
$('decode-backend').value = initialDecode;
if (initialEncode !== initialDecode && !portable) {
  status('Mixed GPU / CPU requires cross-device exact mode', 'error');
  setButtons();
} else {
  worker.postMessage({type: 'init', encodeBackend: initialEncode,
    decodeBackend: initialDecode, resolution, portable});
}
