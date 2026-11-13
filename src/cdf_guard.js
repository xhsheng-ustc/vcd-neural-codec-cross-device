// Sparse encoder-side CDF indexes for predictions close to a bin boundary.
// Each frame contains pass count, then delta-coded positions and index bytes.
function putVarint(bytes, value) {
  do {
    const chunk = value & 127;
    value = Math.floor(value / 128);
    bytes.push(chunk | (value ? 128 : 0));
  } while (value);
}

function getVarint(bytes, cursor) {
  let value = 0;
  let multiplier = 1;
  for (let n = 0; n < 5; n++) {
    if (cursor.offset >= bytes.length) throw new Error('Truncated CDF guard');
    const chunk = bytes[cursor.offset++];
    value += (chunk & 127) * multiplier;
    if (!(chunk & 128)) return value;
    multiplier *= 128;
  }
  throw new Error('Oversized CDF guard value');
}

export function packCdfGuard(passes, payload) {
  const bytes = [];
  putVarint(bytes, passes.length);
  for (const pass of passes) {
    putVarint(bytes, pass.positions.length);
    let previous = -1;
    for (let n = 0; n < pass.positions.length; n++) {
      const position = pass.positions[n];
      putVarint(bytes, position - previous - 1);
      bytes.push(pass.values[n]);
      previous = position;
    }
  }
  const stream = new Uint8Array(bytes.length + payload.length);
  stream.set(bytes);
  stream.set(payload, bytes.length);
  return {stream, headerBytes: bytes.length};
}

export function unpackCdfGuard(stream, expectedPasses) {
  const cursor = {offset: 0};
  const count = getVarint(stream, cursor);
  if (count !== expectedPasses) throw new Error('CDF guard pass count mismatch');
  const passes = [];
  for (let pass = 0; pass < count; pass++) {
    const length = getVarint(stream, cursor);
    if (length > 1000000) throw new Error('CDF guard too large');
    const positions = new Uint32Array(length);
    const values = new Uint8Array(length);
    let previous = -1;
    for (let n = 0; n < length; n++) {
      const position = previous + getVarint(stream, cursor) + 1;
      if (cursor.offset >= stream.length) throw new Error('Truncated CDF guard index');
      positions[n] = position;
      values[n] = stream[cursor.offset++];
      previous = position;
    }
    passes.push({positions, values});
  }
  return {passes, payload: stream.subarray(cursor.offset),
    headerBytes: cursor.offset};
}
