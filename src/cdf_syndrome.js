// Five bytes per entropy pass: 8-bit sum, 16-bit weighted sum, CRC16.
const CRC_TABLE = Uint16Array.from({length: 256}, (_, value) => {
  let crc = value << 8;
  for (let bit = 0; bit < 8; bit++)
    crc = crc & 0x8000 ? ((crc << 1) ^ 0x1021) & 0xffff
      : (crc << 1) & 0xffff;
  return crc;
});

export function cdfSyndrome(indexes) {
  let sum = 0, weighted = 0, crc = 0xffff;
  for (let i = 0; i < indexes.length; i++) {
    const value = indexes[i];
    sum = (sum + value) & 255;
    weighted = (weighted + (i + 1) * value) & 0xffff;
    crc = ((crc << 8) ^ CRC_TABLE[((crc >>> 8) ^ value) & 255]) & 0xffff;
  }
  return {sum, weighted, crc};
}

export function cdfStripe(position, stripes, hashed = false) {
  return hashed ? (position ^ (position >>> 5) ^ (position >>> 11)) & (stripes - 1)
    : position % stripes;
}

export function splitCdfIndexes(indexes, stripes = 2, hashed = false) {
  if (hashed) {
    const counts = new Uint32Array(stripes);
    for (let p = 0; p < indexes.length; p++) counts[cdfStripe(p, stripes, true)]++;
    const result = Array.from(counts, count => new Uint8Array(count));
    const positions = Array.from(counts, count => new Uint32Array(count));
    const rank = new Uint32Array(indexes.length);
    counts.fill(0);
    for (let p = 0; p < indexes.length; p++) {
      const stripe = cdfStripe(p, stripes, true), i = counts[stripe]++;
      result[stripe][i] = indexes[p];
      positions[stripe][i] = p;
      rank[p] = i;
    }
    for (let stripe = 0; stripe < stripes; stripe++) {
      result[stripe].positions = positions[stripe];
      result[stripe].rank = rank;
    }
    return result;
  }
  return Array.from({length: stripes}, (_, stripe) => {
    const result = new Uint8Array(Math.ceil((indexes.length - stripe) / stripes));
    for (let i = 0, p = stripe; p < indexes.length; i++, p += stripes)
      result[i] = indexes[p];
    return result;
  });
}

function same(a, b) {
  return a.sum === b.sum && a.weighted === b.weighted && a.crc === b.crc;
}

function crcBitInfluences(length) {
  const rows = new Array(length);
  const bits = Uint16Array.from({length: 8}, (_, bit) => CRC_TABLE[1 << bit]);
  for (let position = length - 1; position >= 0; position--) {
    rows[position] = Uint16Array.from(bits);
    for (let bit = 0; bit < 8; bit++)
      bits[bit] = ((bits[bit] << 8) ^ CRC_TABLE[bits[bit] >>> 8]) & 0xffff;
  }
  return rows;
}

function crcEffect(bitInfluences, position, before, after) {
  let changed = before ^ after;
  let effect = 0;
  for (let bit = 0; changed; bit++, changed >>>= 1)
    if (changed & 1) effect ^= bitInfluences[position][bit];
  return effect;
}

function syndromeKey(sum, weighted, crc) {
  return sum + weighted * 256 + crc * 16777216;
}

export function repairCdfIndex(indexes, expected, candidates = null,
                               minimum = 0, maximum = 127) {
  const actual = cdfSyndrome(indexes);
  if (same(actual, expected)) return {corrected: 0};
  const delta = ((expected.sum - actual.sum + 0x80) & 255) - 0x80;
  const weightedDelta = (expected.weighted - actual.weighted) & 0xffff;
  if (delta && Math.abs(delta) <= 127) {
    for (let i = 0; i < indexes.length; i++) {
      if (((delta * (i + 1)) & 0xffff) !== weightedDelta) continue;
      const candidate = indexes[i] + delta;
      if (candidate < minimum || candidate > maximum) continue;
      const previous = indexes[i];
      indexes[i] = candidate;
      if (same(cdfSyndrome(indexes), expected))
        return {corrected: 1, position: i, from: previous, to: candidate};
      indexes[i] = previous;
    }
  }
  if (candidates) {
    const positions = candidates();
    const changes = [-2, -1, 1, 2];
    for (let a = 0; a < positions.length; a++) {
      const first = positions[a];
      for (let b = a + 1; b < positions.length; b++) {
        const second = positions[b];
        for (const changeA of changes) for (const changeB of changes) {
          if (((changeA + changeB) & 255) !== (delta & 255) ||
              (((changeA * (first + 1) + changeB * (second + 1)) & 0xffff)
                !== weightedDelta)) continue;
          const nextA = indexes[first] + changeA;
          const nextB = indexes[second] + changeB;
          if (nextA < minimum || nextA > maximum ||
              nextB < minimum || nextB > maximum) continue;
          const oldA = indexes[first], oldB = indexes[second];
          indexes[first] = nextA;
          indexes[second] = nextB;
          if (same(cdfSyndrome(indexes), expected))
            return {corrected: 2, positions: [first, second],
              from: [oldA, oldB], to: [nextA, nextB]};
          indexes[first] = oldA;
          indexes[second] = oldB;
        }
      }
    }
    const lookup = new Map();
    for (const position of positions) for (const change of changes) {
      const key = `${change}:${(change * (position + 1)) & 0xffff}`;
      if (!lookup.has(key)) lookup.set(key, []);
      lookup.get(key).push(position);
    }
    for (let a = 0; a < positions.length; a++) {
      const first = positions[a];
      for (let b = a + 1; b < positions.length; b++) {
        const second = positions[b];
        for (const changeA of changes) for (const changeB of changes) {
          const changeC = delta - changeA - changeB;
          if (!changes.includes(changeC)) continue;
          const needed = (weightedDelta - changeA * (first + 1) -
            changeB * (second + 1)) & 0xffff;
          for (const third of lookup.get(`${changeC}:${needed}`) ?? []) {
            if (third === first || third === second) continue;
            const next = [indexes[first] + changeA, indexes[second] + changeB,
              indexes[third] + changeC];
            if (next.some(value => value < minimum || value > maximum)) continue;
            const old = [indexes[first], indexes[second], indexes[third]];
            indexes[first] = next[0];
            indexes[second] = next[1];
            indexes[third] = next[2];
            if (same(cdfSyndrome(indexes), expected))
              return {corrected: 3, positions: [first, second, third],
                from: old, to: next};
            indexes[first] = old[0];
            indexes[second] = old[1];
            indexes[third] = old[2];
          }
        }
      }
    }
    const influences = crcBitInfluences(indexes.length);
    const pairs = [];
    const pairsByKey = new Map();
    for (let a = 0; a < positions.length; a++) {
      const first = positions[a];
      for (let b = a + 1; b < positions.length; b++) {
        const second = positions[b];
        for (const changeA of [-1, 1]) for (const changeB of [-1, 1]) {
          const nextA = indexes[first] + changeA;
          const nextB = indexes[second] + changeB;
          if (nextA < minimum || nextA > maximum ||
              nextB < minimum || nextB > maximum) continue;
          const sum = (changeA + changeB) & 255;
          const weighted = (changeA * (first + 1) +
            changeB * (second + 1)) & 0xffff;
          const crc = crcEffect(influences, first, indexes[first], nextA) ^
            crcEffect(influences, second, indexes[second], nextB);
          const pair = {first, second, changeA, changeB, sum, weighted, crc};
          pairs.push(pair);
          const key = syndromeKey(sum, weighted, crc);
          if (!pairsByKey.has(key)) pairsByKey.set(key, []);
          pairsByKey.get(key).push(pair);
        }
      }
    }
    const targetSum = (expected.sum - actual.sum) & 255;
    const targetCrc = expected.crc ^ actual.crc;
    for (const left of pairs) {
      const key = syndromeKey((targetSum - left.sum) & 255,
        (weightedDelta - left.weighted) & 0xffff, targetCrc ^ left.crc);
      for (const right of pairsByKey.get(key) ?? []) {
        const positions4 = [left.first, left.second, right.first, right.second];
        if (new Set(positions4).size !== 4) continue;
        const changes4 = [left.changeA, left.changeB, right.changeA, right.changeB];
        const old = positions4.map(position => indexes[position]);
        for (let n = 0; n < 4; n++) indexes[positions4[n]] += changes4[n];
        if (same(cdfSyndrome(indexes), expected))
          return {corrected: 4, positions: positions4, from: old,
            to: positions4.map(position => indexes[position])};
        for (let n = 0; n < 4; n++) indexes[positions4[n]] = old[n];
      }
    }
  }
  throw new Error('CDF index syndrome mismatch exceeds four-symbol repair');
}

export function packCdfSyndromes(checksums, payload) {
  const header = new Uint8Array(checksums.length * 5);
  for (let i = 0; i < checksums.length; i++) {
    const row = checksums[i];
    const offset = i * 5;
    header[offset] = row.sum;
    header[offset + 1] = row.weighted & 255;
    header[offset + 2] = row.weighted >>> 8;
    header[offset + 3] = row.crc & 255;
    header[offset + 4] = row.crc >>> 8;
  }
  const stream = new Uint8Array(header.length + payload.length);
  stream.set(header);
  stream.set(payload, header.length);
  return {stream, headerBytes: header.length};
}

export function unpackCdfSyndromes(stream, expectedPasses) {
  if (stream.length < expectedPasses * 5 + 4)
    throw new Error('CDF syndrome header mismatch');
  const checksums = [];
  for (let i = 0; i < expectedPasses; i++) {
    const offset = i * 5;
    checksums.push({sum: stream[offset],
      weighted: stream[offset + 1] | stream[offset + 2] << 8,
      crc: stream[offset + 3] | stream[offset + 4] << 8});
  }
  return {checksums, payload: stream.subarray(expectedPasses * 5),
    headerBytes: expectedPasses * 5};
}
