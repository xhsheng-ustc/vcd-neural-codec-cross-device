// Byte-aligned rANS compatible with this project's native MLCodec coder.
// The original rANS source is CC0; this port is tested against the native
// Python extension before it is used by browser codec code.

const LOWER_BOUND = 1 << 23;
const SCALE_BITS = 16;
const SCALE_MASK = (1 << SCALE_BITS) - 1;

function byte(source, position) {
  if (position >= source.length) throw new Error('truncated rANS stream');
  return source[position];
}

export class RansTables {
  constructor(cdfs, lengths, offsets) {
    if (cdfs.length !== lengths.length || cdfs.length !== offsets.length) {
      throw new Error('CDF array lengths differ');
    }
    this.cdfs = cdfs.map(row => Int32Array.from(row));
    this.lengths = Int32Array.from(lengths);
    this.offsets = Int32Array.from(offsets);
  }
}

export class RansEncoder {
  constructor(groups) {
    this.groups = groups;
    this.reset();
  }

  reset() { this.tasks = []; }

  encodeY(packed, group) {
    this.tasks.push({type: 'y', packed: Int16Array.from(packed), group});
  }

  encodeZ(symbols, group, startOffset, perChannelSize) {
    this.tasks.push({type: 'z', symbols: Int8Array.from(symbols), group,
                     startOffset, perChannelSize});
  }

  flush() {
    const emitted = [];
    let state = LOWER_BOUND;
    const putBits = (value) => {
      const xMax = (1 << (SCALE_BITS - 2)) * (1 << (23 - SCALE_BITS + 8));
      while (state >= xMax) {
        emitted.push(state & 255);
        state >>>= 8;
      }
      state = ((state << 2) | value) >>> 0;
    };
    const put = (start, freq) => {
      const xMax = freq * (1 << (23 - SCALE_BITS + 8));
      while (state >= xMax) {
        emitted.push(state & 255);
        state >>>= 8;
      }
      state = (Math.floor(state / freq) * (1 << SCALE_BITS) +
               (state % freq) + start) >>> 0;
    };
    const one = (symbol, group, index) => {
      const tables = this.groups[group];
      const cdf = tables.cdfs[index];
      const maxValue = tables.lengths[index] - 2;
      let value = symbol - tables.offsets[index];
      let raw = 0;
      if (value < 0) { raw = -2 * value - 1; value = maxValue; }
      else if (value >= maxValue) { raw = 2 * (value - maxValue); value = maxValue; }
      if (value === maxValue) {
        const bins = [];
        let count = 0;
        while ((raw >>> (count * 2)) !== 0) count++;
        let remaining = count;
        while (remaining >= 3) { bins.push(3); remaining -= 3; }
        bins.push(remaining);
        for (let i = 0; i < count; i++) bins.push((raw >>> (i * 2)) & 3);
        for (let i = bins.length - 1; i >= 0; i--) putBits(bins[i]);
      }
      put(cdf[value], cdf[value + 1] - cdf[value]);
    };
    for (let t = this.tasks.length - 1; t >= 0; t--) {
      const task = this.tasks[t];
      if (task.type === 'y') {
        for (let i = task.packed.length - 1; i >= 0; i--) {
          const combined = task.packed[i];
          one(combined >> 8, task.group, combined & 255);
        }
      } else {
        for (let i = task.symbols.length - 1; i >= 0; i--) {
          const index = Math.floor(i / task.perChannelSize) + task.startOffset;
          one(task.symbols[i], task.group, index);
        }
      }
    }
    if (this.tasks.length === 0) return new Uint8Array(0);
    return Uint8Array.from([state & 255, (state >>> 8) & 255,
                            (state >>> 16) & 255, (state >>> 24) & 255,
                            ...emitted.reverse()]);
  }
}

export class RansDecoder {
  constructor(groups) { this.groups = groups; }

  setStream(stream) {
    this.stream = Uint8Array.from(stream);
    if (this.stream.length < 4) throw new Error('short rANS stream');
    this.state = (this.stream[0] | this.stream[1] << 8 |
                  this.stream[2] << 16 | this.stream[3] << 24) >>> 0;
    this.position = 4;
  }

  getBits() {
    const value = this.state & 3;
    this.state >>>= 2;
    if (this.state < LOWER_BOUND) {
      this.state = ((this.state << 8) | byte(this.stream, this.position++)) >>> 0;
    }
    return value;
  }

  one(group, index) {
    const tables = this.groups[group];
    const cdf = tables.cdfs[index];
    const maxValue = tables.lengths[index] - 2;
    const cum = this.state & SCALE_MASK;
    let symbol = 0;
    while (cdf[symbol + 1] <= cum) symbol++;
    const start = cdf[symbol];
    const freq = cdf[symbol + 1] - start;
    this.state = (freq * (this.state >>> SCALE_BITS) + cum - start) >>> 0;
    while (this.state < LOWER_BOUND) {
      this.state = ((this.state << 8) | byte(this.stream, this.position++)) >>> 0;
    }
    let value = symbol;
    if (value === maxValue) {
      let part = this.getBits();
      let count = part;
      while (part === 3) { part = this.getBits(); count += part; }
      let raw = 0;
      for (let i = 0; i < count; i++) raw |= this.getBits() << (i * 2);
      value = raw >>> 1;
      value = raw & 1 ? -value - 1 : value + maxValue;
    }
    return (value + tables.offsets[index]) << 24 >> 24;
  }

  decodeY(indexes, group) {
    const result = new Int8Array(indexes.length);
    for (let i = 0; i < result.length; i++) result[i] = this.one(group, indexes[i]);
    return result;
  }

  decodeZ(size, group, startOffset, perChannelSize) {
    const result = new Int8Array(size);
    for (let i = 0; i < size; i++) {
      result[i] = this.one(group, Math.floor(i / perChannelSize) + startOffset);
    }
    return result;
  }
}
