import {cdfSyndrome, repairCdfIndex, packCdfSyndromes,
        unpackCdfSyndromes} from './cdf_syndrome.js';

const DEFAULT_BLOCKS = 64;

function featureBlocks(values, grid, blocks) {
  const blockLength = Math.ceil(values.length / blocks);
  return Array.from({length: blocks}, (_, block) => {
    const start = block * blockLength;
    const end = Math.min(values.length, start + blockLength);
    const integers = new Int32Array(Math.max(0, end - start));
    for (let i = start; i < end; i++) integers[i - start] = Math.round(values[i] * grid);
    return integers;
  });
}

export function packFeatureSyndrome(feature, grid, payload,
                                    blocks = DEFAULT_BLOCKS) {
  const checksums = featureBlocks(feature.data, grid, blocks).map(cdfSyndrome);
  return packCdfSyndromes(checksums, payload);
}

export function unpackFeatureSyndrome(stream, blocks = DEFAULT_BLOCKS) {
  return unpackCdfSyndromes(stream, blocks);
}

export function repairFeatureSyndrome(feature, raw, grid, checksums,
                                      blockCount = DEFAULT_BLOCKS) {
  if (!feature || !raw || checksums.length !== blockCount)
    throw new Error('Feature syndrome state missing');
  const blocks = featureBlocks(feature.data, grid, blockCount);
  const blockLength = Math.ceil(feature.data.length / blockCount);
  const events = [];
  for (let block = 0; block < blockCount; block++) {
    const values = blocks[block];
    const expected = checksums[block];
    const actual = cdfSyndrome(values);
    if (actual.sum === expected.sum && actual.weighted === expected.weighted &&
        actual.crc === expected.crc) continue;
    const start = block * blockLength;
    const event = repairCdfIndex(values, expected, () => {
      const ranked = [];
      for (let i = 0; i < values.length; i++) {
        const coordinate = raw[start + i] * grid;
        const distance = Math.abs(coordinate - Math.floor(coordinate) - 0.5);
        ranked.push({i, distance});
      }
      ranked.sort((a, b) => a.distance - b.distance);
      return ranked.slice(0, 256).map(row => row.i);
    }, -32768, 32767);
    for (let i = 0; i < values.length; i++)
      feature.data[start + i] = Math.fround(values[i] / grid);
    events.push({block, ...event});
  }
  return events;
}
