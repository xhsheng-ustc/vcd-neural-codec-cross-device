const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

// Frame-based 30 FPS rate control. The bitstream still carries the QP selected
// for each frame; targets below the codec's QP-63 floor cannot be guaranteed.
export class RateController {
  constructor(targetKbps, initialI = 63, initialP = 63) {
    this.targetKbps = targetKbps;
    this.qp = {i: initialI, p: initialP};
    this.average = {i: null, p: null};
    this.bufferBits = 0;
  }

  targetBits(isI, period) {
    const frames = Math.max(1, period);
    const normal = this.targetKbps * 1000 / 30 * frames / (frames + 7);
    const frameShare = isI ? normal * 8 : normal;
    return clamp(frameShare - this.bufferBits / 20, frameShare * 0.25,
                 frameShare * 4);
  }

  choose(isI) { return this.qp[isI ? 'i' : 'p']; }

  update(isI, period, bits) {
    const key = isI ? 'i' : 'p';
    const target = this.targetBits(isI, period);
    this.average[key] = this.average[key] === null
      ? bits : 0.25 * bits + 0.75 * this.average[key];
    const ratio = Math.max(1, this.average[key]) / Math.max(1, target);
    const step = clamp(Math.round(3 * Math.log2(ratio)), -5, 5);
    this.qp[key] = clamp(this.qp[key] - step, 0, 63);
    const nominalBits = this.targetKbps * 1000 / 30;
    this.bufferBits = clamp(this.bufferBits + bits - nominalBits,
                            -nominalBits * 30, nominalBits * 30);
  }
}
