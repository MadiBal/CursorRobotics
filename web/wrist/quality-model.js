// In-browser inference for BalanceAI's movement-quality model.
//
// The model is a tiny MLP trained offline (training/train_quality_model.py)
// on IntelliRehabDS (IRDS): 2500+ clinician-labelled rehab repetitions from
// 15 real patients and 14 healthy controls. It was trained NOT on raw Kinect
// coordinates but on 8 sensor-agnostic movement-quality features of a joint
// angle trajectory — so the exact same features computed here, from a
// MediaPipe wrist-angle trajectory, are valid model inputs.
//
// Output: P(a clinician would label this repetition "correctly executed").

const FPS_ASSUMED = 30; // features are expressed in real seconds via timestamps

function smooth(x, k = 5) {
  if (x.length < k) return x.slice();
  const out = new Array(x.length).fill(0);
  const half = Math.floor(k / 2);
  for (let i = 0; i < x.length; i++) {
    let sum = 0, cnt = 0;
    for (let j = i - half; j <= i + half; j++) {
      if (j >= 0 && j < x.length) { sum += x[j]; cnt++; }
    }
    out[i] = sum / cnt;
  }
  return out;
}

/**
 * Extract the same 8 features as the training pipeline from a repetition
 * trajectory. `samples` is [{t: ms, angle: deg}, ...] in chronological order.
 * Returns null when the rep is too short to featurise (matches training).
 */
export function extractRepFeatures(samples) {
  const n = samples.length;
  if (n < 15) return null;
  const duration = (samples[n - 1].t - samples[0].t) / 1000;
  if (duration <= 0.2) return null;

  const theta = smooth(samples.map((s) => s.angle));
  const dt = duration / (n - 1);

  const vel = [];
  for (let i = 1; i < n; i++) vel.push((theta[i] - theta[i - 1]) / dt);
  const velS = smooth(vel);
  const acc = [];
  for (let i = 1; i < velS.length; i++) acc.push((velS[i] - velS[i - 1]) / dt);
  const jerk = [];
  for (let i = 1; i < acc.length; i++) jerk.push((acc[i] - acc[i - 1]) / dt);

  const rom = Math.max(...theta) - Math.min(...theta);
  if (rom < 1e-6) return null;

  const peakVel = Math.max(...velS.map(Math.abs));
  const meanVel = velS.reduce((a, v) => a + Math.abs(v), 0) / velS.length;

  const signs = velS.filter((v) => Math.abs(v) > 2).map(Math.sign);
  let reversals = 0;
  for (let i = 1; i < signs.length; i++) if (signs[i] !== signs[i - 1]) reversals++;
  const reversalsPs = reversals / duration;

  const jerkRms = jerk.length
    ? Math.sqrt(jerk.reduce((a, v) => a + v * v, 0) / jerk.length) / (rom * FPS_ASSUMED)
    : 0;

  let hfEnergy = 0, totEnergy = 1e-9;
  for (let i = 0; i < vel.length; i++) {
    const hf = vel[i] - velS[i];
    hfEnergy += hf * hf;
    totEnergy += vel[i] * vel[i];
  }
  const tremorRatio = hfEnergy / totEnergy;

  const tPeakFrac = theta.indexOf(Math.max(...theta)) / n;

  return [rom, duration, peakVel, meanVel, reversalsPs, jerkRms, tremorRatio, tPeakFrac];
}

export class QualityModel {
  constructor() {
    this.weights = null;
  }

  async load(url = "wrist/model-weights.json") {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Could not load model weights (${res.status})`);
    this.weights = await res.json();
    return this.weights.meta;
  }

  get ready() {
    return this.weights !== null;
  }

  /** P(clinician-correct) in [0,1], or null if the rep can't be scored. */
  score(samples) {
    if (!this.ready) return null;
    const feats = extractRepFeatures(samples);
    if (!feats) return null;
    const { norm, W1, b1, W2, b2 } = this.weights;
    const x = feats.map((v, i) => (v - norm.mu[i]) / norm.sigma[i]);
    const nHidden = b1.length;
    let z2 = b2;
    for (let j = 0; j < nHidden; j++) {
      let z = b1[j];
      for (let i = 0; i < x.length; i++) z += x[i] * W1[i][j];
      z2 += Math.tanh(z) * W2[j];
    }
    return 1 / (1 + Math.exp(-z2));
  }

  /** Human-readable driver of a low score, from the raw features. */
  explain(samples) {
    const feats = extractRepFeatures(samples);
    if (!feats) return "";
    const [, , , , reversalsPs, jerkRms, tremorRatio] = feats;
    if (reversalsPs > 3) return "movement hesitated / reversed direction";
    if (tremorRatio > 0.35) return "tremor detected in the motion";
    if (jerkRms > 0.6) return "motion was jerky — aim for one smooth arc";
    return "";
  }
}
