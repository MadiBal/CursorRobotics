// Pure geometry helpers over MediaPipe PoseLandmarker normalized landmarks.
// Each landmark is { x, y, z, visibility?, presence? } in [0,1] image space.

export const LM = {
  L_SHOULDER: 11, R_SHOULDER: 12,
  L_HIP: 23, R_HIP: 24,
  L_KNEE: 25, R_KNEE: 26,
  L_ANKLE: 27, R_ANKLE: 28,
};

/** Angle at vertex b formed by points a-b-c, in degrees. */
export function angleDeg(a, b, c) {
  const ab = { x: a.x - b.x, y: a.y - b.y };
  const cb = { x: c.x - b.x, y: c.y - b.y };
  const magAB = Math.hypot(ab.x, ab.y);
  const magCB = Math.hypot(cb.x, cb.y);
  if (magAB === 0 || magCB === 0) return 0;
  let cos = (ab.x * cb.x + ab.y * cb.y) / (magAB * magCB);
  cos = Math.min(1, Math.max(-1, cos));
  return (Math.acos(cos) * 180) / Math.PI;
}

/** Mean visibility score across a set of landmark indices (0 if missing). */
export function avgVisibility(landmarks, indices) {
  const vis = indices.map((i) => landmarks[i]?.visibility ?? 0);
  return vis.reduce((a, b) => a + b, 0) / vis.length;
}

/** Picks whichever side (left/right) has better average visibility for hip/knee/ankle. */
export function betterSide(landmarks) {
  const leftVis = avgVisibility(landmarks, [LM.L_HIP, LM.L_KNEE, LM.L_ANKLE]);
  const rightVis = avgVisibility(landmarks, [LM.R_HIP, LM.R_KNEE, LM.R_ANKLE]);
  return leftVis >= rightVis ? "left" : "right";
}

export function kneeAngleDeg(landmarks, side) {
  const hip = landmarks[side === "left" ? LM.L_HIP : LM.R_HIP];
  const knee = landmarks[side === "left" ? LM.L_KNEE : LM.R_KNEE];
  const ankle = landmarks[side === "left" ? LM.L_ANKLE : LM.R_ANKLE];
  return angleDeg(hip, knee, ankle);
}

/** Degrees the torso leans from vertical (0 = perfectly upright). */
export function torsoLeanDeg(landmarks, side) {
  const shoulder = landmarks[side === "left" ? LM.L_SHOULDER : LM.R_SHOULDER];
  const hip = landmarks[side === "left" ? LM.L_HIP : LM.R_HIP];
  const dx = shoulder.x - hip.x;
  const dy = shoulder.y - hip.y;
  return Math.abs((Math.atan2(dx, -dy) * 180) / Math.PI);
}

/**
 * Knee width / ankle width. ~1.0 means knees track over ankles.
 * A ratio noticeably below the calibrated baseline means the knees are
 * caving inward relative to the ankles ("knee valgus").
 */
export function kneeValgusRatio(landmarks) {
  const lHip = landmarks[LM.L_HIP], rHip = landmarks[LM.R_HIP];
  const lKnee = landmarks[LM.L_KNEE], rKnee = landmarks[LM.R_KNEE];
  const lAnkle = landmarks[LM.L_ANKLE], rAnkle = landmarks[LM.R_ANKLE];
  const kneeWidth = Math.hypot(rKnee.x - lKnee.x, rKnee.y - lKnee.y);
  const ankleWidth = Math.hypot(rAnkle.x - lAnkle.x, rAnkle.y - lAnkle.y) || 1e-6;
  void lHip; void rHip;
  return kneeWidth / ankleWidth;
}

export function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}
