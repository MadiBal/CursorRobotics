// Pure geometry helpers for wrist rehab metrics, over MediaPipe HandLandmarker
// (21 landmarks) and PoseLandmarker (33 landmarks), both in normalized [0,1]
// image space. Mirrors the style of ../pose-utils.js — no DOM, no state.

export const HL = {
  WRIST: 0,
  THUMB_CMC: 1, THUMB_MCP: 2, THUMB_IP: 3, THUMB_TIP: 4,
  INDEX_MCP: 5, INDEX_PIP: 6, INDEX_DIP: 7, INDEX_TIP: 8,
  MIDDLE_MCP: 9, MIDDLE_PIP: 10, MIDDLE_DIP: 11, MIDDLE_TIP: 12,
  RING_MCP: 13, RING_PIP: 14, RING_DIP: 15, RING_TIP: 16,
  PINKY_MCP: 17, PINKY_PIP: 18, PINKY_DIP: 19, PINKY_TIP: 20,
};

export const PL = {
  L_SHOULDER: 11, R_SHOULDER: 12,
  L_ELBOW: 13, R_ELBOW: 14,
  L_WRIST: 15, R_WRIST: 16,
};

function sub(a, b) { return { x: a.x - b.x, y: a.y - b.y }; }
function mag(v) { return Math.hypot(v.x, v.y); }

/**
 * Signed angle (degrees) from vector u to vector v in the image plane.
 * Positive = counter-clockwise in image coordinates (y grows downward).
 */
export function signedAngleDeg(u, v) {
  const cross = u.x * v.y - u.y * v.x;
  const dot = u.x * v.x + u.y * v.y;
  return (Math.atan2(cross, dot) * 180) / Math.PI;
}

/** Direction of the hand: wrist → middle-finger MCP (the palm's long axis). */
export function handVector(hand) {
  return sub(hand[HL.MIDDLE_MCP], hand[HL.WRIST]);
}

/** Direction of the forearm from pose landmarks: elbow → wrist. */
export function forearmVector(pose, side) {
  const elbow = pose[side === "left" ? PL.L_ELBOW : PL.R_ELBOW];
  const wrist = pose[side === "left" ? PL.L_WRIST : PL.R_WRIST];
  return sub(wrist, elbow);
}

/**
 * Wrist deflection angle in degrees: the signed angle between the forearm's
 * extended line and the hand's palm axis. 0 = wrist neutral (hand in line
 * with the forearm). The sign is protocol-dependent; callers should measure
 * relative to a calibrated neutral and use `radialSign` for anatomical labels.
 */
export function wristDeflectionDeg(hand, pose, side) {
  const fv = forearmVector(pose, side);
  const hv = handVector(hand);
  if (mag(fv) < 1e-6 || mag(hv) < 1e-6) return null;
  return signedAngleDeg(fv, hv);
}

/**
 * Which deflection sign points toward the thumb (radial deviation)?
 * Computed from the actual thumb position, so it is robust to handedness,
 * mirroring, and camera placement. Returns +1 or -1 (multiply the signed
 * deflection by this: positive result = radial, negative = ulnar).
 */
export function radialSign(hand) {
  const axis = handVector(hand);
  const toThumb = sub(hand[HL.THUMB_MCP], hand[HL.WRIST]);
  const cross = axis.x * toThumb.y - axis.y * toThumb.x;
  return cross >= 0 ? 1 : -1;
}

/**
 * Finger-curl metric: mean fingertip distance from the wrist, normalized by
 * palm size (wrist → middle MCP). ~1.9–2.1 for a fully open hand, ~0.9–1.2
 * for a fist. This is a *range-of-motion proxy* for grip — a camera cannot
 * measure force; actual grip strength must come from a dynamometer.
 */
export function fingerCurlRatio(hand) {
  const wrist = hand[HL.WRIST];
  const palm = mag(sub(hand[HL.MIDDLE_MCP], wrist));
  if (palm < 1e-6) return null;
  const tips = [HL.INDEX_TIP, HL.MIDDLE_TIP, HL.RING_TIP, HL.PINKY_TIP];
  const meanTip = tips.reduce((acc, i) => acc + mag(sub(hand[i], wrist)), 0) / tips.length;
  return meanTip / palm;
}

/** Palm size in image space — used to reject too-small / too-far hands. */
export function handScale(hand) {
  return mag(sub(hand[HL.MIDDLE_MCP], hand[HL.WRIST]));
}

/**
 * Which body side does this detected hand belong to? Matches the hand's wrist
 * landmark to the nearer pose wrist. Returns "left"/"right" (subject's side,
 * not the viewer's) or null when the pose wrists are unavailable.
 */
export function matchHandToSide(hand, pose) {
  const hw = hand[HL.WRIST];
  const lw = pose[PL.L_WRIST];
  const rw = pose[PL.R_WRIST];
  if (!lw || !rw) return null;
  const dl = mag(sub(hw, lw));
  const dr = mag(sub(hw, rw));
  return dl <= dr ? "left" : "right";
}

/** Mean pose-landmark visibility over the elbow+wrist of one side. */
export function forearmVisibility(pose, side) {
  const ids = side === "left" ? [PL.L_ELBOW, PL.L_WRIST] : [PL.R_ELBOW, PL.R_WRIST];
  return ids.reduce((a, i) => a + (pose[i]?.visibility ?? 0), 0) / ids.length;
}

export function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}
