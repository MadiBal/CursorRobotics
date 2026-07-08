// Individualized rehab plan for distal-radius (wrist) fracture recovery.
//
// The plan is seeded from a clinician intake: affected side, weeks since cast
// removal, dynamometer grip readings for both hands (a camera cannot measure
// force — grip strength is entered, never inferred), and the measured ROM of
// the UNAFFECTED wrist, captured live during baseline. Weekly targets are a
// progression fraction of the patient's own unaffected side, capped by
// published functional norms — never a generic ideal.

// Functional ROM norms for the wrist (degrees). A wrist reaches "functional"
// range well below anatomical maximums; these are the standard clinical
// targets after distal radius fracture (e.g. Ryu et al., J Hand Surg 1991).
export const FUNCTIONAL_NORMS = {
  flexion: 60,
  extension: 60,
  radial: 17,
  ulnar: 30,
};

// Fraction of the unaffected side's ROM targeted per week post-cast-removal.
// Conservative early ramp: soft tissue remodels slowly and forcing range in
// weeks 1-2 raises re-injury risk.
const WEEKLY_PROGRESSION = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.92, 1.0];

export const EXERCISES = [
  {
    id: "flexext",
    name: "Wrist Flexion / Extension",
    metrics: ["flexion", "extension"],
    setup: "Forearm flat on the table, hand over the edge, palm down, hand side-on to the camera.",
    cue: "Slowly bend the hand down (flexion), then lift it up (extension). One smooth arc.",
    repsPerSet: 8,
  },
  {
    id: "deviation",
    name: "Radial / Ulnar Deviation",
    metrics: ["radial", "ulnar"],
    setup: "Forearm flat on the table, palm down, back of the hand facing the camera.",
    cue: "Keep the forearm still; sweep the hand toward the thumb, then toward the pinky.",
    repsPerSet: 8,
  },
  {
    id: "tendonglide",
    name: "Finger Tendon Glides",
    metrics: ["curl"],
    setup: "Elbow on the table, palm facing the camera, all five fingers visible.",
    cue: "Open the hand fully, then curl into a full fist, then open again. Slow and complete.",
    repsPerSet: 10,
  },
];

const STORAGE_KEY = "balanceai.profile.v1";

export const EXAMPLE_PROFILE = {
  // Example clinician intake used for the live demo. Every value is editable.
  patientName: "Demo Patient",
  affectedSide: "right",
  weeksPostCast: 3,
  gripAffectedKg: 26.5,
  gripUnaffectedKg: 27.0, // ≈2% deficit — the clinical signature to close
  clinicianNote:
    "R distal radius fx, cast off 3 wks. Residual ~2% grip deficit and reduced " +
    "flexion range vs left. Progress ROM before loading; stop on sharp pain.",
  // Unaffected-side ROM (deg) — normally captured live during baseline;
  // pre-filled here so the example works even before a baseline is recorded.
  unaffectedRom: { flexion: 62, extension: 58, radial: 16, ulnar: 28 },
};

export function gripDeficitPct(profile) {
  if (!profile.gripAffectedKg || !profile.gripUnaffectedKg) return null;
  return Math.max(
    0,
    (1 - profile.gripAffectedKg / profile.gripUnaffectedKg) * 100
  );
}

/**
 * This week's per-metric ROM targets (degrees): progression fraction of the
 * patient's own unaffected side, capped at the functional norm so we never
 * chase hypermobility.
 */
export function weeklyTargets(profile) {
  const week = Math.max(1, Math.min(profile.weeksPostCast || 1, WEEKLY_PROGRESSION.length));
  const frac = WEEKLY_PROGRESSION[week - 1];
  const targets = {};
  for (const metric of Object.keys(FUNCTIONAL_NORMS)) {
    const own = profile.unaffectedRom?.[metric];
    const cap = FUNCTIONAL_NORMS[metric];
    const base = own != null ? Math.min(own, cap) : cap;
    targets[metric] = Math.round(base * frac);
  }
  // Tendon glides target full closure: curl ratio target is fixed.
  targets.curl = 1.25; // fingertips-to-wrist / palm-size ratio for a good fist
  return { week, frac, targets };
}

export function loadProfile() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

export function saveProfile(profile) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(profile));
}
