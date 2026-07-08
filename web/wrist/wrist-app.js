// BalanceAI — wrist-fracture rehab coach.
//
// Perception: MediaPipe HandLandmarker (21 pts) + PoseLandmarker (forearm).
// Prediction: rep state machine over the wrist deflection angle + an MLP
//             trained on IntelliRehabDS clinician labels (quality-model.js).
// Action:     live feedback, individualized weekly plan, gamified progression.
//
// The squat coach (FormSense core) is untouched; this page is additive.

import {
  HandLandmarker,
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
import {
  HL,
  handVector,
  wristDeflectionDeg,
  signedAngleDeg,
  radialSign,
  fingerCurlRatio,
  handScale,
  matchHandToSide,
  forearmVisibility,
  clamp,
} from "./wrist-utils.js";
import { QualityModel } from "./quality-model.js";
import {
  EXERCISES,
  EXAMPLE_PROFILE,
  FUNCTIONAL_NORMS,
  gripDeficitPct,
  weeklyTargets,
  loadProfile,
  saveProfile,
} from "./plan.js";
import { Progress } from "./gamification.js";

// ---- Perception-confidence thresholds ----
const MIN_HAND_SCALE = 0.05;     // palm must span ≥5% of frame — else too far
const MIN_FOREARM_VIS = 0.55;    // below this we anchor to neutral, not forearm
const NEUTRAL_CAPTURE_MS = 2000;
const BASELINE_CAPTURE_MS = 10000;

// ---- Rep state machine ----
const REP_START_DEG = 10;   // excursion beyond neutral that starts a rep
const REP_END_DEG = 6;      // returning within this of neutral ends the rep
const REP_MIN_MS = 400;     // faster than this isn't a controlled rehab rep
const CURL_START = 0.25;    // curl-ratio drop from open baseline that starts a rep
const CURL_END = 0.12;

const $ = (id) => document.getElementById(id);
const video = $("video");
const overlay = $("overlay");
const ctx = overlay.getContext("2d");
const stage = $("stage");

let handLandmarker, poseLandmarker, drawingUtils;
let qualityModel = new QualityModel();
let modelMeta = null;

let profile = loadProfile();
let progress = new Progress();

let currentExercise = EXERCISES[0];
let appState = "idle"; // idle | neutral-capture | baseline-capture | tracking
let captureStart = 0;
let captureSamples = [];

// Per-exercise calibration: neutral hand-axis angle (deg, image space),
// thumb-side sign, open-hand curl baseline, neutral MCP position.
let calib = null;

// Live rep tracking
let repActive = false;
let repSamples = [];       // {t, angle} — signed deflection from neutral
let repPeaks = {};         // metric -> max deg this rep
let repStartT = 0;
let belowEndSince = null;

init().catch((err) => {
  console.error(err);
  $("stateText").textContent = "Init failed — check camera permissions and console.";
});

async function init() {
  $("stateText").textContent = "Loading models…";
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  [handLandmarker, poseLandmarker] = await Promise.all([
    HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numHands: 1,
    }),
    PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numPoses: 1,
    }),
  ]);
  drawingUtils = new DrawingUtils(ctx);

  try {
    modelMeta = await qualityModel.load("wrist/model-weights.json");
    const m = modelMeta.test_metrics;
    $("modelInfo").textContent =
      `Quality model: trained on ${modelMeta.n_train_reps} IRDS reps · ` +
      `held-out acc ${(m.accuracy * 100).toFixed(0)}% · AUC ${m.auc.toFixed(2)}`;
  } catch (e) {
    console.warn("Quality model unavailable:", e);
    $("modelInfo").textContent = "Quality model unavailable — geometric feedback only.";
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 },
    audio: false,
  });
  video.srcObject = stream;
  await new Promise((res) => {
    if (video.readyState >= 1) res();
    else video.onloadedmetadata = res;
  });
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;

  progress.startSession();
  renderPlan();
  renderProgress();
  renderExerciseButtons();
  $("stateText").textContent = "Ready — pick an exercise, then Set neutral.";
  requestAnimationFrame(loop);
}

// ------------------------------------------------------------ frame loop --

function loop() {
  requestAnimationFrame(loop);
  if (video.readyState < 2) return;
  const now = performance.now();

  const handRes = handLandmarker.detectForVideo(video, now);
  const poseRes = poseLandmarker.detectForVideo(video, now + 0.01);
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  const hand = handRes.landmarks?.[0];
  const pose = poseRes.landmarks?.[0];

  if (hand) {
    drawingUtils.drawConnectors(hand, HandLandmarker.HAND_CONNECTIONS, {
      color: "#2c7be5", lineWidth: 3,
    });
    drawingUtils.drawLandmarks(hand, { radius: 3, color: "#e8ecef" });
  }

  // --- Uncertainty gate 1: is there a usable hand at all? ---
  if (!hand) {
    setUncertain("No hand detected — bring your hand into frame.");
    return;
  }
  const scale = handScale(hand);
  if (scale < MIN_HAND_SCALE) {
    setUncertain("Hand too small/far — move closer to the camera.");
    return;
  }

  // --- Choose measurement anchor, exposing sensor confidence honestly ---
  let anchor = "neutral";
  let side = null;
  if (pose) {
    side = matchHandToSide(hand, pose);
    if (side && forearmVisibility(pose, side) >= MIN_FOREARM_VIS) anchor = "forearm";
  }
  $("anchorText").textContent =
    anchor === "forearm"
      ? `forearm-anchored (${side} arm tracked)`
      : "neutral-anchored (forearm not confidently visible)";

  const measurement = measure(hand, pose, side, anchor);

  $("confidenceText").textContent =
    `hand ${(clamp(scale / 0.12, 0, 1) * 100).toFixed(0)}%` +
    (pose && side ? ` · forearm ${(forearmVisibility(pose, side) * 100).toFixed(0)}%` : " · forearm —");

  if (appState === "neutral-capture") {
    handleNeutralCapture(now, hand, measurement);
    return;
  }
  if (!calib) {
    setUncertain(`Set neutral first: ${currentExercise.setup}`);
    return;
  }
  if (appState === "baseline-capture") {
    handleBaselineCapture(now, measurement);
    return;
  }
  if (appState === "tracking") {
    handleTracking(now, measurement);
  }
}

/**
 * Compute the live signed metric for the current exercise.
 * flexext / deviation → degrees from calibrated neutral;
 * tendonglide → pseudo-degrees of finger curl (ratio drop × 100).
 */
function measure(hand, pose, side, anchor) {
  if (currentExercise.id === "tendonglide") {
    const curl = fingerCurlRatio(hand);
    return { kind: "curl", curl, angle: calib ? (calib.openCurl - curl) * 100 : 0, hand };
  }

  let rawForearm = null;
  if (anchor === "forearm" && pose && side) {
    rawForearm = wristDeflectionDeg(hand, pose, side);
  }
  let deltaDeg;
  if (rawForearm != null && calib) {
    deltaDeg = rawForearm - calib.neutralForearmDeflection;
  } else {
    // Fall back to absolute hand-axis rotation vs the calibrated neutral axis.
    const hv = handVector(hand);
    deltaDeg = calib ? signedAngleDeg(calib.neutralAxis, hv) : 0;
  }
  return { kind: "angle", angle: deltaDeg, hand, rawForearm };
}

/** Direction label for the current excursion, from geometry at this instant. */
function directionLabel(m) {
  if (currentExercise.id === "tendonglide") return "curl";
  const mcp = m.hand[HL.MIDDLE_MCP];
  if (currentExercise.id === "flexext") {
    // Extension = hand lifted above its neutral height (image y grows down).
    return mcp.y < calib.neutralMcpY - 0.01 ? "extension" : "flexion";
  }
  // Deviation: same rotation sign as "toward the thumb" = radial.
  const hv = handVector(m.hand);
  const rot = signedAngleDeg(calib.neutralAxis, hv);
  return rot * calib.thumbSign >= 0 ? "radial" : "ulnar";
}

// ------------------------------------------------------ calibration flows --

function handleNeutralCapture(now, hand, m) {
  captureSamples.push({ hand, m });
  const left = Math.max(0, NEUTRAL_CAPTURE_MS - (now - captureStart));
  $("stateText").textContent = `Hold the start position… ${(left / 1000).toFixed(1)}s`;
  setRisk("uncertain");
  if (left > 0) return;

  const mid = captureSamples[Math.floor(captureSamples.length / 2)].hand;
  calib = {
    neutralAxis: handVector(mid),
    neutralMcpY: avg(captureSamples.map((s) => s.hand[HL.MIDDLE_MCP].y)),
    thumbSign: radialSign(mid),
    openCurl: avg(captureSamples.map((s) => fingerCurlRatio(s.hand)).filter((v) => v != null)),
    neutralForearmDeflection:
      avg(captureSamples.map((s) => s.m.rawForearm).filter((v) => v != null)) ?? 0,
  };
  appState = "tracking";
  $("stateText").textContent = `Tracking ${currentExercise.name} — go.`;
  $("feedbackText").textContent = currentExercise.cue;
}

function handleBaselineCapture(now, m) {
  if (m.angle != null) captureSamples.push({ t: now, m });
  const left = Math.max(0, BASELINE_CAPTURE_MS - (now - captureStart));
  $("stateText").textContent =
    `Recording HEALTHY side — full range, both directions… ${(left / 1000).toFixed(0)}s`;
  setRisk("uncertain");
  if (left > 0) return;

  if (!profile) profile = { ...EXAMPLE_PROFILE, unaffectedRom: {} };
  profile.unaffectedRom = profile.unaffectedRom || {};
  const peaks = {};
  for (const { m: mm } of captureSamples) {
    const dir = directionLabel(mm);
    peaks[dir] = Math.max(peaks[dir] ?? 0, Math.abs(mm.angle));
  }
  for (const metric of currentExercise.metrics) {
    if (peaks[metric] != null) {
      profile.unaffectedRom[metric] =
        Math.round(Math.min(peaks[metric], FUNCTIONAL_NORMS[metric] ?? 90));
    }
  }
  saveProfile(profile);
  renderPlan();
  appState = "tracking";
  $("stateText").textContent =
    `Baseline saved (${currentExercise.metrics.map((k) => `${k}: ${profile.unaffectedRom[k] ?? "—"}°`).join(", ")}). ` +
    `Now switch to your affected hand.`;
}

// ------------------------------------------------------------- tracking --

function handleTracking(now, m) {
  if (m.angle == null) {
    setUncertain("Lost the measurement — hold steady.");
    return;
  }
  const mag = Math.abs(m.angle);
  const { targets } = profile ? weeklyTargets(profile) : { targets: FUNCTIONAL_NORMS };

  // Live readout + progress vs today's target for the active direction.
  const dir = directionLabel(m);
  const target = targets[dir] ?? null;
  $("angleText").textContent =
    currentExercise.id === "tendonglide"
      ? `curl ${((m.curl ?? 0)).toFixed(2)} (open ${calib.openCurl.toFixed(2)})`
      : `${dir} ${mag.toFixed(0)}°${target ? ` / target ${target}°` : ""}`;

  const startThresh = currentExercise.id === "tendonglide" ? CURL_START * 100 : REP_START_DEG;
  const endThresh = currentExercise.id === "tendonglide" ? CURL_END * 100 : REP_END_DEG;

  if (!repActive && mag > startThresh) {
    repActive = true;
    repSamples = [];
    repPeaks = {};
    repStartT = now;
    belowEndSince = null;
  }

  if (repActive) {
    repSamples.push({ t: now, angle: m.angle });
    if (mag > (repPeaks[dir] ?? 0)) repPeaks[dir] = mag;

    // Risk color while mid-rep: green in range, yellow near target, never
    // punish exceeding it (more range is the goal) — red is reserved for
    // jerky/uncontrolled motion detected at rep end.
    setRisk("safe");
    $("feedbackText").textContent =
      target && mag >= target ? `Target reached — ${dir} ${mag.toFixed(0)}°. Ease back with control.`
        : `Good — keep the motion slow and smooth.`;

    if (mag < endThresh) {
      if (belowEndSince == null) belowEndSince = now;
      if (now - belowEndSince > 300) finishRep(now, targets);
    } else {
      belowEndSince = null;
    }
  } else {
    setRisk("safe");
    $("feedbackText").textContent = currentExercise.cue;
  }
}

function finishRep(now, targets) {
  repActive = false;
  if (now - repStartT < REP_MIN_MS) return;

  // Score on the unsigned deflection so the trajectory shape (rise to a peak,
  // return to neutral) matches the joint-angle trajectories the model was
  // trained on in IRDS.
  const scored = repSamples.map((s) => ({ t: s.t, angle: Math.abs(s.angle) }));
  const quality = qualityModel.ready ? qualityModel.score(scored) : null;
  const why = qualityModel.ready ? qualityModel.explain(scored) : "";

  let logParts = [];
  let hitTarget = false;
  let bestMetric = null, bestRom = 0;
  for (const [metric, rom] of Object.entries(repPeaks)) {
    if (rom < REP_START_DEG) continue;
    logParts.push(`${metric} ${rom.toFixed(0)}°`);
    if (targets[metric] && rom >= targets[metric]) hitTarget = true;
    if (rom > bestRom) { bestRom = rom; bestMetric = metric; }
  }
  if (!logParts.length) return;

  const xp = progress.recordRep({
    exerciseId: currentExercise.id,
    metric: bestMetric,
    romDeg: bestRom,
    quality,
    hitTarget,
  });

  const qStr = quality != null ? ` · quality ${(quality * 100).toFixed(0)}%` : "";
  addRepLog(`${logParts.join(" + ")}${qStr}${hitTarget ? " · 🎯 target" : ""} (+${xp} XP)`,
    quality != null && quality < 0.5 ? "caution" : "safe");

  if (quality != null && quality < 0.5) {
    setRisk("caution");
    $("feedbackText").textContent = why || "That rep looked rushed/unsteady — slower and smoother.";
  } else {
    $("feedbackText").textContent = hitTarget
      ? "Excellent — that's this week's target range."
      : "Clean rep. Try to reach a little further next time — stop on sharp pain.";
  }
  renderPlan();
  renderProgress();
  showBadgeToasts();
}

// ------------------------------------------------------------------- UI --

function setUncertain(msg) {
  setRisk("uncertain");
  $("feedbackText").textContent = msg;
}

function setRisk(level) {
  stage.dataset.risk = level;
}

function avg(arr) {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null;
}

function addRepLog(text, cls) {
  const li = document.createElement("li");
  li.className = cls;
  li.textContent = text;
  $("repLog").prepend(li);
}

function renderExerciseButtons() {
  const wrap = $("exerciseButtons");
  wrap.innerHTML = "";
  for (const ex of EXERCISES) {
    const btn = document.createElement("button");
    btn.textContent = ex.name;
    btn.className = ex.id === currentExercise.id ? "ex-btn active" : "ex-btn";
    btn.onclick = () => {
      currentExercise = ex;
      calib = null;
      repActive = false;
      appState = "idle";
      $("stateText").textContent = `${ex.name}: ${ex.setup} Then click Set neutral.`;
      $("feedbackText").textContent = "—";
      renderExerciseButtons();
    };
    wrap.appendChild(btn);
  }
}

function renderPlan() {
  const card = $("planCard");
  if (!profile) {
    card.innerHTML =
      `<p class="plan-empty">No patient profile yet — load the example clinician intake ` +
      `or fill in the form below.</p>`;
    return;
  }
  const { week, frac, targets } = weeklyTargets(profile);
  const deficit = gripDeficitPct(profile);
  const rows = currentExercise.metrics
    .filter((mtr) => mtr !== "curl")
    .map((mtr) => {
      const best = progress.state.bestRom[mtr] ?? 0;
      const tgt = targets[mtr];
      const pct = clamp((best / tgt) * 100, 0, 100);
      return `<div class="target-row">
        <span>${mtr}</span>
        <div class="bar"><div class="fill" style="width:${pct}%"></div></div>
        <span>${best ? best.toFixed(0) : "–"}° / ${tgt}°</span>
      </div>`;
    })
    .join("");
  card.innerHTML = `
    <div class="plan-head">
      <strong>${profile.patientName}</strong> — ${profile.affectedSide} wrist ·
      week ${week} post-cast (${Math.round(frac * 100)}% of own healthy range)
    </div>
    ${deficit != null ? `<div class="chip">grip deficit ${deficit.toFixed(1)}% (dynamometer)</div>` : ""}
    <p class="clin-note">“${profile.clinicianNote}”</p>
    ${rows || `<div class="target-row"><span>full fist</span><span>curl target ${targets.curl}</span></div>`}
  `;
}

function renderProgress() {
  $("levelText").textContent = `Lv ${progress.level} — ${progress.levelName}`;
  $("xpText").textContent = `${progress.state.xp} XP · streak ${progress.state.streak}🔥`;
  $("xpFill").style.width = `${progress.levelProgress * 100}%`;
  const badges = $("badges");
  badges.innerHTML = "";
  for (const b of progress.badgeList) {
    const el = document.createElement("span");
    el.className = b.earned ? "badge earned" : "badge";
    el.title = b.desc;
    el.textContent = b.name;
    badges.appendChild(el);
  }
}

function showBadgeToasts() {
  for (const b of progress.takeNewBadges()) {
    const toast = document.createElement("div");
    toast.className = "toast";
    toast.textContent = `🏅 Badge unlocked: ${b.name}`;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3500);
  }
}

// ------------------------------------------------------------- controls --

$("neutralBtn").onclick = () => {
  captureSamples = [];
  captureStart = performance.now();
  appState = "neutral-capture";
};

$("baselineBtn").onclick = () => {
  if (!calib) {
    $("stateText").textContent = "Set neutral first (with your HEALTHY hand in position).";
    return;
  }
  captureSamples = [];
  captureStart = performance.now();
  appState = "baseline-capture";
};

$("exampleBtn").onclick = () => {
  profile = structuredClone(EXAMPLE_PROFILE);
  saveProfile(profile);
  fillProfileForm();
  renderPlan();
};

$("profileForm").onsubmit = (e) => {
  e.preventDefault();
  profile = {
    ...(profile ?? structuredClone(EXAMPLE_PROFILE)),
    patientName: $("pfName").value || "Patient",
    affectedSide: $("pfSide").value,
    weeksPostCast: parseInt($("pfWeeks").value, 10) || 1,
    gripAffectedKg: parseFloat($("pfGripA").value) || null,
    gripUnaffectedKg: parseFloat($("pfGripU").value) || null,
    clinicianNote: $("pfNote").value || "",
  };
  saveProfile(profile);
  renderPlan();
};

function fillProfileForm() {
  if (!profile) return;
  $("pfName").value = profile.patientName ?? "";
  $("pfSide").value = profile.affectedSide ?? "right";
  $("pfWeeks").value = profile.weeksPostCast ?? 1;
  $("pfGripA").value = profile.gripAffectedKg ?? "";
  $("pfGripU").value = profile.gripUnaffectedKg ?? "";
  $("pfNote").value = profile.clinicianNote ?? "";
}

window.addEventListener("beforeunload", () => progress.endSession());
fillProfileForm();
