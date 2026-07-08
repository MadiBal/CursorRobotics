import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
import {
  betterSide,
  avgVisibility,
  kneeAngleDeg,
  torsoLeanDeg,
  backAngleDeg,
  kneeValgusRatio,
  kneeFlexionDeg,
  shoulderTiltDeg,
  hipMidY,
  torsoVerticalityDeg,
  LM,
} from "./pose-utils.js";
import { ReactorFeedback } from "./reactor-feedback.js";

const VIS_THRESHOLD = 0.45; // lower = score more (blurry) frames; higher = less noise-driven false alarms
const CALIBRATION_MS = 10000; // longer window so you can fit ~3 calibration squats
const TORSO_LEAN_MARGIN_DEG = 10; // how much extra lean past your calibrated norm before flagging
const BACK_ANGLE_MARGIN_DEG = 12;  // how far the hip/back may fold past your calibrated norm before flagging
const VALGUS_MARGIN_RATIO = 0.06; // lower = more sensitive to knees caving in
const SMOOTH_ALPHA = 0.35; // EMA factor for form metrics — damps single-frame jitter that caused false risks
const MIN_SQUAT_ROM_DEG = 25;    // calibration must show at least this much knee bend to be a real squat
const MIN_REP_INTERVAL_MS = 600; // a genuine rep can't complete faster than this — debounces jitter
const REP_DESCENT_FRACTION = 0.35;  // count a rep after descending this fraction of your range (partial reps count)
const SHALLOW_DEPTH_FRACTION = 0.6; // reps below this fraction of your calibrated depth are labelled "shallow"

// Sit-to-Stand clinical test
const STS_SECONDS = 30;
const STS_STAND_ANGLE = 155; // knee ~straight = standing
const STS_SIT_ANGLE = 110;   // knee bent = seated

// Fall detection (heuristic): a fast hip drop that ends with a non-vertical torso.
const FALL_DROP_FRAC = 0.20;   // hip midpoint drops >20% of frame height...
const FALL_LOOKBACK_MS = 350;  // ...within this window...
const FALL_TORSO_DEG = 50;     // ...and the torso is tilted past this from vertical → likely fall

const HISTORY_KEY = "servolt.sessions";

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const stage = document.getElementById("stage");
const calibrateBtn = document.getElementById("calibrateBtn");
const reactorToggle = document.getElementById("reactorToggle");
const reactorVideo = document.getElementById("reactorVideo");
const stateText = document.getElementById("stateText");
const feedbackText = document.getElementById("feedbackText");
const repCountEl = document.getElementById("repCount");
const metricsEl = document.getElementById("metricsText");
const sessionLog = document.getElementById("sessionLog");

const stageInfo = document.getElementById("stageInfo");
const stageInfoMain = document.getElementById("stageInfoMain");
const stageInfoSub = document.getElementById("stageInfoSub");
const jpShoulders = document.getElementById("jpShoulders");
const jpBack = document.getElementById("jpBack");
const jpKnees = document.getElementById("jpKnees");
const romValueEl = document.getElementById("romValue");
const fallAlert = document.getElementById("fallAlert");
const fallDismissBtn = document.getElementById("fallDismissBtn");

const stsStartBtn = document.getElementById("stsStartBtn");
const stsCountEl = document.getElementById("stsCount");
const stsTimerEl = document.getElementById("stsTimer");
const stsResultEl = document.getElementById("stsResult");
const endSessionBtn = document.getElementById("endSessionBtn");
const summaryEl = document.getElementById("summary");
const historyList = document.getElementById("historyList");

let poseLandmarker;
let drawingUtils;
let rafId;
let cameraStream;

let mode = "coach"; // coach | sts

let appState = "idle"; // idle | calibrating | tracking
let calibrationSamples = [];
let calibrationStart = 0;
let baseline = null;

let repPhase = "up"; // up | down
let repCount = 0;
let worstRiskThisRep = "safe";
let deepestThisRep = Infinity;
let lastRepAt = 0;
let smooth = null; // EMA-smoothed form metrics, so momentary jitter doesn't flag a fault

// Session-wide accumulators (for the summary / fall-risk score)
let sessionRepTotal = 0;
let sessionSafeReps = 0;
let sessionPeakFlexion = 0;
let sessionSts = null;

// Sit-to-Stand test state
let stsActive = false;
let stsCount = 0;
let stsPhase = "sit";
let stsStart = 0;

// Fall detection state
let hipHist = [];
let fallActive = false;

const reactorFeedback = new ReactorFeedback({ videoEl: reactorVideo });
let reactorConnected = false;

const RISK_RANK = { safe: 0, caution: 1, risk: 2, uncertain: 0 };

async function main() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numPoses: 1,
  });
  drawingUtils = new DrawingUtils(ctx);

  cameraStream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 },
    audio: false,
  });
  video.srcObject = cameraStream;
  await new Promise((resolve) => {
    // If metadata is already available, resolve immediately — otherwise the
    // handler is attached too late and init hangs forever.
    if (video.readyState >= 1) resolve();
    else video.onloadedmetadata = resolve;
  });
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;

  setStageInfo("Ready", "calibrate to begin coaching", "idle");
  renderLoop();
}

function renderLoop() {
  rafId = requestAnimationFrame(renderLoop);
  if (video.readyState < 2) return;

  const result = poseLandmarker.detectForVideo(video, performance.now());
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  const landmarks = result.landmarks?.[0];
  if (!landmarks) {
    setRisk("uncertain");
    feedbackText.textContent = "No person detected — step into frame.";
    setStageInfo("Waiting", "no person in frame", "uncertain");
    return;
  }

  drawSkeleton(landmarks);

  // Fall detection runs whenever the body is visible, regardless of mode.
  detectFall(landmarks);

  const side = betterSide(landmarks);
  const visibility = avgVisibility(landmarks, [
    LM.L_SHOULDER, LM.R_SHOULDER, LM.L_HIP, LM.R_HIP, LM.L_KNEE, LM.R_KNEE, LM.L_ANKLE, LM.R_ANKLE,
  ]);

  const metrics = {
    kneeAngle: kneeAngleDeg(landmarks, side),
    torsoLean: torsoLeanDeg(landmarks, side),
    backAngle: backAngleDeg(landmarks, side),
    valgusRatio: kneeValgusRatio(landmarks),
    flexion: kneeFlexionDeg(landmarks, side),
    shoulderTilt: shoulderTiltDeg(landmarks),
  };

  if (visibility < VIS_THRESHOLD) {
    // "Refusal to score" — explicit uncertainty instead of guessing on bad data.
    setRisk("uncertain");
    feedbackText.textContent = "Can't confidently see you — adjust lighting or step back.";
    setStageInfo("Refusing to score", "camera view unclear", "uncertain");
    updateJointParams(metrics, "idle", "idle");
    return;
  }

  // Range of Motion: peak knee flexion reached this session.
  if (metrics.flexion > sessionPeakFlexion) {
    sessionPeakFlexion = metrics.flexion;
    romValueEl.textContent = `${Math.round(sessionPeakFlexion)}°`;
  }

  if (mode === "sts") {
    handleStsFrame(metrics);
    updateJointParams(metrics, "safe", "safe");
    return;
  }

  if (appState === "calibrating") {
    handleCalibrationFrame(metrics);
    updateJointParams(metrics, "idle", "idle");
    return;
  }

  if (appState === "tracking" && baseline) {
    handleTrackingFrame(metrics);
  } else {
    setStageInfo(`Knee flexion ${Math.round(metrics.flexion)}°`, "calibrate to begin coaching", "idle");
    updateJointParams(metrics, "idle", "idle");
  }
}

function drawSkeleton(landmarks) {
  drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, { color: "#2dd4bf", lineWidth: 2.5 });
  drawingUtils.drawLandmarks(landmarks, { radius: 3, color: "#99f6e4" });
}

function handleCalibrationFrame(metrics) {
  calibrationSamples.push(metrics);
  const remaining = Math.max(0, CALIBRATION_MS - (performance.now() - calibrationStart));
  stateText.textContent = `Calibrating… keep doing slow squats (${Math.ceil(remaining / 1000)}s left)`;
  setStageInfo("Calibrating", `${Math.ceil(remaining / 1000)}s — learning your range`, "uncertain");
  if (remaining <= 0) finishCalibration();
}

/** p-th percentile (0–100) of a numeric array. */
function percentile(values, p) {
  const s = [...values].sort((a, b) => a - b);
  const idx = Math.round((p / 100) * (s.length - 1));
  return s[idx];
}

function finishCalibration() {
  const kneeAngles = calibrationSamples.map((s) => s.kneeAngle);
  const torsoLeans = calibrationSamples.map((s) => s.torsoLean);
  const backAngles = calibrationSamples.map((s) => s.backAngle);
  const valgusRatios = calibrationSamples.map((s) => s.valgusRatio);

  if (kneeAngles.length < 10) {
    stateText.textContent = "Calibration too short — click Calibrate and try again.";
    appState = "idle";
    calibrateBtn.disabled = false; // re-enable so the user can actually retry
    return;
  }

  // Guard against a degenerate baseline: if we never saw a real knee bend, the
  // rep thresholds would sit on top of the noisy signal and count phantom reps.
  const rom = Math.max(...kneeAngles) - Math.min(...kneeAngles);
  if (rom < MIN_SQUAT_ROM_DEG) {
    stateText.textContent =
      "Didn't see a real squat — get your hips, knees and ankles in frame and do 3 slow, deep squats.";
    appState = "idle";
    calibrateBtn.disabled = false;
    return;
  }

  // Use percentiles, not raw min/max, so a single noisy calibration frame can't
  // stretch the "normal" envelope so wide that real bad reps never exceed it.
  baseline = {
    kneeAngleMin: Math.min(...kneeAngles), // full ROM is what we want for rep counting
    kneeAngleMax: Math.max(...kneeAngles),
    torsoLeanMax: percentile(torsoLeans, 97),
    backAngleMin: percentile(backAngles, 3), // most you fold during a good rep
    valgusMin: percentile(valgusRatios, 3),
  };

  appState = "tracking";
  stateText.textContent = "Tracking — do your reps.";
  feedbackText.textContent = "Looking good so far.";
  calibrateBtn.textContent = "Recalibrate";
  calibrateBtn.disabled = false;
}

function handleTrackingFrame(metrics) {
  // Score against EMA-smoothed metrics so a single noisy frame can't trip a fault.
  if (!smooth) {
    smooth = { valgusRatio: metrics.valgusRatio, backAngle: metrics.backAngle, torsoLean: metrics.torsoLean };
  } else {
    smooth.valgusRatio += SMOOTH_ALPHA * (metrics.valgusRatio - smooth.valgusRatio);
    smooth.backAngle += SMOOTH_ALPHA * (metrics.backAngle - smooth.backAngle);
    smooth.torsoLean += SMOOTH_ALPHA * (metrics.torsoLean - smooth.torsoLean);
  }

  const leanOver = smooth.torsoLean - (baseline.torsoLeanMax + TORSO_LEAN_MARGIN_DEG);
  const valgusUnder = (baseline.valgusMin - VALGUS_MARGIN_RATIO) - smooth.valgusRatio;
  const backUnder = (baseline.backAngleMin - BACK_ANGLE_MARGIN_DEG) - smooth.backAngle;

  let level = "safe";
  let message = "Good form.";

  if (valgusUnder > 0.05) {
    level = "risk";
    message = "Knees are caving inward — push them out over your toes.";
  } else if (backUnder > 8) {
    level = "risk";
    message = "Back is folding / rounding forward — chest up, keep a neutral spine.";
  } else if (leanOver > 6) {
    level = "risk";
    message = "Leaning too far forward — keep your chest up.";
  } else if (valgusUnder > 0 || backUnder > 0 || leanOver > 0) {
    level = "caution";
    message =
      valgusUnder > 0 ? "Watch your knee position."
      : backUnder > 0 ? "Watch your back — stay tall."
      : "Watch your forward lean.";
  }

  // Live readout: current value vs. the limit it must cross to be flagged.
  const valgusLimit = baseline.valgusMin - VALGUS_MARGIN_RATIO;
  const backLimit = baseline.backAngleMin - BACK_ANGLE_MARGIN_DEG;
  metricsEl.textContent =
    `knee-track ${smooth.valgusRatio.toFixed(2)}/${valgusLimit.toFixed(2)} · ` +
    `back ${smooth.backAngle.toFixed(0)}°/${backLimit.toFixed(0)}°`;

  const backState = backUnder > 8 ? "risk" : backUnder > 0 ? "caution" : "safe";
  const kneeState = valgusUnder > 0.05 ? "risk" : valgusUnder > 0 ? "caution" : "safe";
  updateJointParams(metrics, kneeState, backState);

  const stateLabel =
    level === "safe" ? "✓ Safe range" : level === "caution" ? "⚠ Watch form" : "✕ Unsafe";
  setStageInfo(`Knee flexion ${Math.round(metrics.flexion)}°`, stateLabel, level);

  setRisk(level);
  feedbackText.textContent = message;
  if (RISK_RANK[level] > RISK_RANK[worstRiskThisRep]) worstRiskThisRep = level;

  if (reactorConnected) reactorFeedback.onRiskChange(level);

  trackRep(metrics.kneeAngle);
}

function trackRep(kneeAngle) {
  // Count a rep once you've dipped a real fraction of your calibrated range —
  // not only at full depth — then returned near standing. Partial reps count
  // and get labelled "shallow"; the hysteresis gap prevents double-counting.
  const range = Math.max(baseline.kneeAngleMax - baseline.kneeAngleMin, 1);
  const downThresh = baseline.kneeAngleMax - range * REP_DESCENT_FRACTION;
  const upThresh = baseline.kneeAngleMax - range * 0.1;

  if (repPhase === "down") deepestThisRep = Math.min(deepestThisRep, kneeAngle);

  if (repPhase === "up" && kneeAngle < downThresh) {
    repPhase = "down";
    deepestThisRep = kneeAngle;
  } else if (repPhase === "down" && kneeAngle > upThresh) {
    repPhase = "up";
    const now = performance.now();
    if (now - lastRepAt < MIN_REP_INTERVAL_MS) return; // too fast to be a real rep — ignore
    lastRepAt = now;
    repCount += 1;
    repCountEl.textContent = String(repCount);
    // Depth reached as a fraction of your calibrated full-depth squat.
    const depthFrac = (baseline.kneeAngleMax - deepestThisRep) / range;
    logRep(repCount, worstRiskThisRep, depthFrac < SHALLOW_DEPTH_FRACTION);
    worstRiskThisRep = "safe";
    deepestThisRep = Infinity;
  }
}

function logRep(n, risk, shallow) {
  const li = document.createElement("li");
  li.className = risk;
  li.textContent = `Rep ${n} — ${risk}${shallow ? " · shallow" : ""}`;
  sessionLog.prepend(li);
  sessionRepTotal += 1;
  if (risk === "safe") sessionSafeReps += 1;
}

/* ---------- Real-Time Joint Parameters panel ---------- */
function updateJointParams(metrics, kneeState, backState) {
  // Shoulders: how level the shoulder line is.
  const tilt = metrics.shoulderTilt;
  const sState = tilt < 7 ? "safe" : tilt < 14 ? "caution" : "risk";
  setChip(jpShoulders, tilt < 7 ? "Level" : tilt < 14 ? "Slight tilt" : "Off-level",
    `${tilt.toFixed(0)}° tilt`, sState);

  // Back: fold / rounding relative to your calibrated neutral.
  const bLabel = backState === "safe" ? "Within limit" : backState === "caution" ? "Folding" : "Rounding";
  setChip(jpBack, backState === "idle" ? "—" : bLabel, `${metrics.backAngle.toFixed(0)}°`, backState);

  // Knees: flexion depth + safety.
  const kLabel = kneeState === "safe" ? "Safe depth" : kneeState === "caution" ? "Watch" : kneeState === "risk" ? "Unsafe" : "—";
  setChip(jpKnees, kLabel, `${metrics.flexion.toFixed(0)}° flex`, kneeState);
}

function setChip(el, valText, subText, state) {
  el.querySelector(".jp-val").textContent = valText;
  el.querySelector(".jp-sub").textContent = subText;
  el.dataset.state = state === "idle" ? "idle" : state;
}

function setStageInfo(main, sub, state) {
  stageInfoMain.textContent = main;
  stageInfoSub.textContent = sub;
  stageInfo.dataset.state = state;
}

function setRisk(level) {
  stage.dataset.risk = level;
}

/* ---------- Sit-to-Stand (30s) clinical test ---------- */
function startSts() {
  mode = "sts";
  stsActive = true;
  stsCount = 0;
  stsPhase = "sit";
  stsStart = performance.now();
  stsCountEl.textContent = "0";
  stsResultEl.textContent = "";
  stsStartBtn.disabled = true;
  tickSts();
}

function tickSts() {
  if (!stsActive) return;
  const remaining = Math.max(0, STS_SECONDS - (performance.now() - stsStart) / 1000);
  stsTimerEl.textContent = `${Math.ceil(remaining)}s`;
  if (remaining <= 0) {
    finishSts();
    return;
  }
  requestAnimationFrame(tickSts);
}

function handleStsFrame(metrics) {
  setStageInfo("Sit-to-Stand test", `${stsCount} stands`, "safe");
  if (!stsActive) return;
  const k = metrics.kneeAngle;
  if (stsPhase === "sit" && k > STS_STAND_ANGLE) {
    stsPhase = "stand";
    stsCount += 1;
    stsCountEl.textContent = String(stsCount);
  } else if (stsPhase === "stand" && k < STS_SIT_ANGLE) {
    stsPhase = "sit";
  }
}

function finishSts() {
  stsActive = false;
  mode = "coach";
  stsStartBtn.disabled = false;
  stsTimerEl.textContent = "0s";
  sessionSts = stsCount;
  stsResultEl.textContent = `${stsCount} stands in 30 s — ${stsInterpretation(stsCount)}`;
}

function stsInterpretation(n) {
  // Rough guide based on the 30-second chair-stand test (older-adult norms).
  if (n < 8) return "below average — elevated fall risk, review with a therapist.";
  if (n < 12) return "within the typical older-adult range.";
  return "strong lower-body function.";
}

/* ---------- Fall detection ---------- */
function detectFall(landmarks) {
  const now = performance.now();
  const y = hipMidY(landmarks);
  hipHist.push({ t: now, y });
  while (hipHist.length && now - hipHist[0].t > 900) hipHist.shift();
  if (fallActive) return;

  const past = hipHist.find((h) => now - h.t >= FALL_LOOKBACK_MS);
  if (!past) return;
  const drop = y - past.y; // positive = moved downward in the frame
  if (drop > FALL_DROP_FRAC && torsoVerticalityDeg(landmarks) > FALL_TORSO_DEG) {
    triggerFall();
  }
}

function triggerFall() {
  fallActive = true;
  fallAlert.hidden = false;
}

fallDismissBtn.addEventListener("click", () => {
  fallAlert.hidden = true;
  fallActive = false;
  hipHist = [];
});

/* ---------- Session summary + progress history ---------- */
function endSession() {
  if (sessionRepTotal === 0 && sessionSts === null && sessionPeakFlexion === 0) {
    summaryEl.innerHTML = `<p class="muted">No activity yet — do a coaching set or a Sit-to-Stand test first.</p>`;
    return;
  }
  const quality = sessionRepTotal ? Math.round((sessionSafeReps / sessionRepTotal) * 100) : null;
  const rec = {
    date: new Date().toISOString(),
    reps: sessionRepTotal,
    formQuality: quality,
    peakFlexion: Math.round(sessionPeakFlexion),
    sts: sessionSts,
    fallRisk: fallRiskCategory(quality, sessionSts, sessionPeakFlexion),
  };
  saveSession(rec);
  renderSummary(rec);
  renderHistory();

  // Reset for the next session.
  sessionRepTotal = 0;
  sessionSafeReps = 0;
  sessionPeakFlexion = 0;
  sessionSts = null;
  repCount = 0;
  repCountEl.textContent = "0";
  sessionLog.innerHTML = "";
  romValueEl.textContent = "0°";
}

function fallRiskCategory(quality, sts, flexion) {
  let score = 0;
  if (sts != null) { if (sts < 8) score += 2; else if (sts < 12) score += 1; }
  if (quality != null) { if (quality < 60) score += 2; else if (quality < 85) score += 1; }
  if (flexion && flexion < 70) score += 1; // limited functional range of motion
  return score >= 3 ? "High" : score >= 1 ? "Moderate" : "Low";
}

function renderSummary(rec) {
  const cls = rec.fallRisk.toLowerCase();
  summaryEl.innerHTML = `
    <div class="summary-grid">
      <div class="cell"><div class="k">Reps coached</div><div class="v">${rec.reps}</div></div>
      <div class="cell"><div class="k">Form quality</div><div class="v">${rec.formQuality == null ? "—" : rec.formQuality + "%"}</div></div>
      <div class="cell"><div class="k">Peak knee flexion</div><div class="v">${rec.peakFlexion}°</div></div>
      <div class="cell"><div class="k">Sit-to-Stand</div><div class="v">${rec.sts == null ? "—" : rec.sts}</div></div>
    </div>
    <p style="margin:0.8rem 0 0">12-month fall risk: <span class="badge ${cls}">${rec.fallRisk}</span></p>`;
}

function loadSessions() {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY)) || []; }
  catch { return []; }
}

function saveSession(rec) {
  const arr = loadSessions();
  arr.push(rec);
  localStorage.setItem(HISTORY_KEY, JSON.stringify(arr.slice(-20)));
}

function renderHistory() {
  const arr = loadSessions();
  if (!arr.length) {
    historyList.innerHTML = `<li class="muted">No saved sessions yet.</li>`;
    return;
  }
  historyList.innerHTML = arr.slice().reverse().map((r) => {
    const d = new Date(r.date);
    const date = d.toLocaleDateString(undefined, { month: "short", day: "numeric" }) +
      " " + d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
    const cls = (r.fallRisk || "low").toLowerCase();
    const parts = [];
    parts.push(`${r.reps} reps`);
    if (r.formQuality != null) parts.push(`${r.formQuality}% form`);
    if (r.sts != null) parts.push(`STS ${r.sts}`);
    parts.push(`ROM ${r.peakFlexion}°`);
    return `<li>
      <span class="h-date">${date}</span>
      <span class="h-stats">${parts.join(" · ")}</span>
      <span class="badge ${cls}">${r.fallRisk}</span>
    </li>`;
  }).join("");
}

/* ---------- UI wiring ---------- */
calibrateBtn.addEventListener("click", () => {
  calibrationSamples = [];
  calibrationStart = performance.now();
  appState = "calibrating";
  calibrateBtn.disabled = true;
  smooth = null; // start the EMA fresh for the new baseline
});

stsStartBtn.addEventListener("click", startSts);
endSessionBtn.addEventListener("click", endSession);

document.querySelectorAll(".tab").forEach((t) => {
  t.addEventListener("click", () => selectTab(t.dataset.tab));
});
function selectTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".tab-panel").forEach((p) => { p.hidden = p.dataset.tab !== name; });
}

reactorToggle.addEventListener("change", async () => {
  if (reactorToggle.checked) {
    try {
      stateText.textContent = "Connecting to Reactor…";
      const track = cameraStream.getVideoTracks()[0];
      await reactorFeedback.connect(track);
      reactorConnected = true;
      stateText.textContent = "Reactor overlay connected.";
    } catch (err) {
      console.error(err);
      reactorToggle.checked = false;
      alert(err.message || "Could not connect to Reactor overlay.");
    }
  } else {
    reactorFeedback.disconnect();
    reactorConnected = false;
    reactorVideo.hidden = true;
  }
});

// Populate history before camera init so the Progress tab is never blank.
renderHistory();

main().catch((err) => {
  console.error(err);
  stateText.textContent = "Camera/model init failed — check console and camera permissions.";
  setStageInfo("Init failed", "check camera permissions", "risk");
});
