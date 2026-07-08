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
  LM,
} from "./pose-utils.js";
import { ReactorFeedback } from "./reactor-feedback.js";

const VIS_THRESHOLD = 0.5; // below this, we don't trust the read enough to score it
const CALIBRATION_MS = 6000;
const TORSO_LEAN_MARGIN_DEG = 8;
const BACK_ANGLE_MARGIN_DEG = 10; // how far the hip/back may fold past your calibrated norm before flagging
const VALGUS_MARGIN_RATIO = 0.05; // lower = more sensitive to knees caving in (was 0.08)
const MIN_SQUAT_ROM_DEG = 25;    // calibration must show at least this much knee bend to be a real squat
const MIN_REP_INTERVAL_MS = 600; // a genuine rep can't complete faster than this — debounces jitter
const REP_DESCENT_FRACTION = 0.35;  // count a rep after descending this fraction of your range (partial reps count)
const SHALLOW_DEPTH_FRACTION = 0.6; // reps below this fraction of your calibrated depth are labelled "shallow"

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

let poseLandmarker;
let drawingUtils;
let rafId;
let cameraStream;

let appState = "idle"; // idle | calibrating | tracking
let calibrationSamples = [];
let calibrationStart = 0;
let baseline = null;

let repPhase = "up"; // up | down
let repCount = 0;
let worstRiskThisRep = "safe";
let deepestThisRep = Infinity;
let lastRepAt = 0;

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
    return;
  }

  drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, { color: "#2c7be5", lineWidth: 2 });
  drawingUtils.drawLandmarks(landmarks, { radius: 3, color: "#e8ecef" });

  const side = betterSide(landmarks);
  const visibility = avgVisibility(landmarks, [
    LM.L_SHOULDER, LM.R_SHOULDER, LM.L_HIP, LM.R_HIP, LM.L_KNEE, LM.R_KNEE, LM.L_ANKLE, LM.R_ANKLE,
  ]);

  const metrics = {
    kneeAngle: kneeAngleDeg(landmarks, side),
    torsoLean: torsoLeanDeg(landmarks, side),
    backAngle: backAngleDeg(landmarks, side),
    valgusRatio: kneeValgusRatio(landmarks),
  };

  if (visibility < VIS_THRESHOLD) {
    setRisk("uncertain");
    feedbackText.textContent = "Can't confidently see your legs — adjust lighting or step back.";
    return;
  }

  if (appState === "calibrating") {
    handleCalibrationFrame(metrics);
    return;
  }

  if (appState === "tracking" && baseline) {
    handleTrackingFrame(metrics);
  }
}

function handleCalibrationFrame(metrics) {
  calibrationSamples.push(metrics);
  const remaining = Math.max(0, CALIBRATION_MS - (performance.now() - calibrationStart));
  stateText.textContent = `Calibrating… keep doing slow squats (${Math.ceil(remaining / 1000)}s left)`;
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
      "Didn't see a real squat — get your hips, knees and ankles in frame and do 2 slow, deep squats.";
    appState = "idle";
    calibrateBtn.disabled = false;
    return;
  }

  // Use percentiles, not raw min/max, so a single noisy calibration frame can't
  // stretch the "normal" envelope so wide that real bad reps never exceed it.
  baseline = {
    kneeAngleMin: Math.min(...kneeAngles), // full ROM is what we want for rep counting
    kneeAngleMax: Math.max(...kneeAngles),
    torsoLeanMax: percentile(torsoLeans, 90),
    backAngleMin: percentile(backAngles, 10), // most you fold during a good rep
    valgusMin: percentile(valgusRatios, 10),
  };

  appState = "tracking";
  stateText.textContent = "Tracking — do your reps.";
  feedbackText.textContent = "Looking good so far.";
  calibrateBtn.textContent = "Recalibrate";
  calibrateBtn.disabled = false;
}

function handleTrackingFrame(metrics) {
  const leanOver = metrics.torsoLean - (baseline.torsoLeanMax + TORSO_LEAN_MARGIN_DEG);
  const valgusUnder = (baseline.valgusMin - VALGUS_MARGIN_RATIO) - metrics.valgusRatio;
  const backUnder = (baseline.backAngleMin - BACK_ANGLE_MARGIN_DEG) - metrics.backAngle;

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
  // Lets you see whether a bad rep actually moves the numbers.
  const valgusLimit = baseline.valgusMin - VALGUS_MARGIN_RATIO;
  const backLimit = baseline.backAngleMin - BACK_ANGLE_MARGIN_DEG;
  metricsEl.textContent =
    `knee-track ${metrics.valgusRatio.toFixed(2)}/${valgusLimit.toFixed(2)} · ` +
    `back ${metrics.backAngle.toFixed(0)}°/${backLimit.toFixed(0)}°`;

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
}

function setRisk(level) {
  stage.dataset.risk = level;
}

calibrateBtn.addEventListener("click", () => {
  calibrationSamples = [];
  calibrationStart = performance.now();
  appState = "calibrating";
  calibrateBtn.disabled = true;
});

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

main().catch((err) => {
  console.error(err);
  stateText.textContent = "Camera/model init failed — check console and camera permissions.";
});
