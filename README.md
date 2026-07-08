# Servolt — AI-Powered Physical Therapy

Built for the Cursor Physical Intelligence Hackathon (Almaty, July 8).

**Servolt is clinical-grade rehabilitation monitoring for older adults — not a fitness app.**
It turns an ordinary laptop webcam into a real-time biomechanical coach: it learns *your* safe
range of motion, coaches your form in plain language, runs validated clinical tests, watches for
falls, and tracks progress over time — all on-device, so raw video never leaves the machine.

Optionally, it uses Reactor's `sana-streaming` model to render a live AI visual overlay (a glow
around the frame) that shifts color with your risk level, so feedback is felt in the video itself.

## Why this instead of a generic "AI fitness coach"

- **Built for rehab, not repetitions.** Traditional trackers count reps. Servolt measures joint
  safety, clinical alignment, and fall risk for a frail, age-related population.
- **Personalized baseline, not a generic ideal.** A short calibration captures *your* own range of
  motion before judging anything — "correct form" genuinely differs by body and injury history.
- **Explicit uncertainty ("Refusal to Score").** MediaPipe reports a visibility score per joint.
  When it drops (occlusion, bad angle, low light), Servolt says so instead of scoring bad data.
- **On-device privacy.** All pose estimation runs locally in the browser. Only derived metrics are
  ever stored — and only on your own device (localStorage).

## What it does (mapped to the platform)

- **Active Exercise Coaching** — real-time knee-valgus, torso-lean and back-angle (hip fold)
  analysis against your calibrated baseline, with plain-language cues and rep counting (partial
  reps counted, shallow reps flagged).
- **Real-Time Joint Parameters** — live Shoulders / Back / Knees readout with safe / caution /
  unsafe states.
- **Range of Motion (ROM)** — peak knee flexion in degrees, measured each session.
- **Sit-to-Stand (30-second)** — a validated lower-extremity strength & fall-risk test, scored with
  an older-adult normative interpretation.
- **Advanced Fall Detection** — flags a fast hip drop that ends with a non-vertical torso.
- **Progress Tracking** — per-session summary (reps, form quality, ROM, STS) plus a 12-month
  fall-risk category, saved to a reviewable history.

## Quickstart (works with zero API keys)

```bash
cd server
npm install
npm start
```

Open **http://localhost:3000**, allow camera access. Stand back so your whole body is in frame,
click **Calibrate** and do 3 slow squats, then move normally and watch the live coaching. Use the
**Clinical Tests** tab for the Sit-to-Stand test, and **Progress** to save and review sessions.

> On Safari, hard-reload with a private window (Cmd-Shift-N) or Develop → Empty Caches after code
> changes, since the app is served with normal browser caching.

## Optional: enable the Reactor AI overlay

1. Get a Reactor API key (dashboard, or the `CURSORHACK` hackathon credit code).
2. `cp server/.env.example server/.env` and paste your key into `REACTOR_API_KEY`.
3. Restart the server (`npm start`), reload the page, check **"Enable AI Coach Overlay"**.

The server only ever holds the API key — it mints short-lived tokens for the browser. The overlay
is additive: if it fails or the key is missing, the core coach still works.

## How it works

- `web/app.js` — camera + MediaPipe Tasks Vision `PoseLandmarker`; calibration state machine;
  form scoring; rep counting; joint-parameter panel; ROM; Sit-to-Stand test; fall detection;
  session summary + fall-risk scoring; localStorage progress history.
- `web/pose-utils.js` — pure joint-angle math (knee angle/flexion, torso lean, back angle, knee
  valgus, shoulder tilt, torso verticality) and the visibility/confidence check.
- `web/reactor-feedback.js` — optional Reactor `sana-streaming` overlay module.
- `server/index.js` — tiny Express server: serves `web/` statically and exposes `POST /api/token`
  which exchanges `REACTOR_API_KEY` for a short-lived JWT (the key never reaches the browser).

## Notes & limitations

- **Camera angle trade-off.** Knee-valgus needs a front view; torso lean / back angle read best
  from ~45°. A single webcam can't do both perfectly at once — front-on or a slight angle works well.
- **Fast reps.** A webcam (~30 fps) plus the lite pose model smooth out very fast motion; a
  controlled tempo (~1.5–2 s/rep) detects best — which is also better rehab practice.
- **Clinical assessments are demonstrative**, tuned for a hackathon demo, not a substitute for a
  licensed physical therapist.
