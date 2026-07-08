# FormSense — a real-time PT / rehab movement coach

Built for the Cursor Physical Intelligence Hackathon (Almaty, July 8).

> **This branch adds Servolt** — a wrist-fracture rehab coach at
> `/wrist.html`, built alongside (not on top of) the squat coach: personalized
> ROM targets seeded from a clinician intake, every rep scored live by a model
> trained on the clinician-labelled [IntelliRehabDS](https://zenodo.org/record/4610859)
> dataset, and light gamification (XP, streaks, badges) for home-PT adherence.
> See [PITCH.md](PITCH.md) for the demo script and
> [training/README.md](training/README.md) for the model card. The original
> FormSense squat coach below is unchanged.

FormSense watches a rehab/strength exercise (starting with the squat) through your laptop
webcam, learns *your* safe range of motion instead of a generic "ideal" form, and gives live
feedback that explicitly says when it isn't confident enough to judge — rather than guessing.

Optionally, it uses Reactor's `sana-streaming` model to render a live AI visual overlay (a
glow around the frame) that shifts color with your risk level, so the feedback is felt in the
video itself, not just a text label.

## Why this instead of a generic "AI fitness coach"

- **Personalized baseline, not a generic ideal.** A short calibration captures *your* own range
  of motion before judging anything, because "correct form" genuinely differs by body and injury
  history — judging everyone against one ideal actively fails a chunk of real users.
- **Explicit uncertainty.** MediaPipe reports a visibility/confidence score per joint. When it
  drops (occlusion, bad angle, low light), FormSense says so instead of silently scoring bad data.
- **Physical grounding, not an LLM wrapper.** All feedback comes from real joint-angle geometry
  computed every frame from on-device pose estimation — perception, a simple predictive rep/
  risk model, and (optionally) an action back into the world via the Reactor overlay.

## Quickstart (works with zero API keys)

```bash
cd server
npm install
npm start
```

Open **http://localhost:3000**, allow camera access, click **Calibrate** and do 2 slow reps of
your best-effort squat, then move normally and watch the live feedback. This is the full MVP —
no Reactor key required.

## Optional: enable the Reactor AI overlay

1. Get a Reactor API key (dashboard, or the `CURSORHACK` hackathon credit code).
2. `cp server/.env.example server/.env` and paste your key into `REACTOR_API_KEY`.
3. Restart the server (`npm start`), reload the page, check **"Enable AI Coach Overlay"**.

The server only ever holds the API key — it mints short-lived tokens for the browser, per
Reactor's documented auth flow. The overlay is additive: if it fails or the key is missing, the
core coach still works.

## How it works

- `web/app.js` — camera + MediaPipe Tasks Vision `PoseLandmarker`, calibration state machine,
  rep counter, per-frame risk scoring.
- `web/pose-utils.js` — pure joint-angle math (knee angle, torso lean, knee valgus ratio) and
  the visibility/confidence check.
- `web/reactor-feedback.js` — optional module that connects to Reactor's `sana-streaming`
  model, publishes the webcam track, and updates the edit prompt as risk level changes.
- `server/index.js` — tiny Express server: serves `web/` statically and exposes `POST
  /api/token` which exchanges `REACTOR_API_KEY` for a short-lived JWT (the key never reaches
  the browser).

## Judging rubric, mapped

- **Real problem / audience** — rehab and strength-training form feedback for a named,
  vulnerable-ish audience (post-injury / eldercare / home PT), not a generic demo.
- **Working AI, meaningfully used** — on-device pose estimation drives real joint-angle math and
  a live rep/risk model; nothing is hardcoded.
- **Physical-world grounding + uncertainty** — explicit visibility/confidence gating; the system
  refuses to score frames it can't see well.
- **Live demo** — do a good rep, a risky rep (knees caving / leaning forward), and a
  low-confidence moment (step out of frame) to show all three states.

## Next steps if there's time left

- Second exercise (e.g. shoulder raise) to show the metric system generalizes.
- Session summary screen (reps × risk tags) at the end of a set.
- Swap the Reactor prompt for a literal "ghost" of corrected form instead of a color glow.
