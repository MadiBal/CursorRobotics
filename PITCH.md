# Servolt — pitch notes (live demo script + rubric mapping)

**One-liner:** FormSense taught a webcam to coach squats. Servolt extends the
idea to the highest-volume orthopedic injury there is — wrist (distal radius)
fracture — turning any laptop into a personalized, clinician-seeded,
gamified rehab station.

## The 3-minute demo script

1. **Open `/wrist.html`** (nav bar links both demos — squat coach still works
   untouched at `/`).
2. **Load the example clinician intake** — right wrist fracture, 3 weeks
   post-cast, ~2% residual grip deficit (dynamometer values), clinician note.
   Show how the weekly plan derives targets from the patient's own healthy
   side, not a generic ideal.
3. **Healthy-side baseline** — set neutral with the left hand, record 10 s of
   full-range motion. The plan card updates: this patient's personal 100%.
4. **Switch hands, do reps** — live angle readout vs this week's target,
   progress bars fill, XP/badges accumulate. Every completed rep gets a
   quality score from the IRDS-trained model.
5. **Show the three honesty states:**
   - good rep → green, "clean rep", quality ~90%
   - jerky/hesitant rep → caution, "movement hesitated / reversed direction"
   - pull the hand half out of frame / lean back → **uncertain**, the app
     says it can't measure and refuses to score.
6. **Do a squat** on the other page to show the platform generalizes.

## Rubric mapping (20 pts)

### Impact potential (5)
- Distal radius fractures are among the most common fractures (~18% of all
  fractures in adults; peak incidence in over-50s and children). Nearly every
  patient gets weeks of home ROM exercises with zero feedback between
  clinic visits.
- Audience is named and specific: post-cast wrist fracture patients doing
  home physiotherapy, plus the clinicians who prescribe it.
- Fit: home PT compliance is notoriously poor; feedback + progression +
  gamification is the standard evidence-backed lever for adherence.

### Technical execution (5)
- Working end-to-end: MediaPipe hand + pose tracking → wrist ROM geometry →
  rep segmentation → **MLP trained on IntelliRehabDS** (2524 clinician-labelled
  reps from 15 real patients; 81% held-out accuracy, AUC 0.89, subject-wise
  split) scoring every rep live in the browser.
- AI is used meaningfully twice: perception (pose/hand estimation) and
  assessment (learned movement-quality model), not an LLM wrapper.
- A real user sees it function immediately: numbers move with their wrist.

### Physical grounding (5)
- Perception: on-device landmark tracking; all metrics are real joint-angle
  geometry (wrist deflection vs forearm axis, radial/ulnar sign from actual
  thumb position — robust to handedness and mirroring).
- Prediction: rep state machine + learned quality model anticipate and
  classify movement as it happens.
- Uncertainty is first-class: hand-size gate (too far = no verdict), forearm
  visibility gate with **explicit anchor downgrade** (the UI says when it's
  measuring against a weaker reference), and refusal to score frames or reps
  it cannot see well. A camera can't measure force, so grip strength is
  clinician-entered dynamometer data — stated in the UI, never faked.

### Presentation (5)
- What: personalized wrist rehab coach. Why: highest-volume fracture, home PT
  has no feedback loop. What's next: pronation/supination via depth cues,
  clinician dashboard aggregating session history, more IRDS-style datasets
  (UI-PRMD) for per-exercise scoring heads.

## Honest limitations (say these before the jury asks)

- The quality model transfers *features* (smoothness/tempo/ROM statistics)
  from IRDS's shoulder/elbow gestures to wrist trajectories; it has never
  seen labelled wrist-fracture data. Collecting that is the obvious next step.
- 2D webcam angles are projections; the protocol constrains hand orientation
  (forearm on table) to keep the measured plane aligned with the camera.
- Grip strength is measured by a dynamometer and entered, not sensed.
