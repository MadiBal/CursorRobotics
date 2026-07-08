# Servolt training pipeline

Trains the movement-quality model that scores wrist rehab reps in the browser.

## Dataset

[IntelliRehabDS (IRDS)](https://zenodo.org/record/4610859) — Miron, Sadawi,
Ismail, Hussain & Grosan, *Data* 6(5):46, 2021 (CC-BY 4.0). Kinect recordings
of **15 real rehabilitation patients and 14 healthy controls** performing 9
physiotherapy gestures (2589 repetitions), each labelled by clinicians as
correctly or incorrectly executed. Unlike KIMORE (low-back only), IRDS covers
general range-of-motion assessment movements, and its incorrect executions
come from real patients — not healthy people faking mistakes.

## Why the model transfers from Kinect shoulders to webcam wrists

We never train on raw skeleton coordinates. Every repetition is reduced to the
time series of its primary joint angle, then to 8 sensor- and joint-agnostic
movement-quality features (range of motion, duration, peak/mean angular
velocity, hesitation reversals, jerk RMS, tremor ratio, tempo asymmetry).
Clinicians labelling IRDS were judging exactly these qualities — smoothness,
completeness, control. The browser computes the identical features from the
MediaPipe wrist-angle trajectory, so the trained network's input distribution
is preserved even though sensor and joint differ. This is a deliberate,
documented transfer assumption — not a claim that the model saw wrist data.

## Reproduce

```bash
# 1. Download SkeletonData.zip (190 MB) from https://zenodo.org/record/4610859
#    and extract to training/data/skeleton/
py -m pip install numpy
py training/train_quality_model.py --data training/data/skeleton/SkeletonData/Simplified
```

Outputs `web/wrist/model-weights.json` (weights + normalization + metrics).

## Results (subject-wise held-out test — no subject overlap with training)

| metric | value |
|---|---|
| repetitions used | 2524 (2062 train / 462 test) |
| test accuracy | 0.773 |
| balanced accuracy | 0.810 |
| sensitivity (correct reps) | 0.760 |
| specificity (incorrect reps) | 0.860 |
| AUC | 0.900 |

Model: ensemble of 7 differently-seeded 8→16→1 MLPs (tanh/sigmoid), averaged
probabilities, class-weighted log-loss, Adam, pure numpy. The ensemble beats
any single model on AUC (0.900 vs ~0.89) and on specificity — catching badly
executed reps is the clinically important direction, so that's the trade-off
we optimize.
The split separates by subject (patients and controls stratified), so the
model is evaluated on people it has never seen.

The dataset itself is not committed (190 MB); the exported weights are.
