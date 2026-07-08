"""Train Servolt's movement-quality model on IntelliRehabDS (IRDS).

IRDS (https://zenodo.org/record/4610859, Miron et al., Data 2021) contains
2589 rehabilitation gesture repetitions from 15 real patients and 14 healthy
controls, recorded with a Kinect (25 joints, 30 fps), each labelled by
clinicians as correct (1) or incorrect (2).

Design
------
We do NOT train on raw Kinect coordinates: those would never transfer to a
webcam + MediaPipe at inference time. Instead every repetition is reduced to
the *time series of its primary joint angle* (elbow angle for elbow flexion,
shoulder elevation for shoulder gestures, hip abduction for side taps), and
that trajectory is summarised with 8 sensor-agnostic movement-quality
features:

    rom_deg          range of motion (max - min angle)
    duration_s       repetition duration
    peak_vel         peak angular velocity (deg/s, smoothed)
    mean_vel         mean absolute angular velocity
    reversals_ps     velocity direction reversals per second (hesitations)
    jerk_rms         RMS angular jerk, normalised by ROM (smoothness)
    tremor_ratio     high-frequency energy fraction of the velocity signal
    t_peak_frac      fraction of the rep elapsed at peak angle (tempo symmetry)

These are exactly computable in the browser from a MediaPipe wrist-angle
trajectory, so the trained network scores webcam wrist reps with the same
quality signatures clinicians labelled in IRDS.

Model: 8 -> 16 -> 1 MLP (tanh + sigmoid), pure numpy, class-weighted
log-loss, subject-wise train/test split (patients and controls both split;
no subject appears in train and test). Weights are exported to
web/wrist/model-weights.json for in-browser inference.

Usage:  py training/train_quality_model.py [--data training/data/skeleton]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

# Column order of the "Simplified" IRDS files (25 joints x 3 coords per row).
JOINTS = [
    "SpineBase", "SpineMid", "Neck", "Head",
    "ShoulderLeft", "ElbowLeft", "WristLeft", "HandLeft",
    "ShoulderRight", "ElbowRight", "WristRight", "HandRight",
    "HipLeft", "KneeLeft", "AnkleLeft", "FootLeft",
    "HipRight", "KneeRight", "AnkleRight", "FootRight",
    "SpineShoulder", "HandTipLeft", "ThumbLeft", "HandTipRight", "ThumbRight",
]
J = {name: i for i, name in enumerate(JOINTS)}

# Primary joint-angle definition (a, vertex, c) per IRDS gesture label.
GESTURE_ANGLE = {
    0: ("ShoulderLeft", "ElbowLeft", "WristLeft"),     # Elbow flexion L
    1: ("ShoulderRight", "ElbowRight", "WristRight"),  # Elbow flexion R
    2: ("HipLeft", "ShoulderLeft", "WristLeft"),       # Shoulder flexion L
    3: ("HipRight", "ShoulderRight", "WristRight"),    # Shoulder flexion R
    4: ("HipLeft", "ShoulderLeft", "ElbowLeft"),       # Shoulder abduction L
    5: ("HipRight", "ShoulderRight", "ElbowRight"),    # Shoulder abduction R
    6: ("SpineBase", "SpineShoulder", "WristRight"),   # Shoulder fwd elevation
    7: ("SpineBase", "HipLeft", "AnkleLeft"),          # Side tap L
    8: ("SpineBase", "HipRight", "AnkleRight"),        # Side tap R
}

FEATURE_NAMES = [
    "rom_deg", "duration_s", "peak_vel", "mean_vel",
    "reversals_ps", "jerk_rms", "tremor_ratio", "t_peak_frac",
]

FPS = 30.0
FILE_RE = re.compile(r"^(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\w+)\.txt$")


def angle_series(frames: np.ndarray, gesture: int) -> np.ndarray:
    """Per-frame 3D angle (degrees) at the gesture's primary joint."""
    a_name, b_name, c_name = GESTURE_ANGLE[gesture]
    a = frames[:, J[a_name] * 3 : J[a_name] * 3 + 3]
    b = frames[:, J[b_name] * 3 : J[b_name] * 3 + 3]
    c = frames[:, J[c_name] * 3 : J[c_name] * 3 + 3]
    u, v = a - b, c - b
    nu = np.linalg.norm(u, axis=1)
    nv = np.linalg.norm(v, axis=1)
    ok = (nu > 1e-9) & (nv > 1e-9)
    cosang = np.zeros(len(frames))
    cosang[ok] = np.einsum("ij,ij->i", u[ok], v[ok]) / (nu[ok] * nv[ok])
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))


def smooth(x: np.ndarray, k: int = 5) -> np.ndarray:
    if len(x) < k:
        return x
    return np.convolve(x, np.ones(k) / k, mode="same")


def extract_features(theta: np.ndarray) -> np.ndarray | None:
    """8 movement-quality features from one angle trajectory (see module doc)."""
    n = len(theta)
    if n < 15:  # < 0.5 s of data — too short to featurise
        return None
    theta_s = smooth(theta)
    vel = np.diff(theta_s) * FPS
    vel_s = smooth(vel)
    acc = np.diff(vel_s) * FPS
    jerk = np.diff(acc) * FPS

    rom = float(theta_s.max() - theta_s.min())
    if rom < 1e-6:
        return None
    duration = n / FPS
    peak_vel = float(np.abs(vel_s).max())
    mean_vel = float(np.abs(vel_s).mean())
    signs = np.sign(vel_s[np.abs(vel_s) > 2.0])  # ignore <2 deg/s jitter
    reversals = int(np.sum(np.abs(np.diff(signs)) > 0)) if len(signs) > 1 else 0
    reversals_ps = reversals / duration
    jerk_rms = float(np.sqrt(np.mean(jerk**2)) / (rom * FPS)) if len(jerk) else 0.0
    hf = vel - vel_s  # residual high-frequency component of velocity
    tremor = float(np.sum(hf**2) / (np.sum(vel**2) + 1e-9))
    t_peak = float(np.argmax(theta_s) / n)
    return np.array([rom, duration, peak_vel, mean_vel,
                     reversals_ps, jerk_rms, tremor, t_peak])


def load_dataset(data_dir: Path):
    X, y, subjects, gestures = [], [], [], []
    files = sorted(data_dir.rglob("*.txt"))
    skipped = 0
    for f in files:
        m = FILE_RE.match(f.name)
        if not m:
            continue
        subject, _date, gesture, _rep, correct, _pos = m.groups()
        gesture, correct = int(gesture), int(correct)
        if correct not in (1, 2) or gesture not in GESTURE_ANGLE:
            skipped += 1
            continue
        try:
            frames = np.loadtxt(f, delimiter=",", ndmin=2)
        except ValueError:
            skipped += 1
            continue
        if frames.ndim != 2 or frames.shape[1] != 75:
            skipped += 1
            continue
        feats = extract_features(angle_series(frames, gesture))
        if feats is None:
            skipped += 1
            continue
        X.append(feats)
        y.append(1.0 if correct == 1 else 0.0)  # 1 = clinician-approved
        subjects.append(int(subject))
        gestures.append(gesture)
    print(f"Loaded {len(X)} repetitions ({skipped} skipped) "
          f"from {len(set(subjects))} subjects")
    return np.array(X), np.array(y), np.array(subjects), np.array(gestures)


class MLP:
    """8 -> hidden -> 1 network, trained with Adam on class-weighted log-loss."""

    def __init__(self, n_in: int, n_hidden: int, seed: int = 7):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2.0 / n_in), (n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, np.sqrt(2.0 / n_hidden), (n_hidden, 1))
        self.b2 = np.zeros(1)

    def forward(self, X):
        self.h = np.tanh(X @ self.W1 + self.b1)
        z = self.h @ self.W2 + self.b2
        return 1.0 / (1.0 + np.exp(-z[:, 0]))

    def train(self, X, y, epochs=600, lr=0.01, weight_decay=1e-4):
        n = len(y)
        # Inverse-frequency class weights: IRDS has ~4x more correct reps.
        w_pos = n / (2.0 * max(y.sum(), 1.0))
        w_neg = n / (2.0 * max((1 - y).sum(), 1.0))
        w = np.where(y == 1, w_pos, w_neg)

        params = [self.W1, self.b1, self.W2, self.b2]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        b1m, b2m, eps = 0.9, 0.999, 1e-8

        for t in range(1, epochs + 1):
            p = self.forward(X)
            err = (p - y) * w / n                       # d(loss)/d(z2)
            gW2 = self.h.T @ err[:, None] + weight_decay * self.W2
            gb2 = np.array([err.sum()])
            dh = err[:, None] @ self.W2.T * (1 - self.h**2)
            gW1 = X.T @ dh + weight_decay * self.W1
            gb1 = dh.sum(axis=0)
            for i, g in enumerate([gW1, gb1, gW2, gb2]):
                m[i] = b1m * m[i] + (1 - b1m) * g
                v[i] = b2m * v[i] + (1 - b2m) * g**2
                mhat = m[i] / (1 - b1m**t)
                vhat = v[i] / (1 - b2m**t)
                params[i] -= lr * mhat / (np.sqrt(vhat) + eps)
            if t % 100 == 0:
                loss = -np.mean(w * (y * np.log(p + 1e-9)
                                     + (1 - y) * np.log(1 - p + 1e-9)))
                print(f"  epoch {t:4d}  weighted log-loss {loss:.4f}")


def subject_split(subjects: np.ndarray, test_frac=0.25, seed=13):
    """Split subject IDs so patients (2xx) and controls appear in both sets."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(subjects)
    patients = uniq[(uniq >= 200) & (uniq < 300)]
    controls = uniq[(uniq < 200) | (uniq >= 300)]
    test_ids = set()
    for group in (patients, controls):
        shuffled = rng.permutation(group)
        test_ids |= set(shuffled[: max(1, int(len(group) * test_frac))])
    return ~np.isin(subjects, list(test_ids)), np.isin(subjects, list(test_ids))


def evaluate(name, y_true, p):
    pred = (p >= 0.5).astype(float)
    acc = float((pred == y_true).mean())
    tp = float(((pred == 1) & (y_true == 1)).sum())
    tn = float(((pred == 0) & (y_true == 0)).sum())
    sens = tp / max((y_true == 1).sum(), 1)   # catches correct reps
    spec = tn / max((y_true == 0).sum(), 1)   # catches incorrect reps
    bal = (sens + spec) / 2
    # AUC via rank statistic
    order = np.argsort(p)
    ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
    n1, n0 = (y_true == 1).sum(), (y_true == 0).sum()
    auc = (ranks[y_true == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)
    print(f"{name}: acc {acc:.3f} | balanced acc {bal:.3f} "
          f"| sens {sens:.3f} | spec {spec:.3f} | AUC {auc:.3f} (n={len(p)})")
    return {"accuracy": acc, "balanced_accuracy": bal,
            "sensitivity": sens, "specificity": spec, "auc": float(auc),
            "n": int(len(p))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="training/data/skeleton")
    ap.add_argument("--out", default="web/wrist/model-weights.json")
    ap.add_argument("--hidden", type=int, default=16)
    args = ap.parse_args()

    data_dir = Path(args.data)
    simplified = next((d for d in data_dir.rglob("*")
                       if d.is_dir() and "simplified" in d.name.lower()), data_dir)
    print(f"Reading from: {simplified}")
    X, y, subjects, _ = load_dataset(simplified)
    if len(X) == 0:
        raise SystemExit("No usable repetitions found — check --data path.")

    train_mask, test_mask = subject_split(subjects)
    mu = X[train_mask].mean(axis=0)
    sigma = X[train_mask].std(axis=0) + 1e-9
    Xn = (X - mu) / sigma

    print(f"Train: {train_mask.sum()} reps | Test: {test_mask.sum()} reps "
          f"(subject-wise split, no leakage)")
    model = MLP(X.shape[1], args.hidden)
    model.train(Xn[train_mask], y[train_mask])

    train_metrics = evaluate("TRAIN", y[train_mask], model.forward(Xn[train_mask]))
    test_metrics = evaluate("TEST ", y[test_mask], model.forward(Xn[test_mask]))

    out = {
        "meta": {
            "dataset": "IntelliRehabDS (IRDS), Miron et al., Data 6(5):46, 2021",
            "source": "https://zenodo.org/record/4610859",
            "n_train_reps": int(train_mask.sum()),
            "n_test_reps": int(test_mask.sum()),
            "split": "subject-wise (patients+controls stratified)",
            "features": FEATURE_NAMES,
            "label": "P(clinician labels the repetition as correctly executed)",
            "test_metrics": test_metrics,
            "train_metrics": train_metrics,
        },
        "norm": {"mu": mu.tolist(), "sigma": sigma.tolist()},
        "W1": model.W1.tolist(), "b1": model.b1.tolist(),
        "W2": model.W2[:, 0].tolist(), "b2": float(model.b2[0]),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out))
    print(f"Exported weights -> {out_path}")


if __name__ == "__main__":
    main()
