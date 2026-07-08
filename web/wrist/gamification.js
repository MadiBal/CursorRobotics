// Light gamification layer: XP, levels, daily streak, badges, per-metric
// best-ROM progression. State persists in localStorage so progress survives
// reloads — deliberately simple, no accounts.

const STORAGE_KEY = "balanceai.progress.v1";

const LEVELS = [0, 100, 250, 450, 700, 1000, 1400, 1900, 2500];
export const LEVEL_NAMES = [
  "Fresh Cast-Off", "Range Rookie", "Steady Mover", "Smooth Operator",
  "Flexion Fighter", "Deviation Devotee", "Grip Guardian", "Wrist Warrior",
  "Full Range Hero",
];

const BADGES = [
  { id: "first_rep", name: "First Rep", desc: "Complete your first tracked rep" },
  { id: "ten_reps", name: "On a Roll", desc: "10 reps in one session" },
  { id: "quality_set", name: "Clinician's Choice", desc: "A rep scored 90%+ by the IRDS model" },
  { id: "target_hit", name: "Target Hit", desc: "Reach this week's ROM target" },
  { id: "streak_3", name: "3-Day Streak", desc: "Train 3 days in a row" },
  { id: "all_exercises", name: "Full Protocol", desc: "Do all 3 exercises in one session" },
];

function todayStr() {
  return new Date().toISOString().slice(0, 10);
}

function emptyState() {
  return {
    xp: 0,
    streak: 0,
    lastSessionDay: null,
    badges: [],
    bestRom: {},        // metric -> best degrees ever reached
    sessionHistory: [], // {day, reps, xp, avgQuality}
  };
}

export class Progress {
  constructor() {
    try {
      this.state = JSON.parse(localStorage.getItem(STORAGE_KEY)) || emptyState();
    } catch {
      this.state = emptyState();
    }
    this.sessionReps = 0;
    this.sessionXp = 0;
    this.sessionQualities = [];
    this.sessionExercises = new Set();
    this.newBadges = [];
  }

  save() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(this.state));
  }

  startSession() {
    const today = todayStr();
    if (this.state.lastSessionDay !== today) {
      const yesterday = new Date(Date.now() - 864e5).toISOString().slice(0, 10);
      this.state.streak = this.state.lastSessionDay === yesterday ? this.state.streak + 1 : 1;
      this.state.lastSessionDay = today;
    }
    if (this.state.streak >= 3) this.award("streak_3");
    this.save();
  }

  /**
   * Record a completed rep. XP = 10 base, scaled by the model's quality score
   * (so sloppy reps still earn something but clean reps earn more), +5 bonus
   * when the rep pushed a new personal-best ROM.
   */
  recordRep({ exerciseId, metric, romDeg, quality, hitTarget }) {
    this.sessionReps += 1;
    this.sessionExercises.add(exerciseId);
    let xp = quality != null ? Math.round(10 * (0.5 + quality)) : 10;

    if (metric && romDeg != null) {
      const prev = this.state.bestRom[metric] ?? -Infinity;
      if (romDeg > prev) {
        this.state.bestRom[metric] = Math.round(romDeg * 10) / 10;
        xp += 5;
      }
    }
    this.state.xp += xp;
    this.sessionXp += xp;
    if (quality != null) this.sessionQualities.push(quality);

    this.award("first_rep");
    if (this.sessionReps >= 10) this.award("ten_reps");
    if (quality != null && quality >= 0.9) this.award("quality_set");
    if (hitTarget) this.award("target_hit");
    if (this.sessionExercises.size >= 3) this.award("all_exercises");
    this.save();
    return xp;
  }

  endSession() {
    if (this.sessionReps === 0) return;
    const avgQ = this.sessionQualities.length
      ? this.sessionQualities.reduce((a, b) => a + b, 0) / this.sessionQualities.length
      : null;
    this.state.sessionHistory.push({
      day: todayStr(),
      reps: this.sessionReps,
      xp: this.sessionXp,
      avgQuality: avgQ != null ? Math.round(avgQ * 100) : null,
    });
    if (this.state.sessionHistory.length > 60) this.state.sessionHistory.shift();
    this.save();
  }

  award(badgeId) {
    if (this.state.badges.includes(badgeId)) return;
    this.state.badges.push(badgeId);
    const badge = BADGES.find((b) => b.id === badgeId);
    if (badge) this.newBadges.push(badge);
    this.save();
  }

  /** Pops badges earned since the last call (for toast notifications). */
  takeNewBadges() {
    const out = this.newBadges;
    this.newBadges = [];
    return out;
  }

  get level() {
    let lvl = 0;
    for (let i = 0; i < LEVELS.length; i++) if (this.state.xp >= LEVELS[i]) lvl = i;
    return lvl;
  }

  get levelName() {
    return LEVEL_NAMES[this.level];
  }

  /** Progress within the current level, 0..1. */
  get levelProgress() {
    const lvl = this.level;
    if (lvl >= LEVELS.length - 1) return 1;
    const lo = LEVELS[lvl], hi = LEVELS[lvl + 1];
    return (this.state.xp - lo) / (hi - lo);
  }

  get badgeList() {
    return BADGES.map((b) => ({ ...b, earned: this.state.badges.includes(b.id) }));
  }
}
