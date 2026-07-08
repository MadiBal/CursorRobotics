// Optional Reactor `sana-streaming` integration: renders a live AI visual
// overlay on the video that shifts as risk level changes. Additive only —
// if this fails or no API key is configured server-side, the core coach
// (app.js) still works standalone.
//
// Verified against the Reactor SDK + SANA-Streaming docs:
//   base `Reactor` class from @reactor-team/js-sdk, modelName "sana-streaming",
//   input track "camera", output track "main_video", commands set_prompt /
//   set_mode({mode:"live"}) / start, server token from POST /api/token → { jwt }.
// Docs: https://docs.reactor.inc/model-api-reference/sana-streaming/overview

// Prompts follow SANA's prompt guide: name ONE change, enumerate what stays
// fixed (identity, pose, motion, background, lighting), and assert temporal
// consistency. SANA edits the source video — it does not describe a new scene.
const PRESERVE =
  "Preserve the person's identity, pose and motion, the background and the lighting, " +
  "with seamless temporal consistency across all frames and no jarring frames.";
const PROMPTS = {
  safe: `Add a soft, calm green glow emanating inward from the edges of the frame. ${PRESERVE}`,
  caution: `Add a pulsing amber warning glow around the edges of the frame. ${PRESERVE}`,
  risk: `Add a strong, intense red warning vignette glowing inward from all edges of the frame. ${PRESERVE}`,
};

// Pinned-ish CDN import keeps the app build-tool-free. Swap `@latest` for a
// specific version (e.g. `@2.12.0`) once you've confirmed the one you want.
const SDK_URL = "https://esm.sh/@reactor-team/js-sdk@latest";

export class ReactorFeedback {
  constructor({ videoEl, tokenEndpoint = "/api/token", onError = null } = {}) {
    this.videoEl = videoEl;
    this.tokenEndpoint = tokenEndpoint;
    this.onError = onError;
    this.reactor = null;
    this.ready = false;
    this.lastRisk = null;
  }

  async connect(cameraTrack) {
    const tokenRes = await fetch(this.tokenEndpoint, { method: "POST" });
    if (!tokenRes.ok) {
      throw new Error("Could not get a Reactor token — is REACTOR_API_KEY set on the server?");
    }
    const { jwt } = await tokenRes.json();
    if (!jwt) throw new Error("Token endpoint returned no jwt — check the server /api/token response.");

    // Loaded from CDN as an ESM module so the whole app stays build-tool-free.
    const { Reactor } = await import(SDK_URL);
    this.reactor = new Reactor({ modelName: "sana-streaming" });

    // Surface runtime/connection errors so failures are diagnosable during a demo.
    this.reactor.on("error", (err) => {
      const msg = `[${err?.component ?? "reactor"}] ${err?.code ?? ""} ${err?.message ?? err}`.trim();
      console.error("Reactor error:", err);
      if (this.onError) this.onError(msg);
    });

    // The edited output arrives as the model's "main_video" track.
    this.reactor.on("trackReceived", (name, track, stream) => {
      if (name !== "main_video") return;
      this.videoEl.srcObject = stream ?? new MediaStream([track]);
      this.videoEl.hidden = false;
      this.videoEl.play().catch(() => {});
    });

    // Wait for `ready` before publishing the camera and starting the stream.
    this.reactor.on("statusChanged", async (status) => {
      if (status !== "ready") return;
      try {
        await this.reactor.publishTrack("camera", cameraTrack);
        await this.reactor.sendCommand("set_prompt", { prompt: PROMPTS.safe });
        await this.reactor.sendCommand("set_mode", { mode: "live" });
        await this.reactor.sendCommand("start", {});
        this.ready = true;
        this.lastRisk = "safe";
      } catch (err) {
        console.error("Reactor start sequence failed:", err);
        if (this.onError) this.onError(err.message || "Reactor start failed.");
      }
    });

    await this.reactor.connect(jwt);
  }

  // Called by the coach each time the risk level changes; swaps the edit prompt.
  async onRiskChange(level) {
    if (!this.ready || !this.reactor) return;
    const normalized = PROMPTS[level] ? level : "safe";
    if (normalized === this.lastRisk) return;
    this.lastRisk = normalized;
    try {
      await this.reactor.sendCommand("set_prompt", { prompt: PROMPTS[normalized] });
    } catch (err) {
      console.warn("Reactor prompt update failed:", err);
    }
  }

  disconnect() {
    if (this.reactor) {
      try { this.reactor.disconnect(); } catch { /* ignore */ }
      this.reactor = null;
      this.ready = false;
      this.lastRisk = null;
    }
  }
}
