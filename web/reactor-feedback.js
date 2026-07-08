// Optional Reactor `sana-streaming` integration: renders a live AI visual
// overlay on the video that shifts as risk level changes. Additive only —
// if this fails or no API key is configured server-side, the core coach
// (app.js) still works standalone.
//
// Docs: https://docs.reactor.inc/model-api-reference/sana-streaming/overview

const PROMPTS = {
  safe: "Add a soft calm green glow around the edges of the frame. Keep the person and room exactly the same.",
  caution: "Add a subtle amber warning glow around the edges of the frame. Keep the person and room exactly the same.",
  risk: "Add a strong glowing red warning vignette around the edges of the frame. Keep the person and room exactly the same.",
};

export class ReactorFeedback {
  constructor({ videoEl, tokenEndpoint = "/api/token" }) {
    this.videoEl = videoEl;
    this.tokenEndpoint = tokenEndpoint;
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

    // Loaded from CDN as an ESM module so the whole app stays build-tool-free.
    const { Reactor } = await import("https://esm.sh/@reactor-team/js-sdk@latest");
    this.reactor = new Reactor({ modelName: "sana-streaming" });

    this.reactor.on("trackReceived", (name, track) => {
      if (name !== "main_video") return;
      this.videoEl.srcObject = new MediaStream([track]);
      this.videoEl.hidden = false;
      this.videoEl.play().catch(() => {});
    });

    this.reactor.on("statusChanged", async (status) => {
      if (status !== "ready") return;
      await this.reactor.publishTrack("camera", cameraTrack);
      await this.reactor.sendCommand("set_prompt", { prompt: PROMPTS.safe });
      await this.reactor.sendCommand("set_mode", { mode: "live" });
      await this.reactor.sendCommand("start", {});
      this.ready = true;
      this.lastRisk = "safe";
    });

    await this.reactor.connect(jwt);
  }

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
      this.reactor.disconnect();
      this.reactor = null;
      this.ready = false;
    }
  }
}
