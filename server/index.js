import "dotenv/config";
import express from "express";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();

// Serve the whole app (index.html, app.js, pose-utils.js, reactor-feedback.js, style.css).
app.use(express.static(path.join(__dirname, "..", "web")));

// Mints a short-lived Reactor JWT server-side so REACTOR_API_KEY never reaches the browser.
// See https://docs.reactor.inc/authentication
app.post("/api/token", async (_req, res) => {
  if (!process.env.REACTOR_API_KEY) {
    res.status(400).json({ error: "no_api_key", message: "REACTOR_API_KEY is not set on the server." });
    return;
  }
  try {
    const r = await fetch("https://api.reactor.inc/tokens", {
      method: "POST",
      headers: { "Reactor-API-Key": process.env.REACTOR_API_KEY },
    });
    if (!r.ok) {
      const text = await r.text();
      res.status(r.status).json({ error: "token_mint_failed", detail: text });
      return;
    }
    const data = await r.json();
    res.json(data);
  } catch (err) {
    console.error("token mint error:", err);
    res.status(500).json({ error: "token_mint_failed" });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`FormSense running: http://localhost:${PORT}`);
  console.log(
    process.env.REACTOR_API_KEY
      ? "Reactor overlay: enabled (API key found)"
      : "Reactor overlay: disabled (no REACTOR_API_KEY set) — core coach still works."
  );
});
