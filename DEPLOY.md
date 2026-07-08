# Deploying Servolt / FormSense

The app is a single small Express server (`server/`) that statically serves
`web/` and exposes one optional endpoint (`POST /api/token`, only used for the
Reactor overlay). Anything that can run Node can host it.

**Camera requirement:** browsers only allow webcam access (`getUserMedia`) on
`localhost` or **HTTPS** origins. Every option below provides HTTPS.

## Option A — instant public URL for the demo (no account, ~30 s)

Run the server, then open a Cloudflare quick tunnel to it:

```bash
cd server
npm install
npm start                     # serves on http://localhost:3000

# in a second terminal:
npx cloudflared tunnel --url http://localhost:3000
```

Cloudflared prints a public `https://<random>.trycloudflare.com` URL — share
it or open it on any device. No signup, no password page for visitors. The
URL lives as long as the tunnel process runs; you get a new URL each run.

Alternative with the same flow: `npx localtunnel --port 3000` (visitors must
enter the tunnel password shown at https://loca.lt/mytunnelpassword once).

## Option B — permanent free hosting (Render)

1. Push the branch to GitHub (already done: `BalanceAI` branch).
2. At https://render.com → New → Web Service → connect
   `MadiBal/CursorRobotics`, pick the `BalanceAI` branch.
3. Settings:
   - Root directory: `server`
   - Build command: `npm install`
   - Start command: `npm start`
   - Instance type: Free
4. (Optional) add environment variable `REACTOR_API_KEY` for the AI overlay.
5. Deploy — Render gives you `https://<name>.onrender.com` with HTTPS.
   The server already reads `process.env.PORT`, so no code changes needed.

Railway and Fly.io work identically (Node app, start command `npm start`
inside `server/`, honor `PORT`).

## Notes

- The trained model weights (`web/wrist/model-weights.json`) are committed,
  so deployments need no Python or dataset — training is only for reproducing
  the model (`training/README.md`).
- Progress/plan state is stored in the visitor's browser (localStorage);
  there is no database to provision.
