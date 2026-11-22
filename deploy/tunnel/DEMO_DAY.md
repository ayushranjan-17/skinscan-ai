# Showing the live model from your Mac (free tunnel)

Use this when no hosted backend is available. The Vercel site calls the backend on your Mac through a
temporary public HTTPS address. It only works while your Mac is awake and the script is running.

1. Close any other backend (`lsof -iTCP:8000 -sTCP:LISTEN`), then from the repo root run:
   ```bash
   chmod +x deploy/tunnel/start_demo.sh
   deploy/tunnel/start_demo.sh
   ```
2. Wait for the line `>>> Set VITE_API_URL to this in Vercel, then redeploy: https://...`.
3. In Vercel, **Project Settings > Environment Variables**, set `VITE_API_URL` to that address
   (no trailing slash) for Production, then **Deployments > Redeploy** on the latest one.
4. Open the site in a private window. The badge should read **AI Engine Live** and a scan should return
   the Grad-CAM and hair-removed images. First load can take a few seconds.
5. Stop with **Ctrl+C**. The address stops working immediately.

Things to know:
- **The address changes every run.** Repeat steps 2 and 3 each time (about two minutes). Do it before you present.
- **The address is public while it runs.** Anyone who has it can use the backend. Photos pass through the
  tunnel provider, so demo with public images, not patient photos.
- **Each run starts with an empty database** (`~/.skinscan-demo.db`) and a new login secret, so accounts you
  create are gone next time, and none of your real data is exposed.
- **Install `cloudflared` first** (https://github.com/cloudflare/cloudflared/releases, or `brew install cloudflared`).
  I tried an SSH tunnel through localhost.run instead; its free tier throttles traffic so heavily that scans
  never finished, so the script does not use it.
