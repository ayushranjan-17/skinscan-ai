# Deploying the backend to Hugging Face Spaces

The web app stays on Vercel. The Python backend (model, Grad-CAM, accounts) runs as a Docker Space.

1. Create the Space at https://huggingface.co/new-space: SDK **Docker**, template **Blank**,
   hardware **CPU basic (free)**. Name it, for example, `skinscan-api`.
2. Build the Space folder from this repo (needs `rsync`; run from the repo root):
   ```bash
   chmod +x deploy/huggingface/prepare_space.sh
   deploy/huggingface/prepare_space.sh ~/skinscan-space
   ```
3. Push it. Install Git LFS first (`brew install git-lfs && git lfs install`). When Git asks for a
   password, use a Hugging Face access token with write permission.
   ```bash
   cd ~/skinscan-space
   git init -b main && git add . && git commit -m "Deploy SkinScan API"
   git remote add space https://huggingface.co/spaces/<your-username>/skinscan-api
   git push space main
   ```
4. Optional but recommended: in the Space, **Settings > Variables and secrets**, add a secret
   `SKINSCAN_AUTH_SECRET` with a long random value. Without it, a random one is generated on each
   start, which logs everyone out when the Space restarts.
5. Wait for the build, then open `https://<your-username>-skinscan-api.hf.space/api/health`.
   It should report `model_loaded: true`.
6. In Vercel, **Project Settings > Environment Variables**, add
   `VITE_API_URL=https://<your-username>-skinscan-api.hf.space` (no trailing slash) for
   Production, then redeploy. Vite reads it at build time, so a redeploy is required.

Things to know on the free tier:
- The Space sleeps after a while with no traffic, so the first request after that is slow.
- Its disk is not persistent. Accounts and scan history reset whenever it restarts.
- Inference runs on CPU, so expect roughly one to a few seconds per scan.
