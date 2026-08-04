# Deploying Complexity Lab on Render

Render runs Streamlit as a normal long-lived web service, so the app deploys
**with no code changes**. Two ways to do it — pick one.

> Note on Vercel: Streamlit needs a persistent server holding a live WebSocket
> connection. Vercel's serverless functions are short-lived and stateless, so
> the interactive app can't run there. Render (or Railway / Hugging Face Spaces /
> Streamlit Community Cloud) is the right fit.

## Option A — Blueprint (recommended, uses `render.yaml`)

1. Push this repository to GitHub (or GitLab / Bitbucket).
2. In the Render dashboard: **New +** → **Blueprint**.
3. Select this repo. Render reads `render.yaml` and pre-fills everything:
   - Build: `pip install -r requirements.txt`
   - Start: `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
   - Health check: `/_stcore/health`
4. (Optional) On the service's **Environment** tab, add `OLLAMA_API_KEY` to
   enable Ollama Cloud coaching. Skip it and the app runs fully without Ollama.
5. Click **Apply / Deploy**. First build takes a couple of minutes; then you get
   a public `https://complexity-lab.onrender.com`-style URL.

## Option B — Docker

The included `Dockerfile` works on Render (choose **Docker** runtime) and also on
Railway, Fly.io, or Google Cloud Run.

```bash
docker build -t complexity-lab .
docker run -p 8501:8501 complexity-lab   # open http://localhost:8501
```

On a host, the platform injects `$PORT` automatically; the container binds to it.

## Manual setup (no blueprint)

If you'd rather configure in the UI: **New +** → **Web Service** → connect repo, then:

- **Runtime:** Python
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
- **Health Check Path:** `/_stcore/health`
- **Environment:** `PYTHON_VERSION = 3.12.7`, and optionally `OLLAMA_API_KEY`

## Notes

- **Free plan** sleeps after inactivity and cold-starts on the next visit — fine
  for a portfolio demo. Upgrade the plan to keep it always-on.
- **History (SQLite)** lives at `data/complexity_lab.sqlite3` on the instance's
  ephemeral disk and resets on redeploy. Attach a Render Disk if you want it to
  persist. This doesn't affect any analysis, scoring, or benchmarking features.
- **Public demos:** enable *Static-only public mode* in the sidebar to disable
  code execution while keeping static analysis, scoring, reports, and coaching.
