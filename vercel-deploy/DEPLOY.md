# Deploying Complexity Lab with a Vercel URL

Streamlit can't run on Vercel's serverless model, so this uses the minimal, no-code-change setup:

1. **Host the real Streamlit app** on a free Streamlit-native platform (gives a live URL).
2. **Deploy this `vercel-deploy/` folder to Vercel** as a static site that forwards to that URL.

Result: `your-project.vercel.app` opens a branded page and redirects to the fully working app. Your app code is untouched.

---

## Step 1 — Host the Streamlit app (pick ONE)

### Option A: Streamlit Community Cloud (recommended, ~2 min)
1. Push this whole repo to GitHub (public or private).
2. Go to https://share.streamlit.io → **Create app** → pick your repo.
3. Set **Main file path** to `streamlit_app.py`, branch = your branch.
4. (Optional) In **Advanced settings → Secrets**, add:
   ```
   OLLAMA_API_KEY = "your_key_here"
   ```
5. **Deploy.** Copy the resulting URL, e.g. `https://complexity-lab.streamlit.app`.

### Option B: Hugging Face Spaces (no GitHub needed)
1. https://huggingface.co/new-space → SDK = **Streamlit**.
2. Upload the repo files (or link a repo). Set app file to `streamlit_app.py`.
3. Space → **Settings → Secrets** → add `OLLAMA_API_KEY`.
4. Copy the Space URL, e.g. `https://username-complexity-lab.hf.space`.

> Tip: the Ollama key is optional — the app works fully without it.

---

## Step 2 — Point the forwarder at your live URL

Open `vercel-deploy/index.html` and paste the URL from Step 1 into the one marked line:

```js
const APP_URL = "https://complexity-lab.streamlit.app";
```

Save. That's the only edit.

---

## Step 3 — Deploy the forwarder to Vercel

### Easiest (dashboard):
1. Put the `vercel-deploy/` folder in its own GitHub repo **or** deploy the whole repo and set the **Root Directory** to `vercel-deploy`.
2. https://vercel.com/new → import the repo.
3. Framework preset = **Other** (it's static HTML — no build needed).
4. **Deploy.** You get `https://your-project.vercel.app`.

### Or via CLI:
```bash
cd "vercel-deploy"
npx vercel --prod
```
Follow the prompts (login the first time). It prints your `*.vercel.app` URL.

---

## Verify
Open the Vercel URL → you should see the "Complexity Lab" loading card, then be redirected to the live app. If it says *"Set APP_URL…"*, you skipped Step 2.

## Notes
- To later run the app *natively* on Vercel you'd have to replace Streamlit with a Next.js + serverless-Python rewrite — a full rebuild. This forwarder avoids that entirely.
- Updating the app: redeploy on the Step-1 host; the Vercel URL keeps working (it just forwards).
