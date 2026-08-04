# "hybrid linked" — Setup Guide

Everything you need to get the connected workflow running. Files I've created for you (in this folder):

- **`LinkedIn Draft Queue.xlsx`** — the news-pipeline log (tabs: `Drafts`, `Articles`)
- **`kaushik Personal Brand.xlsx`** — the personal-brand store (tabs: `Context` pre-filled with your profile, `Content`)
- **`hybrid-linked-SETUP-GUIDE.md`** — this guide

I can't reach into your Google Drive / LinkedIn / Telegram accounts to create live credentials, so those steps are yours — but everything below is spelled out.

---

## 1. Create the two Google Sheets

Upload each `.xlsx` to Google Drive and open it with Google Sheets (or in Sheets: *File → Import → Upload*, choose "Insert new sheet(s)"). Keep the **tab names exactly** as they are — the workflow matches tabs by name.

### A. LinkedIn Draft Queue
- **`Drafts`** tab — one row per generated post (57 columns, headers already in place). The workflow appends here and updates Telegram metadata.
- **`Articles`** tab — the dedup log so the same story is never posted twice (`article_key`, `normalized_url`, `article_title`, …).

### B. kaushik Personal Brand
- **`Context`** tab — who you are, audience, topics, writing rules. **Row 2 is pre-filled with your real profile** — edit it anytime.
- **`Content`** tab — short/long/final post history written at posting time.

---

## 2. Point the workflow at your new sheet IDs

After importing, each sheet gets a new ID (the long string in its URL:
`docs.google.com/spreadsheets/d/`**`THIS_PART`**`/edit`). Open the workflow and update the **Document ID** field in these nodes:

| Node | Sheet | Tab |
|---|---|---|
| Google Sheets Append Draft Row | LinkedIn Draft Queue | Drafts |
| Google Sheets Update Telegram Metadata | LinkedIn Draft Queue | Drafts |
| Google Sheets Get Recent Draft History | LinkedIn Draft Queue | Drafts |
| Google Sheets Get Existing Articles | LinkedIn Draft Queue | Articles |
| Get Context | kaushik Personal Brand | Context |
| Get row(s) in sheet | kaushik Personal Brand | Context |
| Save context | kaushik Personal Brand | Context |
| Save | kaushik Personal Brand | Content |
| Update | kaushik Personal Brand | Content |

> Note: in the current workflow, *Get Existing Articles* points at a **different** doc ID from the other Draft-Queue nodes. Set all four Draft-Queue nodes to the **same** `LinkedIn Draft Queue` ID, and select the correct tab (`Drafts` vs `Articles`) on each.

---

## 3. Credentials to connect (in n8n → Credentials)

| Service | Used by | Notes |
|---|---|---|
| **Google Sheets OAuth2** | all Google Sheets nodes | same Google account that owns the two sheets |
| **Google Drive OAuth2** | List / Download Photos | needs access to your "Reference Photos" folder |
| **Telegram API** (bot token) | all Telegram nodes | create a bot via @BotFather |
| **LinkedIn OAuth2** | Post to LinkedIn | your LinkedIn account |
| **Groq API** | Short Post / Long Post / Groq nodes | api key from console.groq.com |
| **Ollama** (Bearer token) | Ollama Draft / Humanize / Chat models | uses `https://ollama.com/api/chat` (cloud) |
| **Google Gemini (PaLM) API** | Gemini Pick Best | api key from aistudio.google.com |

---

## 4. Variables & IDs to set

- **`PERSON_REFERENCE_URL`** (n8n → Variables) — a **public** URL to a clear selfie of you. The image-prompt agent tells the image tool to keep your exact face from this photo. *(I can't create your photo — upload one somewhere public and paste the link.)*
- **Telegram chat ID `1238403806`** — hardcoded in the Telegram nodes (e.g. *Choose Version*). If your chat ID differs, message @userinfobot to get yours and update it.
- **Google Drive "Reference Photos" folder** (`List Photos` node) — currently `1HQqsRJe6DLTZa7GUSfCZfuK22J0TTQtL`. Point it at a Drive folder holding a few reference photos of you, or update the ID.

---

## 5. How the connected flow runs

1. **Schedule Trigger** fires **Mon & Thu 10:00**.
2. Fetches latest AI/ML news → dedup against `Articles` → picks the best story.
3. Drafts a **problem-solver post** using your profile (from `Build Data Scientist Positioning` + `Context`).
4. **Bridge News To Photo Prompt** hands the draft to **Generate Photo Prompt**, which writes a **unique-style DSLR/QHD image prompt** showing what you're building.
5. Telegram sends you the **post + image prompt** (+ a reference photo).
6. You generate the image from the prompt and **upload it back** to the bot.
7. You tap **Approve** → **Post to LinkedIn** → logged to `Content`.

You can also still start it manually by messaging the bot (the original interactive path is intact).

---

## 6. Before you go live

- The workflow is currently **inactive**. Test first — a live run will post to LinkedIn.
- Suggested test: run it once manually in n8n and watch the Telegram handoff before activating the schedule.
- Pre-existing validation warnings (Telegram/Sheets discriminators, `Prepare Sheet Row` ids, orphaned card nodes) are unrelated to the connection edits and don't block execution; tidy-up is optional — happy to do it if you want.
