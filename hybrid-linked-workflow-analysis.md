# Workflow Analysis — "hybrid linked"

- **Workflow ID:** `1ToTRwHydJsyHx1t`
- **Status:** Inactive (not currently running on schedule)
- **Created:** 2026-01-27 · **Last updated:** 2026-08-03
- **Node count:** 91
- **External services:** Google Sheets, Google Drive, Telegram, LinkedIn, Groq, Ollama, Google Gemini, plus many RSS/HTTP news sources and an image-render API.

---

## 1. What the workflow does (high level)

This is an end-to-end **AI LinkedIn content engine** that positions you as a data scientist. On a schedule it pulls the latest AI/ML news from a wide set of sources, deduplicates against what has already been posted, selects the strongest story, drafts a LinkedIn post with LLMs, "humanizes" it, generates a branded social-card image (choosing from a rotating library of visual templates), sends the draft + image to you on Telegram for approval, logs everything to Google Sheets, and — through an interactive Telegram-driven branch — finalizes a photo/prompt and publishes to LinkedIn.

It is "hybrid" because it combines two modes: a **fully-automated card pipeline** (news → templated image → post) and an **interactive human-in-the-loop branch** (Telegram Q&A, uploaded/selected photos, AI-picked best image, manual approve/regenerate).

---

## 2. Trigger & scheduling

- **Schedule Trigger** (`scheduleTrigger`) — the only trigger in the workflow. Rule: interval by **weeks**, on **days [1, 4]** (Monday and Thursday), at **hour 10:00**. So the automated pipeline is designed to run twice a week mid-morning.

Note: `triggerCount` is 0 and the workflow is inactive, so it will not fire until published/activated.

---

## 3. News ingestion (sources)

RSS reader nodes (`rssFeedRead`), each pulling one feed:

- RSS Read arXiv stat.ML
- RSS Read Google Research Blog
- RSS Read Google AI Blog
- RSS Read Hugging Face Blog
- RSS Read OpenAI News
- RSS Read Towards Data Science
- RSS Read KDnuggets
- RSS Read VentureBeat
- RSS Read TechCrunch AI
- RSS Read AI Blog
- RSS Read Testing Catalog

HTTP-based source:

- **HTTP Read MarkTechPost Posts** (`httpRequest`) — fetches MarkTechPost via API/HTML.
- **Normalize MarkTechPost Posts** (`code`) — reshapes those posts into the same item schema as the RSS feeds.

Aggregation:

- **Merge RSS Feeds** (`merge`) — combines all feed outputs into one candidate stream.

---

## 4. Candidate selection & deduplication

- **Prepare Candidate News** (`code`) — cleans/normalizes the merged articles into candidate objects.
- **Build Candidate Keys** (`code`) — derives a stable key (e.g. normalized URL/title) per candidate for dedup.
- **Google Sheets Get Existing Articles** (`googleSheets`) — reads the log of already-used articles.
- **Filter Already Used Articles** (`code`) — removes candidates that match previously posted keys.
- **IF Fresh Candidates?** (`if`) — branches on whether any unused candidates remain.
  - **False → Telegram Send No Fresh Candidates** (`telegram`) — notifies you when nothing new is available.
- **Parse Selected Story** (`code`) — picks/parses the winning story and its metadata (score, selection reason, alternates).

---

## 5. Style, positioning & drafting (LLMs)

- **Google Sheets Get Recent Draft History** (`googleSheets`) — pulls recent drafts so the style engine avoids repetition.
- **Build Style Memory** (`code`) — builds a memory of recent styles/templates used.
- **Choose Post Style** (`code`) — selects post style, opening type, and CTA type for variety.
- **Build Audience Angle** (`code`) — frames the story for the target audience.
- **Build Data Scientist Positioning** (`code`) — adds your personal-brand angle (fresher signal, hiring signal, portfolio proof, IT-team application, time-saver, etc.).
- **Groq Short Post** / **Groq Long Post** (`lmChatGroq`, model `openai/gpt-oss-120b`) — generate short and long variants of the LinkedIn copy.
- **Groq Questions** (`lmChatGroq`) — generates clarifying questions for the interactive branch.
- **Parse Draft JSON** (`code`) — parses the LLM output into structured fields (hook, post, hashtags, headline, etc.).

---

## 6. Humanization

- **Ollama Humanize Draft** (`httpRequest` to Ollama, model `gemma4:31b-cloud`) — rewrites the draft to sound more natural/less AI.
- **Parse Humanized Draft JSON** (`code`) — parses the humanized result back into structured fields.

---

## 7. Visual template selection & card rendering

- **Template Registry** (`code`) — defines the library of available card designs.
- **Choose Daily Template** (`code`) — rotates/selects the day's template so visuals stay varied.
- **Switch Card Template** (`switch`) — routes to the matching "Build Card HTML" node.

Card builders (each a `code` node producing HTML for one design):

- Build Card HTML - Editorial Cover
- Build Card HTML - Magazine Pullquote
- Build Card HTML - Dark Terminal
- Build Card HTML - Sticky Note Board
- Build Card HTML - Timeline Story
- Build Card HTML - Dashboard Stat
- Build Card HTML - Quote Focus
- Build Card HTML - Minimal Brief
- Build Card HTML - Insight Split
- Build Card HTML - Hiring Signal

Rendering:

- **Render Social Card Image** (`httpRequest`) — sends the HTML to an image-render API to produce the final card PNG.

---

## 8. AI scene / personal-branding imagery

- **Build Data Scientist Positioning** feeds imagery cues (also used above).
- **IF AI Scene API Configured** (`if`) — only generates an AI scene if the API is set up.
- **Generate AI Workspace Scene** (`httpRequest`) — calls an image-gen API for a workspace scene.
- **Format AI Workspace Scene HTML** (`code`) / **Generate Workspace Scene Image** (`code`) — format and produce the scene image.
- **Apply Personal Branding Overlay** (`code`) — overlays your branding on the image.
- **Download Person Reference Image** (`httpRequest`) + **IF Approval Image URL Exists** (`if`) — pulls a reference photo of you if available and branches accordingly.

---

## 9. Persistence (Google Sheets log)

- **Prepare Sheet Row** (`set`) — assembles a wide row with ~50+ fields: `draft_id`, article metadata, `linkedin_hook`, `linkedin_post`, `full_linkedin_post`, hashtags, image headline/subheadline/alt-text, `telegram_chat_id`/`message_id`/`callback_key`, `linkedin_post_id`/`url`, `post_style`, `design_template`, `approval_state`, `rejection_count`, timestamps, etc.
- **Format Draft Assets** (`code`) — bundles the final text + image assets.
- **Google Sheets Append Draft Row** (`googleSheets`) — appends the new draft record.
- **Google Sheets Update Telegram Metadata** (`googleSheets`) — writes back Telegram message IDs/callback keys after sending.

---

## 10. Telegram approval (human-in-the-loop)

- **Telegram Send Approval Message** (`telegram`) — sends the draft text for approval.
- **Telegram Send Approval Photo** (`telegram`) — sends the rendered card/image.
- **Merge Draft And Telegram Response** (`merge`) — joins the draft with the Telegram reply.
- **Send For Approval** / **Approved?** (`telegram` / `if`) — approval gate.
- **Notify Regenerating** / **Rebuild For Regen** (`telegram` / `code`) — regeneration path on rejection.
- **Remind to Reply** (`telegram`) — nudge if you don't respond.

---

## 11. Interactive photo-prompt & LinkedIn publishing branch

This cluster is the interactive/agentic side (driven by Telegram interaction):

- **Format Questions Message** / **Store Pending Questions** / **Build Answer Items** (`code`) — manage a Q&A exchange with you.
- **Save Post Context** (`code`) — persists context between steps.
- **Generate Photo Prompt** (`@n8n/n8n-nodes-langchain.agent`) — an AI agent that writes an image prompt, backed by:
  - **Ollama Chat Model (Prompt)** and **Ollama Chat Model (Structure)** (`lmChatOllama`) — the agent's LLMs.
  - **Photo Prompt Memory** (`memoryBufferWindow`) — conversational memory.
- **Send Photo Prompt** / **Send Reference Photo** (`telegram`) — send the prompt/reference image to you.
- **Has Photo?** (`if`) + **Prepare Uploaded Photo** (`code`) — handle a photo you upload.
- Google Drive photo path: **List Photos** → **Download All Photos** → **Combine Images** → **Gemini Pick Best** (`googleGemini`) → **Pick Best Photo** (`code`) → **Download Chosen Photo** (`googleDrive`) — auto-selects the best image from a Drive folder.
- **Split Prompt For Telegram** / **Choose Version** (`code` / `telegram`) — let you choose between short/long versions.
- **Send Post Preview** (`telegram`) — final preview.
- **Pick Text & File** (`code`) — assembles the final text + image file.
- **Download Final Photo** (`telegram`) — retrieves the chosen photo.
- **Post to LinkedIn** (`linkedIn`) — publishes the post.
- **Notify Posted** (`telegram`) — confirms publication.

---

## 12. Observations & things to check

1. **Only one trigger.** The interactive branch (Has Photo?, Send Photo Prompt, Approved?, etc.) implies a Telegram-driven entry point, but the only trigger node present is the Schedule Trigger. There is no Telegram Trigger / webhook node in the graph, so that branch may rely on an external webhook not modeled here, or parts of it may be disconnected. Worth verifying the connections into that cluster.
2. **Inactive / triggerCount 0** — nothing runs until you publish/activate it.
3. **Many credentials required** — Google Sheets, Google Drive, Telegram bot, LinkedIn OAuth, Groq API, Ollama endpoint, Gemini API, and the image-render/AI-scene APIs. Any missing credential will break its branch.
4. **External render/AI-scene APIs** are HTTP calls to services that must be configured (`IF AI Scene API Configured` guards one of them).
5. **Heavy Code-node usage** (~40 `code` nodes) — most logic lives in JavaScript, so behavior depends on those scripts rather than native node config. A deeper per-script review would be the next step if you want correctness/robustness checks.

---

*Analysis based on the workflow's node inventory, types, trigger rule, model configuration, and service usage. Individual Code-node JavaScript bodies were not line-by-line reviewed — say the word if you want a deep dive into specific scripts.*
