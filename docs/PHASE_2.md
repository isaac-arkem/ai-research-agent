# Research Agent — Phase 2 Documentation

**Phase 2 — Headless API Endpoint & Pipeline Alignment**

This document covers Phase 2 only: the research chat API, conversation persistence, pipeline parameter translation, authentication, rate limits, errors, audit logging, and the local chat UI.

This service is **not RAG**. There is no vector store, no embeddings, and no retriever. The operator sends a question; the agent returns a structured scrape plan. Conversation history is ordinary OpenAI chat turns (`role` + `content`) stored in Supabase.

---

## 1. Phase brief

### Primary goal

A simple research chat with connected endpoints, tables, and backend architecture for **input in / output out**.

The operator types a research question (for example, “Find modest fashion creators in Saudi Arabia”) and receives a validated plan plus scrape jobs that match Social Listening pipeline contracts in arkgpt (`src/lib/community-mapper/social-listening.ts`).

### Deliverables (this phase)

| Deliverable | What shipped |
|---|---|
| API endpoint technical specification | Routes, request/response payloads, auth binding, rate limits, HTTP errors (400, 401, 422, 429, 502) |
| Pipeline parameter translation | Three operator flows mapped onto `creator_intelligence` and `reference_profiles` jobs |
| Automated endpoint tests & audit log | Integration tests for scenarios, strategy, context, conversation history; audit log for latency, prompt tracking, errors |

### Out of scope for Phase 2

- Executing scrapes (Jenkins / Railway)
- RAG / embeddings / Qdrant
- Developer-only SSE gates (`requireDeveloperOrThrow`)
- Frontend work inside arkgpt (this service is headless; a local chat page is included for operator testing)

---

## 2. Base URL and interactive docs

| Environment | Base URL |
|---|---|
| Local | `http://localhost:8000` |
| Interactive OpenAPI | `http://localhost:8000/docs` |
| ReDoc | `http://localhost:8000/redoc` |
| Research chat UI | `http://localhost:8000/` |

Run locally:

```bash
source venv/bin/activate
python -m uvicorn app.main:app --reload --port 8000
```

`OPENAI_API_KEY` is required in `.env` or the process environment. Copy `.env.example` if needed.

---

## 3. Endpoint index

| Method | Path | Auth | Rate limit | Purpose |
|---|---|---|---|---|
| `GET` | `/` | No | No | Local research chat UI |
| `GET` | `/health` | No | No | Liveness + markets loaded + auth flag |
| `GET` | `/docs` | No | No | OpenAPI (Swagger UI) |
| `POST` | `/ask` | Bearer when `AUTH_REQUIRED=true` | Yes (20 / 60s default) | Question in, plan + pipeline jobs out |
| `GET` | `/conversations/{conversation_id}` | Bearer when `AUTH_REQUIRED=true` | No | Reload a thread |

Success bodies use `ok: true`. Failures use HTTP 4xx/5xx with:

```json
{
  "ok": false,
  "error": "human-readable message",
  "code": "stable_error_code"
}
```

---

## 4. Authentication

Binding matches arkgpt `src/lib/community-mapper/auth.ts` → `requireSocialListeningUser`:

- `Authorization: Bearer <supabase access_token>`
- Cookie-only auth is rejected
- No developer-flag check (signed-in Social Listening users are enough)

Verification is `GET {NEXT_PUBLIC_SUPABASE_URL}/auth/v1/user` with the anon/public key as `apikey`, same idea as arkgpt `callerUserId()`.

### Local chat (default)

```
AUTH_REQUIRED=false
```

`POST /ask` works without a token. The caller is treated as user `local-dev`.

### Production (next to arkgpt)

```
AUTH_REQUIRED=true
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_PUBLIC_KEY=...
SUPABASE_SECRET_KEY=...
```

```bash
curl -X POST http://localhost:8000/ask \
  -H "Authorization: Bearer <supabase_access_token>" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Find modest fashion creators in Saudi Arabia"}'
```

| Status | Code | When |
|---|---|---|
| 401 | `auth_required` | Missing `Authorization: Bearer` |
| 401 | `invalid_token` | Token rejected by Supabase Auth |
| 502 | `auth_upstream` | Auth provider unreachable |

---

## 5. Rate limits

Applied to **`POST /ask` only**.

| Setting | Env var | Default |
|---|---|---|
| Max requests | `RATE_LIMIT_REQUESTS` | 20 |
| Window | `RATE_LIMIT_WINDOW_SECONDS` | 60 |

Key: authenticated user id, or client IP when the caller is `local-dev`.

On success, headers:

- `X-RateLimit-Limit`
- `X-RateLimit-Remaining`

On exceed: **429** `{ "ok": false, "code": "rate_limited" }` plus `Retry-After`.

---

## 6. HTTP error contract

| Status | Code | Meaning |
|---|---|---|
| 400 | `empty_prompt` | Prompt empty after sanitisation |
| 400 | `unknown_conversation` | `conversation_id` not found for this user |
| 401 | `auth_required` / `invalid_token` | Auth (when enabled) |
| 422 | `validation_error` | Request body failed Pydantic (missing `prompt`, bad UUID, etc.) |
| 422 | `validation_failed` | LLM returned a plan that failed the 9 rules |
| 429 | `rate_limited` | Too many `/ask` calls |
| 502 | `openai_failed` | OpenAI Chat Completions call failed |
| 502 | `json_extract_failed` | LLM text was not parseable JSON |
| 502 | `auth_upstream` | Supabase Auth lookup failed |

Example 422 (bad body):

```json
{
  "ok": false,
  "error": "Invalid request",
  "code": "validation_error",
  "details": [
    {
      "type": "missing",
      "loc": ["body", "prompt"],
      "msg": "Field required"
    }
  ]
}
```

---

## 7. `GET /health`

Public. No OpenAI call.

```http
GET /health
```

```json
{
  "status": "ok",
  "markets": 19,
  "taxonomy": 19,
  "db_connected": true,
  "auth_required": false
}
```

| Field | Meaning |
|---|---|
| `markets` | Country list size (Supabase `markets` or hardcoded fallback) |
| `taxonomy` | Niche alias count |
| `db_connected` | `true` if markets were loaded from Supabase |
| `auth_required` | Current `AUTH_REQUIRED` flag |

---

## 8. `POST /ask`

Core Phase 2 endpoint. Thin route: validate HTTP → load history → OpenAI chat → validate plan → translate pipeline jobs → persist → audit.

```http
POST /ask
Content-Type: application/json
```

### Request

```json
{
  "prompt": "Find modest fashion creators in Saudi Arabia",
  "model": "gpt-4o",
  "conversation_id": null,
  "conversation_history": []
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `prompt` | string | Yes | 1–2000 characters |
| `model` | string | No | Overrides `RESEARCH_AGENT_MODEL` |
| `conversation_id` | UUID | No | Send on follow-ups. Server history wins over `conversation_history` |
| `conversation_history` | `{ role, content }[]` | No | Same shape as arkemgpt-api chat. Used when there is no `conversation_id` |

`role` is `"user"` or `"assistant"`.

Follow-up:

```json
{
  "prompt": "Make it Instagram only",
  "conversation_id": "7f6117f9-f68f-4900-a653-0f676f771339"
}
```

### Success (200)

```json
{
  "ok": true,
  "conversation_id": "7f6117f9-f68f-4900-a653-0f676f771339",
  "message_id": "8e38d484-3179-4c29-a691-427315284cec",
  "flow": "discovery",
  "latency_ms": 5200,
  "llm_latency_ms": 4800,
  "prompt_tokens": 2100,
  "completion_tokens": 640,
  "plan": {
    "summary": "Find modest fashion creators in Saudi Arabia",
    "assumptions": ["Platform defaulted to both TikTok and Instagram."],
    "recommended_runs": [
      {
        "pipeline": "creator_intelligence",
        "countries": ["SA"],
        "platforms": ["tiktok", "instagram"],
        "hashtags": ["modestfashion", "ازياء", "hijabstyle"],
        "niche": "fashion_beauty",
        "max_creators": 50,
        "posts_per_source": 25,
        "recency_days": null,
        "title": "SA modest fashion discovery",
        "rationale": "Hashtag discovery in the named market."
      }
    ],
    "reference_accounts": [],
    "patterns_to_watch": ["Coverage of abaya vs western modest wear."],
    "content_angles": ["Day-to-night modest outfit breakdowns."],
    "risks": ["Arabic hashtag volume may be seasonal around Ramadan."]
  }
}
```

The plan's `recommended_runs` are stored in the `recommended_runs` table with proper foreign keys. When the operator approves a run, arkgpt translates it to scraper format and fires it.

Phase 2 **does not fire the scrape**. The console (or operator) triggers the run when they choose.

---

## 9. `GET /conversations/{conversation_id}`

```http
GET /conversations/7f6117f9-f68f-4900-a653-0f676f771339
```

```json
{
  "id": "7f6117f9-f68f-4900-a653-0f676f771339",
  "title": "Find modest fashion creators in Saudi Arabia",
  "messages": [
    {
      "id": "...",
      "role": "user",
      "content": "Find modest fashion creators in Saudi Arabia",
      "created_at": "2026-09-03T02:37:00Z"
    },
    {
      "id": "...",
      "role": "assistant",
      "content": "Find modest fashion creators in Saudi Arabia",
      "flow": "discovery",
      "plan": { },
      "created_at": "2026-09-03T02:37:05Z"
    }
  ]
}
```

Unknown id → **400** `unknown_conversation`. Threads are scoped to the authenticated user (or `local-dev`).

---

## 10. Three defined flows

Prompt stories live in `app/services/prompt.py`. Classification lives in `app/services/flows.py`.

| Flow | Operator story | Output | Pipeline |
|---|---|---|---|
| **1. Discovery** | “Find modest fashion creators in Saudi Arabia” | One `recommended_runs` entry | `creator_intelligence` · `max_creators` 50 |
| **2. Deep / multi-market** | “Compare cooking creators across the Gulf” | One run **per country**, localized hashtags | `creator_intelligence` · `max_creators` 100 |
| **3. Reference** | “Scrape @khloekardashian on Instagram” | `reference_accounts` only | `reference_profiles` |

Also returned: `mixed` (runs + named accounts) and `off_topic` (empty plan, guidance in `risks`).

Reference `payload` example:

```json
{
  "pipeline": "reference_profiles",
  "title": "Scrape @khloekardashian",
  "accounts": ["https://www.instagram.com/khloekardashian"],
  "account_niches": { "khloekardashian": "fashion_beauty" },
  "posts_per_source": 25
}
```

---

## 11. Conversation architecture (not RAG)

Same idea as arkemgpt-api chat: the model sees a message list, not a retrieved document set.

```
Operator prompt
    → sanitize (length, control chars)
    → system message (identity, markets, taxonomy, flow rules, JSON schema)
    → prior turns as user / assistant (last 8)
    → current question as a fenced user message
    → OpenAI Chat Completions (`response_format: json_object`)
    → validate 9 rules
    → classify flow
    → persist plan + audit
```

History sources:

1. If `conversation_id` is set → load from Supabase (or in-memory fallback).
2. Else if `conversation_history` is sent → use that list (arkemgpt frontend pattern).
3. Else → new thread.

Libraries (Phase 2):

| Package | Role |
|---|---|
| FastAPI + Uvicorn | HTTP |
| openai | Chat Completions SDK |
| supabase | Markets, conversations, audit |
| pydantic / pydantic-settings | Plan schema + env |
| httpx | JWT verify against Supabase Auth |
| pytest | Endpoint and flow tests |

Not used in this phase: LangChain, LangGraph, Qdrant, embeddings, RAG retrievers.

---

## 12. Tables

Apply `sql/001_research_chat.sql` in the Supabase SQL editor. The Python client uses the service role (bypasses RLS). Until the migration runs, `/ask` still works via an in-memory store for the process lifetime.

### `research_conversations`

| Column | Type |
|---|---|
| `id` | uuid PK |
| `user_id` | text |
| `title` | text |
| `created_at` / `updated_at` | timestamptz |

`user_id` is text so `local-dev` can persist as well as a UUID.

### `research_messages`

| Column | Type |
|---|---|
| `id` | uuid PK |
| `conversation_id` | uuid FK |
| `role` | `user` \| `assistant` |
| `content` | text |
| `flow` | text |
| `plan` | jsonb |
| `created_at` | timestamptz |

### `research_audit_logs`

One row per `/ask`: `prompt_hash`, `prompt_preview` (first 240 chars), `model`, `flow`, `ok`, `error` / `error_code`, `latency_ms`, `llm_latency_ms`, `prompt_tokens`, `completion_tokens`, `status_code`.

Structured log line (always, even if the insert 403s):

```
ask user=... conv=... flow=discovery ok=True status=200 latency_ms=5200 llm_ms=4800 prompt_hash=... error_code=None
```

Markets at startup still come from existing table `markets` (`id, country_code, name, region, platform`). If that fetch fails, the hardcoded list in `app/data/markets.py` is used.

---

## 13. Folder layout (Phase 2)

```
app/
  main.py                          # create_app()
  api/v1/api.py
  api/v1/endpoints/health.py
  api/v1/endpoints/research.py     # POST /ask, GET /conversations/{id}
  core/config.py
  core/auth.py                     # Bearer JWT
  core/errors.py
  core/rate_limit.py
  core/supabase.py
  core/dependencies.py             # markets + taxonomy context
  models/requests.py
  models/responses.py
  models/domain.py                 # ResearchPlan, AgentResult
  services/agent.py
  services/prompt.py
  services/validator.py
  services/flows.py                # flow classification
  services/conversations.py
  services/audit.py
  static/index.html                # local chat
sql/001_research_chat.sql
tests/
```

---

## 14. Environment

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | — | LLM |
| `RESEARCH_AGENT_MODEL` | No | `gpt-4o` | OpenAI chat model |
| `NEXT_PUBLIC_SUPABASE_URL` | For DB / auth | — | Same name as arkgpt |
| `SUPABASE_SECRET_KEY` | For DB | — | Service role |
| `NEXT_PUBLIC_SUPABASE_PUBLIC_KEY` | For JWT verify | — | Anon / publishable key |
| `AUTH_REQUIRED` | No | `false` | Require Bearer |
| `RATE_LIMIT_REQUESTS` | No | `20` | `/ask` cap |
| `RATE_LIMIT_WINDOW_SECONDS` | No | `60` | Window |
| `CORS_ORIGINS` | No | localhost 3000 and 8000 | Comma-separated |

---

## 15. Frontend integration

Interactive chat for operators: open `http://localhost:8000/`.

For arkgpt or any client, the contract is the same as arkemgpt-api chat: POST JSON, keep `conversation_id`, optionally send `conversation_history`.

```ts
const API_BASE = "http://localhost:8000";

type ChatMessage = { role: "user" | "assistant"; content: string };

export async function askResearch(
  prompt: string,
  conversationId?: string,
  conversationHistory: ChatMessage[] = [],
  token?: string
) {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (token) headers.Authorization = `Bearer ${token}`;

  const response = await fetch(`${API_BASE}/ask`, {
    method: "POST",
    headers,
    body: JSON.stringify({
      prompt,
      conversation_id: conversationId ?? null,
      conversation_history: conversationHistory,
    }),
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error ?? response.statusText);
  }
  return data;
}
```

To start a scrape later (not this phase), read `recommended_runs` from the plan, translate to scraper format, and POST to arkgpt Social Listening runs with the same Bearer the console already uses.

---

## 16. Tests (Phase 2)

```bash
python -m pytest tests/ -q
```

Coverage:

- Flow 1 / 2 / 3 classification
- Mixed plan (runs + reference accounts)
- `/ask` 200 input → output
- Follow-up `conversation_id` passes prior chat turns to the model
- Client `conversation_history` without an id
- 400 unknown conversation / empty prompt
- 422 request schema / invalid plan
- 429 rate limit
- 502 OpenAI failure
- 401 missing and invalid Bearer
- `GET /conversations/{id}` round trip
- `GET /` chat UI and `GET /health`

LLM calls are mocked at the route. Flow tests use fixture plans only.

---

## 17. Request flow (Phase 2)

```
Client POST /ask
  → CORS
  → Bearer auth (if AUTH_REQUIRED)
  → sliding-window rate limit
  → load or create conversation
  → persist user turn
  → generate_research_plan
        sanitize → OpenAI chat messages → Chat Completions
        → JSON extract → 9-rule validator
        → classify_flow
  → persist assistant turn (plan + flow)
  → write_audit (log + research_audit_logs)
  → 200 plan | 4xx/5xx error body
```
