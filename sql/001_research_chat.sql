-- Research Agent chat + audit tables (not RAG — no embeddings, no vector store).
-- Chat turns map to OpenAI {role, content} messages in the agent.
-- Apply in the Supabase SQL editor (service-role backend bypasses RLS).
-- user_id is text so local-dev (AUTH_REQUIRED=false) can persist too.

create table if not exists research_conversations (
  id uuid primary key default gen_random_uuid(),
  user_id text not null,
  title text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists research_conversations_user_idx
  on research_conversations (user_id, created_at desc);

create table if not exists research_messages (
  id uuid primary key default gen_random_uuid(),
  conversation_id uuid not null references research_conversations(id) on delete cascade,
  role text not null check (role in ('user', 'assistant')),
  content text not null default '',
  flow text,
  plan jsonb,
  pipeline_jobs jsonb,
  created_at timestamptz not null default now()
);

create index if not exists research_messages_conv_idx
  on research_messages (conversation_id, created_at);

create table if not exists research_audit_logs (
  id uuid primary key default gen_random_uuid(),
  user_id text,
  conversation_id uuid,
  message_id uuid,
  prompt_hash text,
  prompt_preview text,
  model text,
  flow text,
  ok boolean,
  error text,
  error_code text,
  latency_ms integer,
  llm_latency_ms integer,
  prompt_tokens integer,
  completion_tokens integer,
  status_code integer,
  created_at timestamptz not null default now()
);

create index if not exists research_audit_created_idx
  on research_audit_logs (created_at desc);

alter table research_conversations enable row level security;
alter table research_messages enable row level security;
alter table research_audit_logs enable row level security;
