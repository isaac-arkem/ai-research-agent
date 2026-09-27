-- Owner RLS for the tables the research agent writes.
--
-- sql/001 enabled RLS and added no policies, which locked the tables to
-- service_role only. That is still how the FastAPI app writes (it uses the
-- secret key, which bypasses RLS). These policies are for a signed-in client
-- talking to PostgREST. Conversations and messages are owner-scoped: each
-- user sees and writes only their own rows. research_audit_logs is readable
-- by any signed-in user (it is an ops log); inserts still have to be yours.
--
-- "Authenticated can write" does NOT mean any signed-in user can write any
-- row. Inserts and updates must be owned by auth.uid() — otherwise a client
-- could attach messages to someone else's conversation or stamp another
-- user_id on an audit row.
--
-- No DELETE policies. Dropping a thread is not a client operation today.
--
-- user_id is text (sql/001) and auth.uid() is uuid, so every comparison
-- casts. (select auth.uid()) is the initplan wrap used on
-- apify_supported_countries, so the uid is resolved once per statement.
--
-- research_messages has no user_id. Ownership is the parent conversation.

begin;

grant select, insert, update on table public.research_conversations to authenticated;
grant select, insert, update on table public.research_messages to authenticated;
grant select, insert on table public.research_audit_logs to authenticated;

-- ── research_conversations ──────────────────────────────────────────

drop policy if exists authenticated_select_own_research_conversations
  on public.research_conversations;
drop policy if exists authenticated_insert_own_research_conversations
  on public.research_conversations;
drop policy if exists authenticated_update_own_research_conversations
  on public.research_conversations;

create policy authenticated_select_own_research_conversations
  on public.research_conversations
  for select
  to authenticated
  using (user_id = (select auth.uid())::text);

create policy authenticated_insert_own_research_conversations
  on public.research_conversations
  for insert
  to authenticated
  with check (user_id = (select auth.uid())::text);

create policy authenticated_update_own_research_conversations
  on public.research_conversations
  for update
  to authenticated
  using (user_id = (select auth.uid())::text)
  with check (user_id = (select auth.uid())::text);

-- ── research_messages ───────────────────────────────────────────────

drop policy if exists authenticated_select_own_research_messages
  on public.research_messages;
drop policy if exists authenticated_insert_own_research_messages
  on public.research_messages;
drop policy if exists authenticated_update_own_research_messages
  on public.research_messages;

create policy authenticated_select_own_research_messages
  on public.research_messages
  for select
  to authenticated
  using (
    exists (
      select 1
      from public.research_conversations c
      where c.id = conversation_id
        and c.user_id = (select auth.uid())::text
    )
  );

create policy authenticated_insert_own_research_messages
  on public.research_messages
  for insert
  to authenticated
  with check (
    exists (
      select 1
      from public.research_conversations c
      where c.id = conversation_id
        and c.user_id = (select auth.uid())::text
    )
  );

create policy authenticated_update_own_research_messages
  on public.research_messages
  for update
  to authenticated
  using (
    exists (
      select 1
      from public.research_conversations c
      where c.id = conversation_id
        and c.user_id = (select auth.uid())::text
    )
  )
  with check (
    exists (
      select 1
      from public.research_conversations c
      where c.id = conversation_id
        and c.user_id = (select auth.uid())::text
    )
  );

-- ── research_audit_logs ─────────────────────────────────────────────
-- SELECT any signed-in user (ops log, not chat). INSERT still has to be
-- your own row. No UPDATE: audit rows are not edited after the fact.

drop policy if exists authenticated_select_own_research_audit_logs
  on public.research_audit_logs;
drop policy if exists authenticated_select_research_audit_logs
  on public.research_audit_logs;
drop policy if exists authenticated_insert_own_research_audit_logs
  on public.research_audit_logs;

create policy authenticated_select_research_audit_logs
  on public.research_audit_logs
  for select
  to authenticated
  using ((select auth.uid()) is not null);

create policy authenticated_insert_own_research_audit_logs
  on public.research_audit_logs
  for insert
  to authenticated
  with check (
    user_id = (select auth.uid())::text
    and (
      conversation_id is null
      or exists (
        select 1
        from public.research_conversations c
        where c.id = conversation_id
          and c.user_id = (select auth.uid())::text
      )
    )
  );

commit;
