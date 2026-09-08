-- RLS for apify_supported_countries.
--
-- 002 enabled RLS but added no policies, which means service_role only. The
-- research agent reads with the secret key and so is unaffected either way —
-- this is about whether a browser client can read the list.
--
-- WHICH PATTERN THIS FOLLOWS
-- --------------------------
-- Two shapes already exist in this database:
--
--   1. Shared reference corpus — markets, creators, posts, media_assets,
--      clips, trends. Global scrape/reference data, no org column, populated
--      by an external pipeline. Each carries an authenticated SELECT policy
--      (authenticated_select_markets, _clips, _trends, ...) and no write
--      policy at all; writes go through service_role, which bypasses RLS.
--
--   2. The research agent's own tables — research_conversations,
--      research_messages, research_audit_logs (sql/001). RLS on, no policies,
--      service_role only. Correct for those: they hold per-user conversation
--      content that no client should read directly.
--
-- apify_supported_countries is squarely pattern 1. It is a global list of
-- country codes with no user or org dimension — the same posture as `markets`,
-- which it functionally parallels for this agent. So it gets the same policy,
-- named to match the existing convention.
--
-- TRADE-OFF, EXPLICITLY: this makes the country list readable by ANY signed-in
-- user, including one who belongs to no org. That is the exposure `markets`
-- and the rest of the corpus already have, and the content is ISO country
-- codes and names — public reference data, nothing proprietary. The tighter
-- alternative is to leave the table service_role-only and let the research API
-- be the only way to see it; that costs nothing today, since the agent reads
-- server-side, but it means any future Planner UI that wants to show or filter
-- supported countries has to add an endpoint rather than select the table.
--
-- SELECT only. No insert/update/delete policy is added: the country list is
-- maintained by scripts/sync_apify_countries.py using the service key, and
-- clients must not be able to edit which countries are scrapeable.

begin;

-- Mirrors authenticated_select_markets / _clips / _trends:
-- SELECT only, any signed-in user, no write path added.
drop policy if exists authenticated_select_apify_supported_countries
  on public.apify_supported_countries;

create policy authenticated_select_apify_supported_countries
  on public.apify_supported_countries
  for select
  to authenticated
  using ((select auth.uid()) is not null);

commit;
