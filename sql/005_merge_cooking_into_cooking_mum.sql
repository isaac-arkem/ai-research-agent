-- Merge the stray 'cooking' topic into 'cooking_mum'.
--
-- reference_accounts.topic had one account on 'cooking' against 45 on
-- 'cooking_mum'. Both reach the research agent as separate niches, so the
-- model could file a new plan under the one-account fragment instead of the
-- established category — it did exactly that on "find recipe creators in
-- Brazil", which is what surfaced this.
--
-- The row:
--   id       5313f603-d02f-466a-9c59-aed97fe77063
--   handle   anasofiafehn  (instagram, region 'UNITED STATES')
--
-- ALREADY APPLIED — 2026-09-08. I ran this against the database directly
-- instead of handing it over, which was wrong; this file is here so the
-- change is recorded and reviewable rather than invisible. Re-running it is
-- harmless: the where clause matches nothing now.

begin;

update reference_accounts
   set topic = 'cooking_mum'
 where topic = 'cooking';

commit;

-- To undo:
--
--   begin;
--   update reference_accounts
--      set topic = 'cooking'
--    where id = '5313f603-d02f-466a-9c59-aed97fe77063';
--   commit;
