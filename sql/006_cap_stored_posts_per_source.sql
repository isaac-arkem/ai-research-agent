-- Cap posts_per_source at 100 in stored plans.
--
-- The limit was 200 and is now 100. The pydantic model enforces it on read,
-- so any stored plan above the cap fails to load and takes its conversation
-- with it. At the time of writing exactly one row is affected:
--
--   research_messages.id = f4b3f40e-9e04-42f3-90be-0cfbd468f1e5
--   (conversation 92d139d6-1725-4e71-b819-3ce912f90dda, posts_per_source 200)
--
-- The UPDATE is written against the whole table rather than that id, so it
-- stays correct if another lands before it is run. It is idempotent: rows
-- already at or below 100 are untouched.

-- Look first.
select
    m.id,
    m.conversation_id,
    job ->> 'posts_per_source' as posts_per_source,
    key                        as job_kind
from research_messages m
cross join lateral (values ('recommended_runs'), ('reference_accounts')) as k(key)
cross join lateral jsonb_array_elements(coalesce(m.plan -> k.key, '[]'::jsonb)) as job
where (job ->> 'posts_per_source') ~ '^[0-9]+$'
  and (job ->> 'posts_per_source')::int > 100;

-- Then cap it, in both job kinds, leaving every other field alone.
with capped as (
    select
        m.id,
        jsonb_set(
            jsonb_set(
                m.plan,
                '{recommended_runs}',
                coalesce((
                    select jsonb_agg(
                        case
                            when (job ->> 'posts_per_source') ~ '^[0-9]+$'
                             and (job ->> 'posts_per_source')::int > 100
                            then jsonb_set(job, '{posts_per_source}', '100'::jsonb)
                            else job
                        end
                    )
                    from jsonb_array_elements(coalesce(m.plan -> 'recommended_runs', '[]'::jsonb)) as job
                ), '[]'::jsonb)
            ),
            '{reference_accounts}',
            coalesce((
                select jsonb_agg(
                    case
                        when (job ->> 'posts_per_source') ~ '^[0-9]+$'
                         and (job ->> 'posts_per_source')::int > 100
                        then jsonb_set(job, '{posts_per_source}', '100'::jsonb)
                        else job
                    end
                )
                from jsonb_array_elements(coalesce(m.plan -> 'reference_accounts', '[]'::jsonb)) as job
            ), '[]'::jsonb)
        ) as plan
    from research_messages m
    where m.plan is not null
      and exists (
          select 1
          from (values ('recommended_runs'), ('reference_accounts')) as k(key)
          cross join lateral jsonb_array_elements(coalesce(m.plan -> k.key, '[]'::jsonb)) as job
          where (job ->> 'posts_per_source') ~ '^[0-9]+$'
            and (job ->> 'posts_per_source')::int > 100
      )
)
update research_messages m
set plan = capped.plan
from capped
where m.id = capped.id;

-- Re-run the first query; it should return no rows.
