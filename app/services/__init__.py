# The "services" package is where the actual work happens.
#
# Routes (in api/) receive the HTTP request and hand it to a service.
# Services contain the business logic — they don't know about HTTP,
# they just know how to do things:
#
#   prompt.py         → system prompt + OpenAI chat messages
#   validator.py      → plan quality gate
#   flows.py          → classify 3 operator flows and translate pipeline params
#   conversations.py  → Supabase / memory persistence for chat turns
#   audit.py          → latency / prompt / error log
#   agent.py          → sanitise → OpenAI chat.completions.create → validate → translate
