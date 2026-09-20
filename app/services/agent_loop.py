"""The turn, decided by the model instead of by a fixed sequence.

gather_web_context walks a path someone wrote down: route, then platform,
then basis, then plan, then lanes, then extract, then answer. Every question
that did not fit that path needed a new branch, a new field on the router's
form, or a new word in a list — and the model, which had usually understood
the question perfectly, had nowhere to put what it knew.

Here it calls tools until it has an answer. The tools are the same machinery
the pipeline uses; what changes is who decides the order.

Returns the same WebContext the pipeline returns, so agent.py, the endpoint
and the console are untouched, and the whole thing is one flag away from
being switched off.
"""

from __future__ import annotations

import json
import logging
from typing import Optional, Sequence

from openai import OpenAI

from app.models.domain import AgentContext, ChatTurn
from app.services.tools import TERMINAL, TOOL_SCHEMAS, ToolContext, run_tool

logger = logging.getLogger(__name__)

SYSTEM = """You are the research assistant for a social-listening tool. Operators use you to find creators, hashtags and markets worth scraping, and to interrogate what you find before they commit to a plan.

You have tools. Use them rather than answering from memory: your knowledge of who is popular, what their handle is, or how many followers they have is out of date and was never reliable.

WHAT THE OPERATOR IS ACTUALLY DOING. Some turns ask for a LIST — creators, hashtags, countries — and that list is what they build a scrape plan from. Other turns interrogate a list you already gave them: how recent is this, why is that account here, which of these are really Ghanaian, is this enough to plan with. Those want an ANSWER, not another list. Read which one you are being asked for.

NEVER INVENT WHAT A TOOL WOULD HAVE TOLD YOU.
- A handle comes from extract_creators or search_social. Never write one into your reply.
- A follower count comes from a tool that returned it. If nothing did, say so.
- If the answer needs something nobody gathered, say that plainly and say what would get it. "Nothing I have records where these people are based" is a good answer. A confident guess is not.

SEARCH WITH THE OPERATOR'S OWN WORDS. Do not turn their question into keywords. "popular music artists in Ghana" searched as "top Instagram influencers Ghana" comes back full of influencers. Their phrasing carries intent a paraphrase drops.

LOOK AT AN ACCOUNT BEFORE LOOKING FOR PEOPLE LIKE IT. When the operator names accounts as a yardstick — "creators like @janedoe", "similar to @x but not @y" — read those accounts first with profile. A hashtag sweep returns whoever posted under a tag; it cannot tell you what @janedoe is like, so searching before you have looked is guessing with their money.

IF YOUR REPLY WOULD CONTAIN A QUESTION, CALL ask_operator INSTEAD. Do not call answer with a question inside it: the operator gets a list they did not ask for with a question sitting on top of it.

ASK WHEN THE ANSWER IS THEIRS TO GIVE, not when you could find out. "Similar in which way — same style, same size, same scene?" is worth a turn, because each gives a different list and only they know which they meant. "Which platform?" is worth asking when nothing can be searched without it. Do not ask for things you can look up.

SCRAPING COSTS MONEY. search_social is refused unless the operator named that platform. If it refuses, do not try another platform — ask them.

WHEN YOU ARE DONE, call answer. The lists you extracted are attached to your reply automatically, so do not repeat them: say what the evidence adds up to, what stands out, and what is thin or missing. Then offer one real next step built from THIS result, not a generic question.

If you cannot do what was asked, answer anyway and say why. An honest refusal naming what is missing is more useful than a list nobody asked for."""


def _recent(history: Optional[Sequence[ChatTurn]], keep: int = 6) -> list:
    out = []
    for turn in list(history or [])[-keep:]:
        role = getattr(turn, "role", None) or (turn.get("role") if isinstance(turn, dict) else None)
        content = getattr(turn, "content", None) or (
            turn.get("content") if isinstance(turn, dict) else None
        )
        if role in ("user", "assistant") and content:
            out.append({"role": role, "content": str(content)[:4000]})
    return out


def run_turn(
    prompt: str,
    ctx: AgentContext,
    history: Optional[Sequence[ChatTurn]] = None,
    *,
    settings,
    on_progress=None,
):
    """Let the model work the turn with tools. Returns a WebContext.

    None is never returned: every failure ends as a WebContext the caller
    already knows how to render, because falling back to the pipeline
    mid-turn would spend everything twice.
    """
    from app.services.grounding import WebContext, _trim

    emit = on_progress or (lambda *a, **k: None)
    tc = ToolContext(
        settings=settings, ctx=ctx, prompt=prompt, history=history, on_progress=emit
    )
    max_steps = int(getattr(settings, "tools_max_steps", 8))
    model = getattr(settings, "research_plan_model", "gpt-4o")

    messages = (
        [{"role": "system", "content": SYSTEM}]
        + _recent(history)
        # Fenced for the same reason the router fences it: this is the
        # operator's text, and it is data rather than instruction.
        + [{"role": "user", "content": f'The operator said:\n"""\n{prompt}\n"""'}]
    )

    client = OpenAI(api_key=settings.openai_api_key,
                    timeout=max(float(getattr(settings, "search_timeout", 15.0)), 60.0))
    emit("thinking", detail="Working out what this needs")

    for step in range(max_steps):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, tools=TOOL_SCHEMAS, temperature=0.2,
            )
        except Exception as exc:
            logger.warning("agent loop: model call failed at step %d: %s", step, exc)
            return _finish(tc, None, None, WebContext, _trim,
                           reason=f"model call failed: {type(exc).__name__}")

        choice = response.choices[0].message
        calls = getattr(choice, "tool_calls", None)
        if not calls:
            # It replied in prose without calling answer. Treat the prose as
            # the answer rather than burning a step correcting it.
            return _finish(tc, (choice.content or "").strip() or None, None,
                           WebContext, _trim)

        messages.append(choice)
        for call in calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except ValueError:
                args = {}

            if name in TERMINAL:
                if name == "answer":
                    return _finish(tc, args.get("reply"), args.get("next_step"),
                                   WebContext, _trim)
                return _ask(tc, args, WebContext)

            emit("searching" if name.startswith("search") else "reading",
                 detail=_say(name, args))
            result = run_tool(name, args, tc)
            logger.info("agent loop: %s(%s) -> %s", name,
                        json.dumps(args)[:120], result[:160].replace("\n", " "))
            messages.append({
                "role": "tool", "tool_call_id": call.id, "content": result,
            })

    logger.info("agent loop: hit the %d-step cap — answering with what it has", max_steps)
    return _finish(tc, None, None, WebContext, _trim, reason="step cap reached")


def _say(name: str, args: dict) -> str:
    if name == "search_web":
        return f"Searching the web for {args.get('query', '')!r}"
    if name == "search_social":
        return f"Searching {args.get('platform', '')}"
    if name == "resolve_account":
        return f"Working out who {args.get('name', '')} is"
    if name == "profile":
        return f"Reading @{args.get('handle', '')}"
    if name == "read_source":
        return "Reading a source"
    return "Reading what came back"


def _ask(tc: ToolContext, args: dict, WebContext):
    """The model stopped to ask. Options become the choices on screen."""
    from app.models.domain import ComparisonBasis

    options = [str(o).strip() for o in (args.get("options") or []) if str(o).strip()]
    return WebContext(
        action="ask",
        question=str(args.get("question") or "").strip() or "Could you tell me a little more?",
        missing=["answer"],
        # One option is the answer, not a question — the same rule the
        # pipeline uses for the basis choice.
        comparison_bases=[ComparisonBasis(label=o) for o in options] if len(options) > 1 else [],
        provider="tools",
    )


def _finish(tc: ToolContext, reply, next_step, WebContext, _trim, reason: str = ""):
    """Whatever was gathered, in the shape the caller already renders.

    A turn that gathered nothing is a "respond": prose with no sources under
    it. A turn that searched is a review turn, so the sources show.
    """
    if tc.creators:
        from app.services.grounding import drop_irrelevant_creators

        tc.creators = drop_irrelevant_creators(
            tc.creators, tc.prompt,
            openai_key=tc.settings.openai_api_key,
            model=tc.settings.grounding_model,
            timeout=float(getattr(tc.settings, "search_timeout", 15.0)),
        )
    gathered = bool(tc.findings or tc.creators or tc.markets)
    if not reply:
        reply = ("I could not work that out." if not gathered
                 else "Here is what came back.")
        if reason:
            logger.info("agent loop: finishing without a reply (%s)", reason)
    return WebContext(
        action="search" if gathered else "respond",
        query=tc.prompt,
        findings=[_trim(f) for f in tc.findings],
        creators=tc.creators,
        hashtags=tc.hashtags,
        markets=tc.markets,
        prose=reply,
        next_step=next_step,
        provider="tools:" + (",".join(f"{k}={v}" for k, v in sorted(tc.lane_status.items())) or "none"),
    )
