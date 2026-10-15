"""HTTP request bodies."""

from typing import List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator

from app.utils.guards import MAX_PROMPT_LENGTH

# Same ceiling as the current question. A stored plan, a review payload, or
# a long brief all need room. 20 turns × this size is the request-body
# ceiling; the planner only reads the last 8 anyway.
MAX_HISTORY_TURNS = 20
MAX_HISTORY_CONTENT = MAX_PROMPT_LENGTH


class ChatMessage(BaseModel):
    """One prior turn, same shape arkemgpt-api's frontend sends as conversation_history."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_HISTORY_CONTENT)


# A 3 MB file is about 4 MB once base64-encoded.
MAX_BRIEF_BASE64 = 4_100_000


class BriefUpload(BaseModel):
    """A content brief the operator uploaded. Read into a request, then the
    turn runs exactly as if they had typed it."""

    filename: str = Field(min_length=1, max_length=255)
    content_base64: str = Field(min_length=1, max_length=MAX_BRIEF_BASE64)


class AskRequest(BaseModel):
    prompt: str = Field(
        min_length=1,
        max_length=MAX_PROMPT_LENGTH,
        description="The operator's research question.",
    )
    conversation_id: Optional[UUID] = Field(
        default=None,
        description="Server-side thread id from a previous /ask response. Leave out on the first message.",
    )
    conversation_history: List[ChatMessage] = Field(
        default_factory=list,
        max_length=MAX_HISTORY_TURNS,
        description="Optional client-side history. Leave empty when using conversation_id.",
    )
    brief: Optional[BriefUpload] = Field(
        default=None,
        description=(
            "A content brief file (PDF, .docx, .txt, .md). The turn reads it into research "
            "tracks for approval and searches nothing. `prompt` is then an optional note on it."
        ),
    )
    track: Optional[str] = Field(
        default=None,
        max_length=200,
        description="The research-brief track this question runs, so the thread keeps its group.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {"prompt": "Find modest fashion creators in Saudi Arabia"}
        }
    }

    @model_validator(mode="before")
    @classmethod
    def a_brief_may_come_without_a_note(cls, data):
        # A brief on its own is a complete request; its note is optional. Without
        # a brief, `prompt` keeps exactly the rules it always had.
        if isinstance(data, dict) and data.get("brief") and not str(data.get("prompt") or "").strip():
            return {**data, "prompt": " "}
        return data

    @field_validator("conversation_id", mode="before")
    @classmethod
    def blank_id_is_none(cls, value):
        if value == "" or value is None:
            return None
        return value

    @field_validator("conversation_history", mode="before")
    @classmethod
    def drop_empty_history(cls, value):
        if not value:
            return []
        cleaned = []
        for item in value:
            content = item.get("content") if isinstance(item, dict) else getattr(item, "content", "")
            if content and str(content).strip() and str(content).strip() != "string":
                cleaned.append(item)
        return cleaned
