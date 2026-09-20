"""HTTP request bodies."""

from typing import List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

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

    model_config = {
        "json_schema_extra": {
            "example": {"prompt": "Find modest fashion creators in Saudi Arabia"}
        }
    }

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
