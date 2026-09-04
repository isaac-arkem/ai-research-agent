# Conversation store — persists chat turns for follow-ups.
# Uses Supabase when configured; otherwise an in-memory fallback so local
# chat still works without a database.

import logging
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import uuid4

from app.core.supabase import get_supabase_admin
from app.models.domain import ChatTurn, ResearchPlan
from app.models.responses import (
    ConversationMessage,
    ConversationResponse,
    ConversationSummary,
)

logger = logging.getLogger(__name__)

CONVERSATIONS_TABLE = "research_conversations"
MESSAGES_TABLE = "research_messages"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _summary(row: dict) -> ConversationSummary:
    return ConversationSummary(
        id=row["id"],
        title=row.get("title"),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at") or row.get("created_at"),
    )


class ConversationStore:
    def create(self, user_id: str, title: str) -> str:
        raise NotImplementedError

    def list(self, user_id: str, limit: int = 50) -> List[ConversationSummary]:
        raise NotImplementedError

    def get(self, conversation_id: str, user_id: str) -> Optional[ConversationResponse]:
        raise NotImplementedError

    def history(self, conversation_id: str, user_id: str) -> List[ChatTurn]:
        conv = self.get(conversation_id, user_id)
        if conv is None:
            return []
        return [ChatTurn(role=m.role, content=m.content) for m in conv.messages]

    def add_user_message(self, conversation_id: str, user_id: str, content: str) -> str:
        raise NotImplementedError

    def add_assistant_message(
        self,
        conversation_id: str,
        user_id: str,
        content: str,
        *,
        flow: Optional[str] = None,
        plan: Optional[ResearchPlan] = None,
    ) -> str:
        raise NotImplementedError


class MemoryConversationStore(ConversationStore):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._conversations: Dict[str, dict] = {}
        self._messages: Dict[str, List[dict]] = {}

    def create(self, user_id: str, title: str) -> str:
        cid = str(uuid4())
        now = _now()
        with self._lock:
            self._conversations[cid] = {
                "id": cid,
                "user_id": user_id,
                "title": title[:120],
                "created_at": now,
                "updated_at": now,
            }
            self._messages[cid] = []
        return cid

    def list(self, user_id: str, limit: int = 50) -> List[ConversationSummary]:
        with self._lock:
            rows = [
                _summary(row)
                for row in self._conversations.values()
                if row["user_id"] == user_id
            ]
        rows.sort(key=lambda s: s.updated_at or s.created_at or "", reverse=True)
        return rows[: max(limit, 0)]

    def get(self, conversation_id: str, user_id: str) -> Optional[ConversationResponse]:
        with self._lock:
            row = self._conversations.get(conversation_id)
            if row is None or row["user_id"] != user_id:
                return None
            messages = [
                ConversationMessage(
                    id=m["id"],
                    role=m["role"],
                    content=m["content"],
                    flow=m.get("flow"),
                    plan=m.get("plan"),
                    created_at=m.get("created_at"),
                )
                for m in self._messages.get(conversation_id, [])
            ]
        return ConversationResponse(
            id=row["id"],
            title=row.get("title"),
            messages=messages,
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
        )

    def add_user_message(self, conversation_id: str, user_id: str, content: str) -> str:
        mid = str(uuid4())
        now = _now()
        with self._lock:
            row = self._conversations.get(conversation_id)
            if row is None or row["user_id"] != user_id:
                raise KeyError("conversation not found")
            row["updated_at"] = now
            self._messages[conversation_id].append(
                {
                    "id": mid,
                    "role": "user",
                    "content": content,
                    "created_at": now,
                }
            )
        return mid

    def add_assistant_message(
        self,
        conversation_id: str,
        user_id: str,
        content: str,
        *,
        flow: Optional[str] = None,
        plan: Optional[ResearchPlan] = None,
    ) -> str:
        mid = str(uuid4())
        now = _now()
        with self._lock:
            row = self._conversations.get(conversation_id)
            if row is None or row["user_id"] != user_id:
                raise KeyError("conversation not found")
            row["updated_at"] = now
            self._messages[conversation_id].append(
                {
                    "id": mid,
                    "role": "assistant",
                    "content": content,
                    "flow": flow,
                    "plan": plan,
                    "created_at": now,
                }
            )
        return mid


class SupabaseConversationStore(ConversationStore):
    def __init__(self, fallback: ConversationStore) -> None:
        self._fallback = fallback

    def _client(self):
        return get_supabase_admin()

    def _touch(self, conversation_id: str) -> None:
        client = self._client()
        if client is None:
            return
        try:
            (
                client.table(CONVERSATIONS_TABLE)
                .update({"updated_at": _now()})
                .eq("id", conversation_id)
                .execute()
            )
        except Exception as exc:
            logger.debug("conversation touch failed: %s", exc)

    def create(self, user_id: str, title: str) -> str:
        client = self._client()
        if client is None:
            return self._fallback.create(user_id, title)
        try:
            result = (
                client.table(CONVERSATIONS_TABLE)
                .insert({"user_id": user_id, "title": title[:120]})
                .execute()
            )
            return result.data[0]["id"]
        except Exception as exc:
            logger.warning("conversation create failed (%s) — using memory store", exc)
            return self._fallback.create(user_id, title)

    def list(self, user_id: str, limit: int = 50) -> List[ConversationSummary]:
        items: Dict[str, ConversationSummary] = {
            item.id: item for item in self._fallback.list(user_id, limit=limit)
        }
        client = self._client()
        if client is not None:
            try:
                result = (
                    client.table(CONVERSATIONS_TABLE)
                    .select("id, title, created_at, updated_at")
                    .eq("user_id", user_id)
                    .order("updated_at", desc=True)
                    .limit(limit)
                    .execute()
                )
                for row in result.data or []:
                    items[row["id"]] = _summary(row)
            except Exception as exc:
                logger.warning("conversation list failed: %s", exc)
        rows = list(items.values())
        rows.sort(key=lambda s: s.updated_at or s.created_at or "", reverse=True)
        return rows[: max(limit, 0)]

    def get(self, conversation_id: str, user_id: str) -> Optional[ConversationResponse]:
        mem = self._fallback.get(conversation_id, user_id)
        if mem is not None:
            return mem
        client = self._client()
        if client is None:
            return None
        try:
            conv = (
                client.table(CONVERSATIONS_TABLE)
                .select("id, title, user_id, created_at, updated_at")
                .eq("id", conversation_id)
                .eq("user_id", user_id)
                .limit(1)
                .execute()
            )
            if not conv.data:
                return None
            msgs = (
                client.table(MESSAGES_TABLE)
                .select("id, role, content, flow, plan, created_at")
                .eq("conversation_id", conversation_id)
                .order("created_at")
                .execute()
            )
            messages = []
            for m in msgs.data or []:
                plan = m.get("plan")
                messages.append(
                    ConversationMessage(
                        id=m["id"],
                        role=m["role"],
                        content=m["content"] or "",
                        flow=m.get("flow"),
                        plan=ResearchPlan(**plan) if plan else None,
                        created_at=m.get("created_at"),
                    )
                )
            row = conv.data[0]
            return ConversationResponse(
                id=row["id"],
                title=row.get("title"),
                messages=messages,
                created_at=row.get("created_at"),
                updated_at=row.get("updated_at"),
            )
        except Exception as exc:
            logger.warning("conversation get failed: %s", exc)
            return None

    def add_user_message(self, conversation_id: str, user_id: str, content: str) -> str:
        if self._fallback.get(conversation_id, user_id) is not None:
            return self._fallback.add_user_message(conversation_id, user_id, content)
        client = self._client()
        if client is None:
            return self._fallback.add_user_message(conversation_id, user_id, content)
        try:
            result = (
                client.table(MESSAGES_TABLE)
                .insert(
                    {
                        "conversation_id": conversation_id,
                        "role": "user",
                        "content": content,
                    }
                )
                .execute()
            )
            self._touch(conversation_id)
            return result.data[0]["id"]
        except Exception as exc:
            logger.warning("user message insert failed: %s", exc)
            raise

    def add_assistant_message(
        self,
        conversation_id: str,
        user_id: str,
        content: str,
        *,
        flow: Optional[str] = None,
        plan: Optional[ResearchPlan] = None,
    ) -> str:
        if self._fallback.get(conversation_id, user_id) is not None:
            return self._fallback.add_assistant_message(
                conversation_id,
                user_id,
                content,
                flow=flow,
                plan=plan,
            )
        client = self._client()
        if client is None:
            return self._fallback.add_assistant_message(
                conversation_id,
                user_id,
                content,
                flow=flow,
                plan=plan,
            )
        try:
            result = (
                client.table(MESSAGES_TABLE)
                .insert(
                    {
                        "conversation_id": conversation_id,
                        "role": "assistant",
                        "content": content,
                        "flow": flow,
                        "plan": plan.model_dump() if plan else None,
                    }
                )
                .execute()
            )
            self._touch(conversation_id)
            return result.data[0]["id"]
        except Exception as exc:
            logger.warning("assistant message insert failed: %s", exc)
            raise


_memory = MemoryConversationStore()
store: ConversationStore = SupabaseConversationStore(_memory)
