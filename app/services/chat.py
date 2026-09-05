from __future__ import annotations

import json
import uuid

from sqlalchemy.orm import Session

from app.agents.factory import create_agent_runtime
from app.core.config import Settings
from app.core.enums import MessageRole
from app.models.entities import ChatMessage, ChatSession, PsychologicalReport, UserAccount
from app.schemas.dtos import ChatRequest, ChatStreamEvent
from app.services.ai import AiClient
from app.services.memory import RedisShortTermMemoryStore
from app.services.mcp_client import McpToolError, MindBridgeMcpToolClient
from app.services.privacy import PrivacySanitizer
from app.services.tool_queue import ToolQueueService


class ChatService:
    def __init__(self, db: Session, settings: Settings):
        self.db = db
        self.settings = settings
        self.privacy = PrivacySanitizer()
        self.memory = RedisShortTermMemoryStore(settings)
        self.ai = AiClient(settings)

    async def stream_chat(self, user: UserAccount, request: ChatRequest):
        prepared = self.prepare(user, request)
        yield sse("meta", ChatStreamEvent(type="meta", sessionId=prepared["session"].public_id).model_dump(by_alias=True))
        assistant = []
        async for token in self.ai.stream(prepared["messages"]):
            assistant.append(token)
            yield sse("token", ChatStreamEvent(type="token", sessionId=prepared["session"].public_id, content=token).model_dump())
        if assistant:
            self.save_message(user, prepared["session"], MessageRole.ASSISTANT, "".join(assistant))
        if prepared["report_id"] is not None:
            if self.settings.tool_queue_enabled:
                ToolQueueService(self.db, self.settings).enqueue_report(prepared["report_id"], prepared["risk_level"])
            else:
                try:
                    await MindBridgeMcpToolClient(self.settings).handle_report(prepared["report_id"], prepared["risk_level"])
                except McpToolError as exc:
                    yield sse(
                        "error",
                        ChatStreamEvent(
                            type="error",
                            sessionId=prepared["session"].public_id,
                            message=f"MCP 工具调用失败：{exc}",
                        ).model_dump(),
                    )
                    return
        yield sse("done", ChatStreamEvent(type="done", sessionId=prepared["session"].public_id).model_dump())

    def prepare(self, user: UserAccount, request: ChatRequest) -> dict:
        text = request.message.strip()
        model_input = self.privacy.sanitize(text)
        session = self.resolve_session(user, request.sessionId, text)
        agent_run = create_agent_runtime(self.db, self.settings).run(user, session, text, model_input)
        self.save_message(user, session, MessageRole.USER, text)
        report_id = None
        if agent_run.requires_report and agent_run.assessment is not None:
            report = PsychologicalReport(
                user_id=user.id,
                session_id=session.id,
                content=text,
                intent=agent_run.intent.value,
                emotion=agent_run.assessment.emotion.value,
                emotion_score=agent_run.assessment.emotion_score,
                risk_level=agent_run.assessment.risk.value,
                confidence=agent_run.assessment.confidence,
                summary=agent_run.assessment.summary,
            )
            self.db.add(report)
            self.db.commit()
            report_id = report.id
            risk_level = report.risk_level
        else:
            risk_level = None
        return {
            "session": session,
            "messages": agent_run.response_messages,
            "report_id": report_id,
            "risk_level": risk_level,
        }

    def resolve_session(self, user: UserAccount, public_id: str | None, text: str) -> ChatSession:
        if public_id:
            session = self.db.query(ChatSession).filter(ChatSession.public_id == public_id, ChatSession.user_id == user.id).first()
            if session is None:
                raise ValueError("Session not found")
            return session
        session = ChatSession(public_id=uuid.uuid4().hex, user_id=user.id, title=text[:36])
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        return session

    def save_message(self, user: UserAccount, session: ChatSession, role: MessageRole, content: str) -> None:
        self.db.add(ChatMessage(user_id=user.id, session_id=session.id, role=role.value, content=content))
        session.touch()
        self.db.add(session)
        self.db.commit()
        self.memory.append(session.public_id, role.value, content)


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"
