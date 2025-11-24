from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Literal, Optional

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    role: Role = "user"
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=100)
    chat_id: Optional[str] = None
    history: List[ChatMessage] = Field(default_factory=list)
    channel: str = Field(default="web")
    user_id: Optional[int] = None
    user_profile: Optional[dict] = None  # опционально передаваемый слепок профиля


class ChatResponse(BaseModel):
    answer: str
    chat_id: str
    channel: str = "web"


class ErrorResponse(BaseModel):
    detail: str


class ToolCall(BaseModel):
    name: str
    description: Optional[str] = None


@dataclass
class RegisteredTool:
    """Represents a callable that can be exposed to other channels."""

    name: str
    description: str
    handler: Callable[[dict], str]
