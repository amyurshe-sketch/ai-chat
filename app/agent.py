import uuid
from typing import Dict, Iterable, List, Optional, Tuple

from openai import AsyncOpenAI
from openai._types import NOT_GIVEN

from .config import Settings
from .schemas import ChatMessage, ChatRequest, ChatResponse, RegisteredTool


class ToolRegistry:
    """Collects callable tools so future channels (e.g., Telegram) can reuse them."""

    def __init__(self) -> None:
        self._tools: Dict[str, RegisteredTool] = {}

    def register(self, tool: RegisteredTool) -> None:
        self._tools[tool.name] = tool

    def register_callable(self, name: str, description: str, handler) -> None:
        self.register(RegisteredTool(name=name, description=description, handler=handler))

    def get(self, name: str) -> Optional[RegisteredTool]:
        return self._tools.get(name)

    def all(self) -> List[RegisteredTool]:
        return list(self._tools.values())


class YandexGPTAgent:
    def __init__(
        self,
        settings: Settings,
        tool_registry: Optional[ToolRegistry] = None,
    ) -> None:
        self.settings = settings
        self._client = AsyncOpenAI(
            api_key=settings.yandex_api_key,
            base_url="https://rest-assistant.api.cloud.yandex.net/v1",
            project=settings.yandex_folder_id,
            timeout=settings.request_timeout,
        )
        self.tool_registry = tool_registry or ToolRegistry()

    async def close(self) -> None:
        await self._client.close()

    @staticmethod
    def _split_instructions(messages: List[ChatMessage]) -> Tuple[Optional[str], List[ChatMessage]]:
        """Collect system messages into instructions and return remaining history."""
        system_parts = [m.content for m in messages if m.role == "system"]
        non_system = [m for m in messages if m.role != "system"]
        instructions = "\n\n".join(system_parts) if system_parts else None
        return instructions, non_system

    def _build_input(self, chat_request: ChatRequest):
        instructions, history = self._split_instructions(chat_request.history)
        messages = history + [ChatMessage(role="user", content=chat_request.message)]
        input_messages = [
            {"role": m.role, "content": [{"type": "input_text", "text": m.content}]}
            for m in messages
        ]
        return instructions, input_messages

    async def generate_reply(self, chat_request: ChatRequest) -> ChatResponse:
        if not self.settings.yandex_api_key or not self.settings.yandex_folder_id:
            raise RuntimeError("Yandex credentials are not configured.")

        if not hasattr(self._client, "responses"):
            raise RuntimeError(
                "OpenAI client is too old for Yandex assistant API. "
                "Update dependency: pip install -U 'openai>=1.55'"
            )

        instructions, input_messages = self._build_input(chat_request)

        response = await self._client.responses.create(
            model=self.settings.model_uri,
            temperature=self.settings.yandex_temperature,
            max_output_tokens=self.settings.yandex_max_tokens,
            tools=[{"type": "web_search"}],
            instructions=instructions if instructions else NOT_GIVEN,
            input=input_messages,
        )

        answer = response.output_text or "Ответ не получен."
        chat_id = chat_request.chat_id or str(uuid.uuid4())
        return ChatResponse(answer=answer, chat_id=chat_id, channel=chat_request.channel)
