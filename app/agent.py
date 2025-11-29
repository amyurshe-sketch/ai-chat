import uuid
from typing import Dict, Iterable, List, Optional, Tuple

from openai import AsyncOpenAI
from openai._types import NOT_GIVEN

# Some packaged builds of openai lack the responses resource; import defensively.
try:
    from openai.resources.responses import Responses
except ImportError:
    Responses = None


class _FallbackResponses:
    """Minimal fallback to avoid startup failure when openai.responses is absent."""

    async def create(self, **kwargs):
        class _Result:
            def __init__(self, text: str):
                self.output_text = text

        return _Result(
            "AI agent backend missing 'responses' resource in openai client. "
            "Please reinstall openai>=1.55 with assistant support on the server."
        )

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
        # Fallback: ensure responses resource exists even if the client package lacks it
        if Responses:
            self._client.responses = getattr(self._client, "responses", Responses(self._client))
        else:
            # Attach a stub so the service still starts; real replies require proper client build
            self._client.responses = _FallbackResponses()
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

        instructions, input_messages = self._build_input(chat_request)
        # Ensure search answers carry sources
        source_hint = "If you use web search, include the source URL(s) in your answer."
        instructions = (
            f"{instructions}\n\n{source_hint}"
            if instructions
            else source_hint
        )

        tools = [{"type": "web_search"}]

        response = await self._client.responses.create(
            model=self.settings.model_uri,
            temperature=self.settings.yandex_temperature,
            max_output_tokens=self.settings.yandex_max_tokens,
            tools=tools,
            instructions=instructions if instructions else NOT_GIVEN,
            input=input_messages,
        )

        answer = response.output_text or "Ответ не получен."
        chat_id = chat_request.chat_id or str(uuid.uuid4())
        return ChatResponse(answer=answer, chat_id=chat_id, channel=chat_request.channel)
