import uuid
from typing import Dict, Iterable, List, Optional

import httpx

from .config import Settings
from .schemas import ChatRequest, ChatResponse, RegisteredTool


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
    base_url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

    def __init__(
        self,
        settings: Settings,
        http_client: Optional[httpx.AsyncClient] = None,
        tool_registry: Optional[ToolRegistry] = None,
    ) -> None:
        self.settings = settings
        self._client = http_client or httpx.AsyncClient(timeout=settings.request_timeout)
        self.tool_registry = tool_registry or ToolRegistry()

    async def close(self) -> None:
        await self._client.aclose()

    def _build_messages(self, chat_request: ChatRequest):
        for message in chat_request.history:
            yield {"role": message.role, "text": message.content}
        yield {"role": "user", "text": chat_request.message}

    def _build_payload(self, chat_request: ChatRequest) -> Dict:
        return {
            "modelUri": self.settings.model_uri,
            "completionOptions": {
                "stream": False,
                "temperature": self.settings.yandex_temperature,
                "maxTokens": self.settings.yandex_max_tokens,
            },
            "messages": list(self._build_messages(chat_request)),
        }

    async def generate_reply(self, chat_request: ChatRequest) -> ChatResponse:
        if not self.settings.yandex_api_key or not self.settings.yandex_folder_id:
            raise RuntimeError("Yandex credentials are not configured.")

        payload = self._build_payload(chat_request)
        headers = {
            "Authorization": f"Api-Key {self.settings.yandex_api_key}",
            "x-folder-id": self.settings.yandex_folder_id,
        }

        response = await self._client.post(self.base_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        answer = self._extract_answer(data)

        chat_id = chat_request.chat_id or str(uuid.uuid4())
        return ChatResponse(answer=answer, chat_id=chat_id, channel=chat_request.channel)

    @staticmethod
    def _extract_answer(payload: Dict) -> str:
        try:
            alternatives = payload["result"]["alternatives"]
            message = alternatives[0]["message"]
            return message.get("text") or ""
        except (KeyError, IndexError):
            return "Не удалось получить ответ от Yandex GPT."
