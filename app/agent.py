import json
from datetime import datetime, timezone
import uuid
from typing import Dict, List, Optional

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
        # Simple in-memory map chat_id -> last agent response id (for previous_response_id chaining)
        self._agent_sessions: Dict[str, str] = {}

    async def close(self) -> None:
        await self._client.aclose()

    def _build_messages(self, chat_request: ChatRequest):
        if self.settings.yandex_system_prompt:
            yield {"role": "system", "text": self.settings.yandex_system_prompt}
        # Передаём актуальную дату/время (UTC) как часть системного контекста
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
        yield {"role": "system", "text": f"Current date/time (UTC): {now_utc}"}
        if getattr(chat_request, "user_profile", None):
            try:
                profile_text = json.dumps(chat_request.user_profile, ensure_ascii=False)
            except TypeError:
                profile_text = str(chat_request.user_profile)
            yield {"role": "system", "text": f"User profile: {profile_text}"}
        for message in chat_request.history:
            yield {"role": message.role, "text": message.content}
        yield {"role": "user", "text": chat_request.message}

    def _build_payload(self, chat_request: ChatRequest) -> Dict:
        return {
            "modelUri": self.settings.model_uri,
            "completionOptions": {
                "stream": self.settings.yandex_stream,
                "temperature": self.settings.yandex_temperature,
                "maxTokens": self.settings.yandex_max_tokens,
            },
            "messages": list(self._build_messages(chat_request)),
        }

    async def generate_reply(self, chat_request: ChatRequest) -> ChatResponse:
        if not self.settings.yandex_api_key or not self.settings.yandex_folder_id:
            raise RuntimeError("Yandex credentials are not configured.")

        headers = {
            "Authorization": f"Api-Key {self.settings.yandex_api_key}",
            "x-folder-id": self.settings.yandex_folder_id,
        }

        chat_id = chat_request.chat_id or str(uuid.uuid4())

        if self.settings.yandex_agent_id:
            prev_response_id = self._agent_sessions.get(chat_id)
            answer, response_id, ok = await self._answer_agent_prompt(
                chat_request, headers, prev_response_id
            )
            if ok and response_id:
                self._agent_sessions[chat_id] = response_id
            elif not ok:
                # Сброс цепочки и повтор без previous_response_id
                self._agent_sessions.pop(chat_id, None)
                answer, response_id, ok = await self._answer_agent_prompt(
                    chat_request, headers, None
                )
                if ok and response_id:
                    self._agent_sessions[chat_id] = response_id
                if not ok:
                    answer = "Что-то пошло не так, давай попробуем снова задать вопрос."
        elif self.settings.memory_enabled:
            answer = await self._answer_with_memory(chat_request, headers)
            if not answer or answer.startswith(
                (
                    "Ошибка ассистента",
                    "Не удалось распарсить",
                    "Не удалось получить ответ",
                )
            ):
                # Fallback to plain completion if assistant/memory call failed
                payload = self._build_payload(chat_request)
                answer = await self._completion_answer(headers, payload)
        else:
            payload = self._build_payload(chat_request)
            answer = await self._completion_answer(headers, payload)

        return ChatResponse(answer=answer, chat_id=chat_id, channel=chat_request.channel)

    @staticmethod
    def _extract_answer(payload: Dict) -> str:
        try:
            alternatives = payload["result"]["alternatives"]
            message = alternatives[0]["message"]
            return message.get("text") or ""
        except (KeyError, IndexError):
            return "Не удалось получить ответ от Yandex GPT."

    async def _streaming_answer(self, headers: Dict[str, str], payload: Dict) -> str:
        """Stream tokens from Yandex GPT when completionOptions.stream=True."""
        full_text: List[str] = []
        async with self._client.stream(
            "POST", self.base_url, headers=headers, json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                chunk_str = line
                if line.startswith("data:"):
                    chunk_str = line.removeprefix("data:").strip()
                if chunk_str in ("", "[DONE]"):
                    continue
                try:
                    chunk = json.loads(chunk_str)
                    text = self._extract_stream_text(chunk)
                    if text:
                        # API часто присылает полный текст на каждом чанке, поэтому берём дельту
                        accumulated = "".join(full_text)
                        if text.startswith(accumulated):
                            delta = text[len(accumulated) :]
                        else:
                            delta = text
                        if delta:
                            full_text.append(delta)
                except json.JSONDecodeError:
                    continue

        return "".join(full_text) if full_text else "Не удалось получить ответ от Yandex GPT."

    @staticmethod
    def _extract_stream_text(chunk: Dict) -> str:
        """Extract text from streaming chunk with different payload shapes."""
        try:
            alternatives = chunk["result"]["alternatives"]
            message = alternatives[0].get("message", {})
            # Streaming может присылать полный текст, либо содержимое по частям
            if message.get("text"):
                return message["text"]
            content_parts = message.get("content") or []
            texts = [p.get("text") for p in content_parts if isinstance(p, dict)]
            return "".join(filter(None, texts))
        except (KeyError, IndexError, TypeError):
            return ""

    async def _completion_answer(self, headers: Dict[str, str], payload: Dict) -> str:
        if self.settings.yandex_stream:
            return await self._streaming_answer(headers, payload)
        response = await self._client.post(self.base_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return self._extract_answer(data)

    def _build_memory_payload(self, chat_request: ChatRequest) -> Dict:
        payload = {
            "model": self.settings.model_uri,
            "input": chat_request.message,
            "temperature": self.settings.yandex_temperature,
            "max_output_tokens": self.settings.yandex_max_tokens,
            "instructions": (
                "Ты помощник, отвечающий на вопросы пользователя. "
                "Всегда сначала используй tool file_search с запросом пользователя, чтобы найти контекст в векторном индексе, "
                "а затем отвечай, опираясь на найденные данные. "
                "Если индекс не дал результатов, сообщи об этом явно и попробуй ответить кратко своими знаниями. "
                "Не отказывайся без попытки поиска."
            ),
            "tool_choice": "auto",
            "tools": [
                {
                    "file_search": {
                        "vector_store_ids": self.settings.vector_store_ids,
                    }
                }
            ],
        }
        # Передаём историю в формате messages, если есть
        if chat_request.history:
            payload["messages"] = list(self._build_messages(chat_request))
        return payload

    async def _answer_with_memory(
        self, chat_request: ChatRequest, headers: Dict[str, str]
    ) -> str:
        """Call Assistant API with file_search tool to use vector store memory."""
        payload = self._build_memory_payload(chat_request)
        response = await self._client.post(
            self.settings.yandex_assistant_url,
            headers=headers,
            json=payload,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            # Try to surface error body from Yandex API
            detail = exc.response.text
            return f"Ошибка ассистента: {detail or str(exc)}"
        except httpx.HTTPError as exc:
            return f"Ошибка ассистента (сетевое обращение): {exc}"

        data = response.json()
        # Surface error field if present
        if isinstance(data, dict) and data.get("error"):
            err = data["error"]
            if isinstance(err, dict):
                msg = err.get("message") or err.get("code") or str(err)
            else:
                msg = str(err)
            return f"Ошибка ассистента: {msg}"

        answer = self._extract_memory_answer(data)
        if answer.strip():
            return answer
        # Fallback: return raw response for debugging
        return f"Не удалось распарсить ответ ассистента: {data}"

    @staticmethod
    def _extract_memory_answer(payload: Dict) -> str:
        """Extract answer from Assistant API response."""
        if not isinstance(payload, dict):
            return "Не удалось получить ответ от Yandex GPT."

        # 1) OpenAI-compatible schema
        if payload.get("output_text"):
            return payload["output_text"]

        # 2) Assistant API nested result
        result = payload.get("result") or {}
        if isinstance(result, dict):
            if result.get("output_text"):
                return result["output_text"]
            # Alternatives-style
            try:
                alternatives = result["alternatives"]
                message = alternatives[0].get("message", {})
                if message.get("text"):
                    return message["text"]
                content_parts = message.get("content") or []
                texts = [p.get("text") for p in content_parts if isinstance(p, dict)]
                if texts:
                    return "".join(filter(None, texts))
            except (KeyError, IndexError, TypeError):
                pass
            # response.text style
            response_obj = result.get("response") or {}
            if isinstance(response_obj, dict) and response_obj.get("text"):
                return response_obj["text"]

        # 3) choices/message style
        if isinstance(payload.get("choices"), list):
            choice0 = payload["choices"][0]
            message = choice0.get("message") if isinstance(choice0, dict) else {}
            if isinstance(message, dict):
                if message.get("content"):
                    return message["content"]
                if message.get("text"):
                    return message["text"]

        # 4) output -> content (Assistant API schema)
        output = payload.get("output")
        if isinstance(output, list) and output:
            msg = output[0]
            if isinstance(msg, dict):
                content = msg.get("content") or []
                texts = [
                    c.get("text")
                    for c in content
                    if isinstance(c, dict) and c.get("text")
                ]
                if texts:
                    return "".join(texts)
                if msg.get("text"):
                    return msg["text"]

        # 4) response at root
        response_obj = payload.get("response") or {}
        if isinstance(response_obj, dict) and response_obj.get("text"):
            return response_obj["text"]

        return "Не удалось получить ответ от Yandex GPT."

    async def _answer_agent_prompt(
        self,
        chat_request: ChatRequest,
        headers: Dict[str, str],
        previous_response_id: Optional[str] = None,
    ) -> tuple[str, Optional[str], bool]:
        """Invoke a specific AI Studio agent by ID via Assistant API."""
        payload: Dict = {
            "prompt": {"id": self.settings.yandex_agent_id},
            "input": chat_request.message,
        }
        if previous_response_id:
            payload["previous_response_id"] = previous_response_id
        # Прокидываем векторные индексы агенту, если заданы
        tools: List[Dict] = []
        if self.settings.vector_store_ids:
            tools.append(
                {
                    "file_search": {
                        "vector_store_ids": self.settings.vector_store_ids,
                    }
                }
            )
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
            payload["parallel_tool_calls"] = True

        response = await self._client.post(
            self.settings.yandex_assistant_url, headers=headers, json=payload
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text
            return f"Ошибка агента: {detail or str(exc)}", None, False
        except httpx.HTTPError as exc:
            return f"Ошибка агента (сетевое обращение): {exc}", None, False

        data = response.json()
        if isinstance(data, dict) and data.get("error"):
            err = data["error"]
            if isinstance(err, dict):
                msg = err.get("message") or err.get("code") or str(err)
            else:
                msg = str(err)
            return f"Ошибка агента: {msg}", None, False

        # Try output_text root-level
        if isinstance(data, dict):
            if data.get("output_text"):
                return data["output_text"], data.get("id"), True
            output = data.get("output")
            if isinstance(output, list) and output:
                # Ищем первое message с контентом/текстом
                for msg in output:
                    if not isinstance(msg, dict):
                        continue
                    content = msg.get("content") or []
                    texts = [
                        c.get("text")
                        for c in content
                        if isinstance(c, dict) and c.get("text")
                    ]
                    if texts:
                        return "".join(texts), data.get("id"), True
                    if msg.get("text"):
                        return msg["text"], data.get("id"), True
                # fallback: если первый элемент содержит text
                msg = output[0]
                if isinstance(msg, dict) and msg.get("text"):
                    return msg["text"], data.get("id"), True
        return f"Ответ агента без текста: {data}", data.get("id"), False
