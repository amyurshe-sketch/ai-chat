from typing import Tuple
from pathlib import Path
from contextlib import asynccontextmanager
from collections import deque
import json
import time
from typing import Deque, Dict, List, Optional

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .agent import YandexGPTAgent
from .config import Settings, get_settings
from .schemas import ChatMessage, ChatRequest, ChatResponse, ErrorResponse
from .db import AIChat, AIMessage, db_session, init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    agent = YandexGPTAgent(settings=settings)
    db_sessionmaker = init_db(settings)
    limiter = RateLimiter(
        limit=settings.rate_limit_requests,
        window_sec=settings.rate_limit_window_sec,
    )
    app.state.agent = agent
    app.state.limiter = limiter
    app.state.db_sessionmaker = db_sessionmaker
    try:
        yield
    finally:
        await agent.close()


def get_agent(request: Request) -> YandexGPTAgent:
    return request.app.state.agent


def get_rate_limiter(request: Request) -> "RateLimiter":
    return request.app.state.limiter


def get_db_sessionmaker(request: Request):
    return getattr(request.app.state, "db_sessionmaker", None)


def require_agent_secret(request: Request, settings=get_settings()):
    secret = settings.ai_agent_secret
    if not secret:
        return
    incoming = request.headers.get("X-Agent-Secret")
    if incoming != secret:
        raise HTTPException(status_code=403, detail="Forbidden")

def require_internal(request: Request):
    host = request.client.host if request and request.client else ""
    if host not in {"127.0.0.1", "::1"}:
        raise HTTPException(status_code=403, detail="Forbidden")

def _rate_limit_message(request: Optional[Request]) -> str:
    lang = ""
    if request and request.headers:
        lang = (request.headers.get("Accept-Language") or "").lower()
    is_ru = lang.startswith("ru")
    return (
        "Слишком много запросов. Попробуйте позже."
        if is_ru
        else "Too many requests. Please try again later."
    )


def require_rate_limit(
    request: Request,
    limiter: "RateLimiter" = Depends(get_rate_limiter),
) -> None:
    client_ip = request.client.host if request.client else "unknown"
    allowed, retry_after = limiter.allow(client_ip)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=_rate_limit_message(request),
            headers={"Retry-After": str(int(retry_after))},
        )


def create_app() -> FastAPI:
    static_dir = Path(__file__).resolve().parent / "static"
    app = FastAPI(
        title="AI Chat",
        version="0.1.0",
        description="HTTP API for chatting with Yandex GPT and connecting external chats.",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def root(request: Request):
        require_internal(request)
        return HTMLResponse(status_code=403, content="Forbidden")

    @app.get("/health")
    async def healthcheck(request: Request):
        require_internal(request)
        return {"status": "ok"}

    @app.get("/healthz", include_in_schema=False)
    async def healthz(request: Request):
        require_internal(request)
        return {"status": "ok"}

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return FileResponse(static_dir / "favicon.svg", media_type="image/svg+xml")

    @app.post(
        "/api/chat",
        response_model=ChatResponse,
        responses={502: {"model": ErrorResponse}},
    )
    async def chat(
        chat_request: ChatRequest,
        _: None = Depends(require_agent_secret),
        agent: YandexGPTAgent = Depends(get_agent),
        request: Request = None,  # type: ignore[assignment]
        db_sessionmaker=Depends(get_db_sessionmaker),
        ) -> ChatResponse:
        # Enforce rate limit per user_id (if provided) else per IP
        limiter: RateLimiter = get_rate_limiter(request)
        client_ip = request.client.host if request and request.client else "unknown"
        key = None
        if chat_request.user_id is not None:
            key = f"user:{chat_request.user_id}"
        elif request:
            header_uid = request.headers.get("X-User-Id")
            if header_uid and header_uid.isdigit():
                key = f"user:{header_uid}"
        if key is None:
            key = f"ip:{client_ip}"
        allowed, retry_after = limiter.allow(key)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=_rate_limit_message(request),
                headers={"Retry-After": str(int(retry_after))},
            )
        # Подхватываем user_id / профиль из заголовков, если в теле нет
        user_id = chat_request.user_id
        user_profile = chat_request.user_profile
        if user_id is None and request:
            header_uid = request.headers.get("X-User-Id")
            if header_uid and header_uid.isdigit():
                user_id = int(header_uid)
        if user_profile is None and request:
            header_profile = request.headers.get("X-User-Profile")
            if header_profile:
                try:
                    user_profile = json.loads(header_profile)
                except json.JSONDecodeError:
                    user_profile = None

        # Если история не передана, попробуем взять из БД по chat_id
        effective_history = chat_request.history
        if not effective_history and chat_request.chat_id and db_sessionmaker:
            with db_session(db_sessionmaker) as session:
                rows = (
                    session.query(AIMessage)
                    .filter(AIMessage.chat_id == chat_request.chat_id)
                    .order_by(AIMessage.created_at.asc())
                    .all()
                )
                effective_history = [ChatMessage(role=m.role, content=m.content) for m in rows]
        prepared_request = chat_request.model_copy(
            update={
                "history": effective_history,
                "user_id": user_id,
                "user_profile": user_profile,
            }
        )
        try:
            response = await agent.generate_reply(prepared_request)
            # Сохраняем историю в БД, если доступна
            if db_sessionmaker:
                with db_session(db_sessionmaker) as session:
                    chat_row = session.get(AIChat, response.chat_id)
                    if not chat_row:
                        chat_row = AIChat(chat_id=response.chat_id, user_id=prepared_request.user_id)
                        session.add(chat_row)
                    snapshot = prepared_request.user_profile
                    session.add(
                        AIMessage(
                            chat_id=response.chat_id,
                            user_id=prepared_request.user_id,
                            role="user",
                            content=prepared_request.message,
                            profile_snapshot=snapshot,
                        )
                    )
                    session.add(
                        AIMessage(
                        chat_id=response.chat_id,
                        user_id=prepared_request.user_id,
                        role="assistant",
                        content=response.answer,
                        profile_snapshot=snapshot,
                        )
                    )
            return response
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=exc.response.status_code, detail=exc.response.text
            ) from exc
        except (httpx.TimeoutException, httpx.HTTPError) as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    # Совместимость со старым путём /api/ai-chat (фронт/бэкенд могут слать сюда)
    @app.post(
        "/api/ai-chat",
        response_model=ChatResponse,
        responses={502: {"model": ErrorResponse}},
        include_in_schema=False,
    )
    async def legacy_chat(
        chat_request: ChatRequest,
        _: None = Depends(require_agent_secret),
        agent: YandexGPTAgent = Depends(get_agent),
        request: Request = None,  # type: ignore[assignment]
    ) -> ChatResponse:
        # Reuse the same per-user/IP limiter logic
        limiter: RateLimiter = get_rate_limiter(request)
        client_ip = request.client.host if request and request.client else "unknown"
        key = None
        if chat_request.user_id is not None:
            key = f"user:{chat_request.user_id}"
        elif request:
            header_uid = request.headers.get("X-User-Id")
            if header_uid and header_uid.isdigit():
                key = f"user:{header_uid}"
        if key is None:
            key = f"ip:{client_ip}"
        allowed, retry_after = limiter.allow(key)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=_rate_limit_message(request),
                headers={"Retry-After": str(int(retry_after))},
            )
        try:
            return await agent.generate_reply(chat_request)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=exc.response.status_code, detail=exc.response.text
            ) from exc
        except (httpx.TimeoutException, httpx.HTTPError) as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    return app


class RateLimiter:
    """In-memory sliding window limiter per client."""

    def __init__(self, limit: int, window_sec: float):
        self.limit = max(1, limit)
        self.window = max(1.0, window_sec)
        self._hits: Dict[str, Deque[float]] = {}

    def allow(self, key: str) -> Tuple[bool, float]:
        now = time.time()
        bucket = self._hits.setdefault(key, deque())
        while bucket and now - bucket[0] > self.window:
            bucket.popleft()
        if len(bucket) >= self.limit:
            retry_after = max(1.0, self.window - (now - bucket[0]))
            return False, retry_after
        bucket.append(now)
        return True, 0.0
