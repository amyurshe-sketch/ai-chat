from pathlib import Path
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .agent import YandexGPTAgent
from .config import get_settings
from .schemas import ChatRequest, ChatResponse, ErrorResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    agent = YandexGPTAgent(settings=settings)
    app.state.agent = agent
    try:
        yield
    finally:
        await agent.close()


def get_agent(request: Request) -> YandexGPTAgent:
    return request.app.state.agent


def create_app() -> FastAPI:
    static_dir = Path(__file__).resolve().parent / "static"
    app = FastAPI(
        title="AI Chat",
        version="0.1.0",
        description="HTTP API для общения с Yandex GPT и подключения внешних чатов.",
        lifespan=lifespan,
    )
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def root():
        return """
        <!doctype html>
        <html lang="ru">
          <head>
            <meta charset="UTF-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0" />
            <title>AI Chat</title>
            <link rel="icon" type="image/svg+xml" href="/favicon.ico" />
          </head>
          <body style="font-family: Arial, sans-serif; background:#0b1220; color:#f8fafc; margin:0; display:flex; align-items:center; justify-content:center; min-height:100vh;">
            <div style="text-align:center;">
              <div style="display:inline-flex; align-items:center; justify-content:center; width:72px; height:72px; border-radius:16px; background:linear-gradient(120deg,#0b1220,#111a2f); border:1px solid rgba(148,163,184,0.2); margin-bottom:14px;">
                <span style="font-size:34px; font-weight:800; letter-spacing:0.02em;">AI</span>
              </div>
              <h1 style="margin:0 0 8px; font-size:28px;">AI Chat</h1>
              <p style="margin:0; color:rgba(248,250,252,0.8);">Сервис запущен · /api/chat</p>
            </div>
          </body>
        </html>
        """

    @app.get("/health")
    async def healthcheck():
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
        agent: YandexGPTAgent = Depends(get_agent),
    ) -> ChatResponse:
        try:
            return await agent.generate_reply(chat_request)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=exc.response.status_code, detail=exc.response.text
            ) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    return app
