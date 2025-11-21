# FastAPI + Yandex GPT agent

HTTP API поверх FastAPI для общения с Yandex GPT и интеграции с внешними чатами (веб, Telegram, ML-боты и т.д.). Фронтенд не включён — сервис предназначен для подключения к уже существующим чат-интерфейсам.

## Стэк
- FastAPI + Uvicorn
- HTTPX для вызова Yandex GPT API
- Docker/Docker Compose для деплоя

## Структура
```
app/
  agent.py        # Логика общения с Yandex GPT и реестр функций
  config.py       # Конфигурация через env-переменные
  main.py         # Точка входа FastAPI
  schemas.py      # Pydantic-схемы API
```

## Запуск
1. Скопируйте `.env.example` → `.env` и заполните `YANDEX_API_KEY`, `YANDEX_FOLDER_ID`, при необходимости модель и параметры генерации.
2. Запуск в Docker:
   - Через Docker Compose (если установлен плагин `docker compose` / `docker-compose`):
     ```bash
     # с новым плагином
     docker compose up --build
     # или со старым бинарём
     docker-compose up --build
     ```
   - Без Compose, обычным Docker:
     ```bash
     docker build -t ai-agent .
     docker run --rm -p 8000:8000 --env-file .env ai-agent
     ```
3. Отправляйте запросы на `POST /api/chat` из своего клиентского приложения/сайта.

### Локально (без Docker)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:create_app --factory --reload
```

## Расширение
- В `ToolRegistry` можно регистрировать функции, чтобы переиспользовать их в Telegram-ботах, ML-пайплайнах и других каналах.
- Для нового канала создайте адаптер, который обращается к `YandexGPTAgent.generate_reply` и транслирует ответы пользователям.
