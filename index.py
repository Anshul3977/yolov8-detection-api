"""Vercel entrypoint (zero-config FastAPI): the same ASGI app that Docker / uvicorn runs."""
from app.main import app  # noqa: F401
