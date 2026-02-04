"""
Entry point cho Render.com - uvicorn api.main:app

Re-export FastAPI app từ api.app để Render có thể chạy:
  uvicorn api.main:app --host 0.0.0.0 --port $PORT
"""
from .app import app

__all__ = ["app"]
