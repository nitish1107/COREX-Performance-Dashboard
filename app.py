"""ASGI entry point used by Vercel (and by `uvicorn app:app`). The UI lives in dashboard.py."""
from pathlib import Path

from streamlit.starlette import App

app = App(Path(__file__).parent / "dashboard.py")
