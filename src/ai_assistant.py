from __future__ import annotations

from typing import Optional

from .config import AppConfig

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover - optional dependency at runtime
    genai = None  # type: ignore


def generate_recommendations(prompt: str, config: AppConfig) -> Optional[str]:
    if not config.google_api_key or genai is None:
        return None
    try:
        genai.configure(api_key=config.google_api_key)
        model_name = config.default_model
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        text = getattr(response, "text", None)
        if text:
            return text
        # Fallback parsing
        parts = getattr(response, "candidates", None)
        if parts:
            return str(parts)
    except Exception:
        return None
    return None
