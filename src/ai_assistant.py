from __future__ import annotations

from typing import Optional

from .config import AppConfig

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover - optional dependency at runtime
    genai = None  # type: ignore


def generate_recommendations(prompt: str, config: AppConfig) -> Optional[str]:
	if genai is None:
		return None
	# Resolve API key priorizando st.secrets
	api_key: Optional[str] = None
	try:  # pragma: no cover
		import streamlit as st  # type: ignore
		api_key = st.secrets.get("GOOGLE_API_KEY")  # type: ignore[attr-defined]
	except Exception:
		pass
	api_key = api_key or config.google_api_key
	if not api_key:
		return None
	try:
		genai.configure(api_key=api_key)
		model_name = config.default_model
		model = genai.GenerativeModel(model_name)
		response = model.generate_content(prompt)
		text = getattr(response, "text", None)
		if text:
			return text
		parts = getattr(response, "candidates", None)
		if parts:
			return str(parts)
	except Exception:
		return None
	return None
