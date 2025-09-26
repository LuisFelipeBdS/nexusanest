from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

try:
	import google.generativeai as genai  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
	genai = None  # type: ignore


# -----------------------------
# Constantes do Modelo / Tokens
# -----------------------------
DEFAULT_MODEL_NAME = "gemini-2.5-pro"
MAX_INPUT_TOKENS = 32768
MAX_OUTPUT_TOKENS = 2048
TEMPERATURE = 0.3
TOP_P = 0.95
TOP_K = 40

# Timeouts e Retry (segundos)
DEFAULT_TIMEOUT_SECONDS = 30
RETRY_MAX_ATTEMPTS = 3
RETRY_BACKOFF_INITIAL = 1.0
RETRY_BACKOFF_MAX = 8.0


def _get_logger() -> logging.Logger:
	logger = logging.getLogger("helpanest.ai")
	if not logger.handlers:
		handler = logging.StreamHandler()
		formatter = logging.Formatter(
			"[%(asctime)s] %(levelname)s %(name)s - %(message)s",
			datefmt="%Y-%m-%d %H:%M:%S",
		)
		handler.setFormatter(formatter)
		logger.addHandler(handler)
		logger.setLevel(logging.INFO)
	return logger


logger = _get_logger()


class AppConfig(BaseModel):
	app_name: str = Field(default="HelpAnest - Plataforma de Risco Perioperatório")
	google_api_key: Optional[str] = Field(default=None)
	default_model: str = Field(default=DEFAULT_MODEL_NAME)
	reports_dir: str = Field(default="reports")

	# Geração / Modelo
	max_input_tokens: int = Field(default=MAX_INPUT_TOKENS)
	max_output_tokens: int = Field(default=MAX_OUTPUT_TOKENS)
	temperature: float = Field(default=TEMPERATURE)
	top_p: float = Field(default=TOP_P)
	top_k: int = Field(default=TOP_K)

	# Timeout e retry
	timeout_seconds: int = Field(default=DEFAULT_TIMEOUT_SECONDS)
	retry_max_attempts: int = Field(default=RETRY_MAX_ATTEMPTS)
	retry_backoff_initial: float = Field(default=RETRY_BACKOFF_INITIAL)
	retry_backoff_max: float = Field(default=RETRY_BACKOFF_MAX)

	@property
	def reports_path(self) -> Path:
		return Path(self.reports_dir)


def load_env_api_key(env_file: str | None = ".env") -> Optional[str]:
	"""Carrega a GOOGLE_API_KEY do .env, se existir."""
	if env_file and Path(env_file).exists():
		load_dotenv(env_file)
	return os.getenv("GOOGLE_API_KEY")


def load_config(env_file: str | None = ".env") -> AppConfig:
	if env_file and Path(env_file).exists():
		load_dotenv(env_file)
	return AppConfig(
		app_name=os.getenv("APP_NAME", "HelpAnest - Plataforma de Risco Perioperatório"),
		google_api_key=os.getenv("GOOGLE_API_KEY"),
		default_model=os.getenv("DEFAULT_MODEL", DEFAULT_MODEL_NAME),
		reports_dir=os.getenv("REPORTS_DIR", "reports"),
	)


def _build_generation_config(cfg: AppConfig) -> dict[str, Any]:
	return {
		"temperature": cfg.temperature,
		"top_p": cfg.top_p,
		"top_k": cfg.top_k,
		"max_output_tokens": cfg.max_output_tokens,
	}


def create_gemini_model(cfg: AppConfig):
	"""Cria e retorna o modelo Gemini configurado. Retorna None se indisponível."""
	if genai is None or not (cfg.google_api_key or os.getenv("GOOGLE_API_KEY")):
		logger.warning("Gemini SDK não disponível ou GOOGLE_API_KEY ausente.")
		return None
	try:
		api_key = cfg.google_api_key or os.getenv("GOOGLE_API_KEY")
		genai.configure(api_key=api_key)
		model = genai.GenerativeModel(cfg.default_model)
		return model
	except Exception as exc:
		logger.error("Falha ao criar modelo Gemini: %s", exc)
		return None


def test_gemini_connection(cfg: AppConfig) -> bool:
	"""Realiza uma chamada simples ao Gemini para testar conectividade.
	Inclui retry exponencial simples e timeout."""
	model = create_gemini_model(cfg)
	if model is None:
		return False

	attempt = 0
	while attempt < cfg.retry_max_attempts:
		attempt += 1
		try:
			logger.info("Teste de conexão ao Gemini (tentativa %s)", attempt)
			resp = model.generate_content(
				"ping",
				generation_config=_build_generation_config(cfg),
				request_options={"timeout": cfg.timeout_seconds},
			)
			# Alguns SDKs retornam .text, outros candidates; validar mínimo
			text = getattr(resp, "text", None)
			if text is None:
				cands = getattr(resp, "candidates", None)
				if not cands:
					raise RuntimeError("Resposta vazia do Gemini")
			logger.info("Conexão ao Gemini validada.")
			return True
		except Exception as exc:
			logger.warning("Erro ao contatar Gemini: %s", exc)
			if attempt >= cfg.retry_max_attempts:
				logger.error("Excedido número máximo de tentativas (%s).", cfg.retry_max_attempts)
				return False
			# Backoff exponencial limitado
			delay = min(cfg.retry_backoff_initial * (2 ** (attempt - 1)), cfg.retry_backoff_max)
			logger.info("Aguardando %.1fs antes de tentar novamente...", delay)
			time.sleep(delay)
	return False
