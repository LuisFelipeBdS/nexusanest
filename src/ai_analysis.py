from __future__ import annotations

import hashlib
import json
import ast
from typing import Any, Dict, Optional, Tuple, List, Callable

from .config import AppConfig, create_gemini_model, _build_generation_config, logger


def _hash_patient_payload(payload: Dict[str, Any], namespace: str = "default") -> str:
	serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
	return hashlib.sha256((namespace + "::" + serialized).encode("utf-8")).hexdigest()


def _json(obj: Any) -> str:
	return json.dumps(obj, ensure_ascii=False)


def _build_prompt_general(payload: Dict[str, Any]) -> str:
	patient = payload.get("patient", {})
	scores = payload.get("scores", {})

	demo = patient.get("demographics", {})
	comorb = patient.get("comorbidities", {})
	meds = patient.get("medications", {})
	labs = patient.get("labs", {})
	surgical = patient.get("surgical", {})
	functional = patient.get("functional", {})
	physical = patient.get("physical_exam", {})

	asa = scores.get("asa")
	nsqip = scores.get("nsqip")
	rcri = scores.get("rcri")
	ariscat = scores.get("ariscat")
	akics = scores.get("akics")
	predeliric = scores.get("pre_deliric")

	header = (
		"Você é um anestesiologista especialista em avaliação pré-operatória.\n"
		"Analise este paciente baseando-se nos escores validados calculados e forneça uma avaliação de risco perioperatório estruturada.\n\n"
		"INSTRUÇÕES CRÍTICAS: NÃO escreva preâmbulos, saudações ou confirmações (ex.: 'Com certeza...'). "
		"Responda APENAS com um JSON válido exatamente no formato solicitado, iniciando pelo caractere { e terminando em }.\n\n"
		f"DADOS DO PACIENTE: {_json(patient)}\n\n"
		"ESCORES CALCULADOS:\n"
		f"ASA Physical Status: {_json(asa)}\n"
		f"NSQIP Risk Calculator: {_json(nsqip)}\n"
		f"RCRI (Revised Cardiac Risk Index): {_json(rcri)}\n"
		f"ARISCAT: {_json(ariscat)}\n"
		f"AKICS (se aplicável): {_json(akics)}\n"
		f"PRE-DELIRIC: {_json(predeliric)}\n\n"
		f"CIRURGIA: {_json(surgical)}\n\n"
		"Forneça resposta em JSON ESTRITO no formato abaixo, baseada EXCLUSIVAMENTE nos escores validados:\n"
	)
	template = """
{
  "resumo_executivo": "...",
  "por_sistemas": {
    "cardiovascular": ["..."],
    "pulmonar": ["..."],
    "renal": ["..."],
    "delirium": ["..."]
  },
  "estratificacao_geral": "...",
  "recomendacoes": ["..."],
  "medicacoes": {"suspender": ["..."], "manter": ["..."], "ajustar": ["..."]},
  "monitorizacao": ["..."]
}
"""
	footer = "Use linguagem técnica adequada para anestesiologistas e cite guidelines (ACC/AHA, ESC/ESA, ASA, ERAS, ACS-NSQIP) quando relevante."
	return header + template + footer


def _build_prompt_medications(payload: Dict[str, Any]) -> str:
	patient = payload.get("patient", {})
	scores = payload.get("scores", {})
	surgical = patient.get("surgical", {})
	meds = patient.get("medications", {})

	header = (
		"Baseado nos escores de risco calculados e dados clínicos, analise as medicações em uso seguindo guidelines baseadas em evidência.\n\n"
		"INSTRUÇÕES CRÍTICAS: NÃO escreva preâmbulos, saudações ou confirmações. "
		"Responda APENAS com um JSON válido exatamente no formato solicitado, iniciando em { e terminando em }.\n\n"
		"ESCORES DE RISCO:\n"
		f"RCRI: {_json(scores.get('rcri'))}\n"
		f"ARISCAT: {_json(scores.get('ariscat'))}\n"
		f"AKICS: {_json(scores.get('akics'))}\n"
		f"ASA: {_json(scores.get('asa'))}\n\n"
		f"MEDICAÇÕES ATUAIS: {_json(meds)}\n"
		f"TIPO DE CIRURGIA: {_json(surgical)}\n\n"
		"Responda em JSON ESTRITO:\n"
	)
	template = """
{
  "suspender": ["medicação e antecedência (com justificativa)"],
  "manter": ["medicação (com justificativa)"],
  "ajustar": ["medicação e ajuste (baseado em função renal/cardíaca)"],
  "profilaxias": ["profilaxias específicas baseadas nos escores"],
  "bridge": ["cenários de terapia ponte e como conduzir"]
}
"""
	footer = "Referencie guidelines (ACC/AHA, ESC/ESA, ASA) quando aplicável."
	return header + template + footer


def _build_prompt_scores_interpretation(payload: Dict[str, Any]) -> str:
	scores = payload.get("scores", {})
	header = (
		"Interprete os resultados dos escores de forma integrada e clinicamente relevante. Responda em JSON ESTRITO.\n"
		"INSTRUÇÕES CRÍTICAS: Não escreva preâmbulos/saudações/'com certeza'; responda apenas com um JSON válido (inicie em { e termine em }).\n"
		"Resultados:\n"
		f"ASA: {_json(scores.get('asa'))}\n"
		f"NSQIP: {_json(scores.get('nsqip'))}\n"
		f"RCRI: {_json(scores.get('rcri'))}\n"
		f"ARISCAT: {_json(scores.get('ariscat'))}\n"
		f"AKICS: {_json(scores.get('akics'))}\n"
		f"PRE-DELIRIC: {_json(scores.get('pre_deliric'))}\n\n"
		"Formato:\n"
	)
	template = """
{
  "concordancia": ["onde os escores convergem"],
  "divergencia": ["onde divergem e por quê"],
  "relevancia": "qual escore é mais relevante",
  "limitacoes": ["limitações dos escores neste caso"],
  "risco_global": "síntese do risco global",
  "pontos_atencao": ["itens críticos"],
  "otimizacao_preop": ["ações de otimização"]
}
"""
	footer = "Mantenha linguagem técnica apropriada para anestesiologistas."
	return header + template + footer


class AIAnalysisCache:
	def __init__(self) -> None:
		self._cache: Dict[str, Dict[str, Any]] = {}

	def get(self, key: str) -> Optional[Dict[str, Any]]:
		return self._cache.get(key)

	def set(self, key: str, value: Dict[str, Any]) -> None:
		self._cache[key] = value


_cache = AIAnalysisCache()


def _parse_response_text(text: str, expected_keys: Optional[List[str]] = None, defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
	text = (text or "").strip()
	if expected_keys is None:
		expected_keys = ["resumo_executivo", "por_sistemas", "estratificacao_geral", "recomendacoes", "medicacoes", "monitorizacao"]
	# Remove cercas de código ``` ou ```json
	if text.startswith("```"):
		lines = text.splitlines()
		# remove primeira e última linha se cercadas
		if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].strip().startswith("```"):
			text = "\n".join(lines[1:-1]).strip()
	# Tenta carregar diretamente
	for attempt in range(2):
		try:
			data = json.loads(text)
			if isinstance(data, dict):
				# garante chaves esperadas
				for k in expected_keys:
					data.setdefault(k, [] if k not in ("por_sistemas", "medicacoes") else ({} if k == "por_sistemas" else {"suspender": [], "manter": [], "ajustar": []}))
				return data
		except Exception:
			pass
		# segunda tentativa: extrair substring do primeiro '{' ao último '}'
		l = text.find("{")
		r = text.rfind("}")
		if l != -1 and r != -1 and r > l:
			text = text[l : r + 1]
			continue
		break
	# Fallback to Python literal dict
	try:
		data = ast.literal_eval(text)
		if isinstance(data, dict):
			# Normalize keys to expected names
			key_map = {
				"resumo": "resumo_executivo",
				"resumo_executivo": "resumo_executivo",
				"por_sistemas": "por_sistemas",
				"estratificacao_geral": "estratificacao_geral",
				"recomendacoes": "recomendacoes",
				"medicacoes": "medicacoes",
				"monitorizacao": "monitorizacao",
			}
			norm: Dict[str, Any] = {}
			for k, v in data.items():
				kk = key_map.get(str(k), str(k))
				norm[kk] = v
			for k in expected_keys:
				if k == "por_sistemas":
					norm.setdefault(k, {})
				elif k == "medicacoes":
					norm.setdefault(k, {"suspender": [], "manter": [], "ajustar": []})
				else:
					norm.setdefault(k, [])
			return norm
	except Exception:
		pass

	base = defaults.copy() if defaults else {}
	if not base:
		base = {}
	for k in expected_keys:
		if k == "por_sistemas":
			base.setdefault(k, {})
		elif k == "medicacoes":
			base.setdefault(k, {"manter": [], "suspender": [], "ajustar": []})
		else:
			base.setdefault(k, [])
	base.setdefault("_raw_text", text)
	return base


def _run_gemini(prompt: str, cfg: AppConfig) -> Optional[str]:
	model = create_gemini_model(cfg)
	if model is None:
		return None
	attempts = max(1, cfg.retry_max_attempts)
	last_text = None
	for _ in range(attempts):
		try:
			resp = model.generate_content(
				prompt,
				generation_config=_build_generation_config(cfg),
				request_options={"timeout": cfg.timeout_seconds},
			)
			text = getattr(resp, "text", None)
			if not text:
				cands = getattr(resp, "candidates", None)
				text = json.dumps(cands, ensure_ascii=False) if cands else None
			if text:
				return text
			last_text = text or last_text
		except Exception:
			continue
	return last_text


def analyze_general(payload: Dict[str, Any], cfg: AppConfig) -> Tuple[Dict[str, Any], str]:
	key = _hash_patient_payload(payload, namespace="general")
	cached = _cache.get(key)
	if cached:
		return cached, cached.get("_raw", "")
	prompt = _build_prompt_general(payload)
	text = _run_gemini(prompt, cfg)
	expected = ["resumo_executivo", "por_sistemas", "estratificacao_geral", "recomendacoes", "medicacoes", "monitorizacao"]
	defaults = {
		"por_sistemas": {"cardiovascular": [], "pulmonar": [], "renal": [], "delirium": []},
		"medicacoes": {"suspender": [], "manter": [], "ajustar": []},
	}
	if not text:
		fallback = {
			"resumo_executivo": "IA indisponível. Verifique a GOOGLE_API_KEY e conectividade.",
			"por_sistemas": {"cardiovascular": [], "pulmonar": [], "renal": [], "delirium": []},
			"estratificacao_geral": "",
			"recomendacoes": [],
			"medicacoes": {"suspender": [], "manter": [], "ajustar": []},
			"monitorizacao": [],
		}
		_cache.set(key, {**fallback, "_raw": ""})
		logger.warning("AI response empty; returning fallback")
		return fallback, ""
	parsed = _parse_response_text(text, expected_keys=expected, defaults=defaults)
	_cache.set(key, {**parsed, "_raw": text})
	return parsed, text


def analyze_medications(payload: Dict[str, Any], cfg: AppConfig) -> Tuple[Dict[str, Any], str]:
	key = _hash_patient_payload(payload, namespace="medications")
	cached = _cache.get(key)
	if cached:
		return cached, cached.get("_raw", "")
	prompt = _build_prompt_medications(payload)
	text = _run_gemini(prompt, cfg)
	expected = ["suspender", "manter", "ajustar", "profilaxias", "bridge"]
	defaults = {"suspender": [], "manter": [], "ajustar": [], "profilaxias": [], "bridge": []}
	if not text:
		_cache.set(key, {**defaults, "_raw": ""})
		return defaults, ""
	parsed = _parse_response_text(text, expected_keys=expected, defaults=defaults)
	_cache.set(key, {**parsed, "_raw": text})
	return parsed, text


def analyze_scores_interpretation(payload: Dict[str, Any], cfg: AppConfig) -> Tuple[Dict[str, Any], str]:
	key = _hash_patient_payload(payload, namespace="scores_interpretation")
	cached = _cache.get(key)
	if cached:
		return cached, cached.get("_raw", "")
	prompt = _build_prompt_scores_interpretation(payload)
	text = _run_gemini(prompt, cfg)
	expected = ["concordancia", "divergencia", "relevancia", "limitacoes", "risco_global", "pontos_atencao", "otimizacao_preop"]
	defaults = {
		"concordancia": [],
		"divergencia": [],
		"relevancia": "",
		"limitacoes": [],
		"risco_global": "",
		"pontos_atencao": [],
		"otimizacao_preop": [],
	}
	if not text:
		_cache.set(key, {**defaults, "_raw": ""})
		return defaults, ""
	parsed = _parse_response_text(text, expected_keys=expected, defaults=defaults)
	_cache.set(key, {**parsed, "_raw": text})
	return parsed, text


# Back-compat: função genérica anterior

def analyze_with_gemini(payload: Dict[str, Any], cfg: AppConfig) -> Tuple[Dict[str, Any], str]:
	return analyze_general(payload, cfg)
