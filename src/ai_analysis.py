from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional, Tuple, List, Callable

from .config import AppConfig, create_gemini_model, _build_generation_config


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

	prompt = f"""
Você é um anestesiologista especialista em avaliação pré-operatória. NÃO escreva preâmbulos, saudações ou confirmações (ex.: "Com certeza...")
Analise este paciente baseando-se nos escores validados calculados e forneça uma avaliação de risco perioperatório estruturada.

INSTRUÇÕES CRÍTICAS: NÃO escreva preâmbulos, saudações ou confirmações (ex.: "Com certeza..."). Responda APENAS com um JSON válido exatamente no formato solicitado, iniciando pelo caractere { e terminando em }.

DADOS DO PACIENTE: {_json(patient)}

ESCORES CALCULADOS:
ASA Physical Status: {_json(asa)}
NSQIP Risk Calculator: {_json(nsqip)}
RCRI (Revised Cardiac Risk Index): {_json(rcri)}
ARISCAT: {_json(ariscat)}
AKICS (se aplicável): {_json(akics)}
PRE-DELIRIC: {_json(predeliric)}

CIRURGIA: {_json(surgical)}

Forneça resposta em JSON ESTRITO no formato abaixo, baseada EXCLUSIVAMENTE nos escores validados:
{{
  "resumo_executivo": "...",
  "por_sistemas": {{
    "cardiovascular": ["..."],
    "pulmonar": ["..."],
    "renal": ["..."],
    "delirium": ["..."]
  }},
  "estratificacao_geral": "...",
  "recomendacoes": ["..."],
  "medicacoes": {{"suspender": ["..."], "manter": ["..."], "ajustar": ["..."]}},
  "monitorizacao": ["..."]
}}
NÃO escreva preâmbulos, saudações ou confirmações (ex.: "Com certeza...") Use linguagem técnica adequada para anestesiologistas e cite guidelines (ACC/AHA, ESC/ESA, ASA, ERAS, ACS-NSQIP) quando relevante.
"""
	return prompt


def _build_prompt_medications(payload: Dict[str, Any]) -> str:
	patient = payload.get("patient", {})
	scores = payload.get("scores", {})
	surgical = patient.get("surgical", {})
	meds = patient.get("medications", {})

	prompt = f"""
Baseado nos escores de risco calculados e dados clínicos, analise as medicações em uso seguindo guidelines baseadas em evidência. NÃO escreva preâmbulos, saudações ou confirmações (ex.: "Com certeza...")

INSTRUÇÕES CRÍTICAS: NÃO escreva preâmbulos, saudações ou confirmações. Responda APENAS com um JSON válido exatamente no formato solicitado, iniciando em { e terminando em }.

ESCORES DE RISCO:
RCRI: {_json(scores.get('rcri'))}
ARISCAT: {_json(scores.get('ariscat'))}
AKICS: {_json(scores.get('akics'))}
ASA: {_json(scores.get('asa'))}

MEDICAÇÕES ATUAIS: {_json(meds)}
TIPO DE CIRURGIA: {_json(surgical)}

Responda em JSON ESTRITO:
{{
  "suspender": ["medicação e antecedência (com justificativa)"],
  "manter": ["medicação (com justificativa)"],
  "ajustar": ["medicação e ajuste (baseado em função renal/cardíaca)"],
  "profilaxias": ["profilaxias específicas baseadas nos escores"],
  "bridge": ["cenários de terapia ponte e como conduzir"]
}}
Referencie guidelines (ACC/AHA, ESC/ESA, ASA) quando aplicável.
"""
	return prompt


def _build_prompt_scores_interpretation(payload: Dict[str, Any]) -> str:
	scores = payload.get("scores", {})
	prompt = f"""
Interprete os resultados dos escores de forma integrada e clinicamente relevante. Responda em JSON ESTRITO.
INSTRUÇÕES CRÍTICAS: Não escreva preâmbulos/saudações/"com certeza"; responda apenas com um JSON válido (inicie em { e termine em ).
Resultados:
ASA: {_json(scores.get('asa'))}
NSQIP: {_json(scores.get('nsqip'))}
RCRI: {_json(scores.get('rcri'))}
ARISCAT: {_json(scores.get('ariscat'))}
AKICS: {_json(scores.get('akics'))}
PRE-DELIRIC: {_json(scores.get('pre_deliric'))}

Formato:
{{
  "concordancia": ["onde os escores convergem"],
  "divergencia": ["onde divergem e por quê"],
  "relevancia": "qual escore é mais relevante",
  "limitacoes": ["limitações dos escores neste caso"],
  "risco_global": "síntese do risco global",
  "pontos_atencao": ["itens críticos"],
  "otimizacao_preop": ["ações de otimização"]
}}
Mantenha linguagem técnica apropriada para anestesiologistas.
"""
	return prompt


class AIAnalysisCache:
	def __init__(self) -> None:
		self._cache: Dict[str, Dict[str, Any]] = {}

	def get(self, key: str) -> Optional[Dict[str, Any]]:
		return self._cache.get(key)

	def set(self, key: str, value: Dict[str, Any]) -> None:
		self._cache[key] = value


_cache = AIAnalysisCache()


def _parse_response_text(text: str, expected_keys: Optional[List[str]] = None, defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
	text = text.strip()
	if expected_keys is None:
		expected_keys = ["riscos", "recomendacoes", "medicacoes", "consideracoes"]
	try:
		data = json.loads(text)
		if isinstance(data, dict):
			for k in expected_keys:
				if k not in data:
					raise ValueError("missing key")
			return data
	except Exception:
		pass
	# Fallback
	base = defaults.copy() if defaults else {}
	if not base:
		base = {}
	for k in expected_keys:
		base.setdefault(k, [] if k != "medicacoes" else {"manter": [], "suspender": [], "ajustar": []})
	# se não houver nada, coloque texto bruto em uma chave genérica
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
			"resumo_executivo": "IA indisponível",
			"por_sistemas": {"cardiovascular": [], "pulmonar": [], "renal": [], "delirium": []},
			"estratificacao_geral": "",
			"recomendacoes": [],
			"medicacoes": {"suspender": [], "manter": [], "ajustar": []},
			"monitorizacao": [],
		}
		_cache.set(key, {**fallback, "_raw": ""})
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
