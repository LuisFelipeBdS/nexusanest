from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class ScoreResult:
    score: int
    risk_category: str
    details: Dict[str, int]


# RCRI (Revised Cardiac Risk Index)
RCRI_FACTORS = {
    "high_risk_surgery": 1,
    "history_ischemic_heart_disease": 1,
    "history_congestive_heart_failure": 1,
    "history_cerebrovascular_disease": 1,
    "insulin_therapy_diabetes": 1,
    "preoperative_creatinine_gt_2mg_dl": 1,
}


def calculate_rcri(**factors: bool) -> ScoreResult:
    details: Dict[str, int] = {}
    for key in RCRI_FACTORS:
        details[key] = 1 if factors.get(key, False) else 0
    score = sum(details.values())
    if score == 0:
        risk = "Baixo"
    elif score in (1, 2):
        risk = "Intermediário"
    else:
        risk = "Alto"
    return ScoreResult(score=score, risk_category=risk, details=details)


# ARISCAT (pulmonary complications)
ARISCAT_WEIGHTS = {
    # Example simplified weights
    "age_51_80": 3,
    "age_gt_80": 16,
    "low_spo2": 24,  # SpO2 91-95%
    "very_low_spo2": 27,  # SpO2 <= 90%
    "resp_infection_last_month": 17,
    "anemia": 11,
    "surgery_upper_abdominal": 15,
    "surgery_intrathoracic": 24,
    "duration_2_to_3h": 16,
    "duration_gt_3h": 23,
    "emergency_surgery": 8,
}


def calculate_ariscat(**factors: bool) -> Tuple[int, str, Dict[str, int]]:
    details: Dict[str, int] = {}
    for key, weight in ARISCAT_WEIGHTS.items():
        details[key] = weight if factors.get(key, False) else 0
    score = sum(details.values())
    if score < 26:
        risk = "Baixo"
    elif score < 45:
        risk = "Intermediário"
    else:
        risk = "Alto"
    return score, risk, details


# STOP-Bang (obstructive sleep apnea)
STOPBANG_ITEMS = (
    "snoring",
    "tired",
    "observed_apnea",
    "high_bp",
    "bmi_over_35",
    "age_over_50",
    "neck_circ_over_40cm",
    "male",
)


def calculate_stopbang(**answers: bool) -> ScoreResult:
    details: Dict[str, int] = {}
    for key in STOPBANG_ITEMS:
        details[key] = 1 if answers.get(key, False) else 0
    score = sum(details.values())
    if score <= 2:
        risk = "Baixo"
    elif score <= 4:
        risk = "Intermediário"
    else:
        risk = "Alto"
    return ScoreResult(score=score, risk_category=risk, details=details)
