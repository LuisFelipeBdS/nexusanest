from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class ScoreOutput:
    result: Dict
    interpretation: str
    references: Tuple[str, ...]


# -----------------------
# ASA Physical Status
# -----------------------
ASA_DESCRIPTIONS = {
    "I": "ASA I – Paciente saudável, sem doença sistêmica.",
    "II": "ASA II – Doença sistêmica leve sem limitações funcionais significativas.",
    "III": "ASA III – Doença sistêmica grave com limitações funcionais.",
    "IV": "ASA IV – Doença sistêmica grave com ameaça constante à vida.",
    "V": "ASA V – Paciente moribundo, não se espera que sobreviva sem a cirurgia.",
    "VI": "ASA VI – Paciente com morte cerebral, mantido para doação de órgãos.",
}

ASA_BASE_RISK = {
    # Riscos qualitativos auxiliares (não oficiais), para apoio e triagem
    "I": "Muito baixo",
    "II": "Baixo",
    "III": "Intermediário",
    "IV": "Alto",
    "V": "Muito alto",
    "VI": "N/A",
}


def classify_asa(asa_class: str, emergency_modifier: bool = False) -> ScoreOutput:
    asa_class = asa_class.strip().upper()
    if asa_class not in ASA_DESCRIPTIONS:
        raise ValueError("ASA inválido. Use I, II, III, IV, V ou VI.")

    label = asa_class + ("-E" if emergency_modifier else "")
    description = ASA_DESCRIPTIONS[asa_class]
    base_risk = ASA_BASE_RISK.get(asa_class, "")
    if emergency_modifier and asa_class != "VI":
        # Emergência agrava risco; anotação qualitativa
        base_risk = base_risk + " (aumentado por emergência)"

    result = {
        "asa": asa_class,
        "label": label,
        "description": description,
        "risk": base_risk,
        "emergency": emergency_modifier,
    }

    interpretation = (
        "A classificação ASA descreve o estado físico pré-operatório. O modificador E indica procedimento de emergência."
    )

    refs = (
        "ASA Physical Status Classification System (ASA).",
        "Daabiss M. American Society of Anaesthesiologists physical status classification. Int J Periop Clin. 2011.",
    )

    return ScoreOutput(result=result, interpretation=interpretation, references=refs)


# -----------------------
# NSQIP Risk (Proxy Model)
# -----------------------
# Aviso: O NSQIP real utiliza modelos proprietários com CPT codes e algoritmos específicos (ACS). 
# Abaixo, implementamos um proxy simplificado para apoio educacional, NÃO substitui a ferramenta oficial.


def nsqip_proxy(
    *,
    idade: int,
    sexo: str,
    status_funcional: str,
    emergencia: bool,
    asa: str,
    diabetes: bool,
    hipertensao: bool,
    dpoc: bool,
    insuficiencia_cardiaca: bool,
    procedimento: str,
    hematocrito: float,
    creatinina: float,
    albumina: float,
    plaquetas: float,
) -> ScoreOutput:
    """Proxy simplificado do NSQIP.

    Retorna riscos percentuais estimados com base em regras heurísticas. 
    NÃO é o algoritmo oficial do ACS-NSQIP.
    """
    # Normalizações
    sexo = sexo.lower()
    status_funcional = status_funcional.lower()
    asa = asa.strip().upper()

    # Pontuação heurística
    score = 0.0

    # Idade
    if idade >= 80:
        score += 3.0
    elif idade >= 70:
        score += 2.0
    elif idade >= 60:
        score += 1.0

    # Sexo (algumas complicações variam, efeito modesto)
    if sexo == "masculino":
        score += 0.3

    # Status funcional (NSQIP)
    if status_funcional in ("totalmente dependente", "totalmente dependente", "dependente total"):
        score += 3.0
    elif status_funcional in ("parcialmente dependente", "dependente parcial"):
        score += 1.5

    # Emergência
    if emergencia:
        score += 2.5

    # ASA
    asa_map = {"I": 0.0, "II": 0.5, "III": 1.5, "IV": 3.0, "V": 5.0}
    score += asa_map.get(asa, 0.0)

    # Comorbidades principais
    if diabetes:
        score += 0.5
    if hipertensao:
        score += 0.4
    if dpoc:
        score += 0.8
    if insuficiencia_cardiaca:
        score += 1.5

    # Procedimento (categoria aproximada por texto)
    proc = procedimento.lower()
    if any(k in proc for k in ["card", "coron", "valv"]):
        score += 3.0
    elif any(k in proc for k in ["vascular", "aorta", "suprainguinal"]):
        score += 2.5
    elif any(k in proc for k in ["torac", "pulmon", "esofag", "mediast"]):
        score += 2.0
    elif any(k in proc for k in ["abdom", "colect", "gastrect", "hepatec"]):
        score += 1.8
    elif any(k in proc for k in ["ortop", "arthro", "quadril", "joelho"]):
        score += 1.2

    # Laboratório
    if hematocrito < 30:
        score += 1.2
    if creatinina >= 1.5:
        score += 1.0
    if albumina < 3.5:
        score += 1.3
    if plaquetas < 150:
        score += 0.8

    # Converter pontuação em riscos percentuais aproximados (não validados)
    def clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    base = max(0.5, 0.2 * score)

    risks = {
        "mortality_30d_pct": clamp(base * 1.2, 0.1, 25.0),
        "cardiac_complication_pct": clamp(base * 1.0, 0.1, 20.0),
        "pneumonia_pct": clamp(base * 0.9, 0.1, 20.0),
        "ssi_pct": clamp(base * 0.8, 0.1, 20.0),
        "uti_pct": clamp(base * 0.6, 0.1, 15.0),
        "venous_thromboembolism_pct": clamp(base * 0.7, 0.1, 10.0),
        "renal_failure_pct": clamp(base * 0.9, 0.1, 15.0),
        "readmission_pct": clamp(base * 1.1, 0.1, 25.0),
        "reoperation_pct": clamp(base * 0.8, 0.1, 15.0),
        "length_of_stay_days": clamp(1.0 + score * 0.6, 0.5, 30.0),
    }

    interpretation = (
        "Estimativas aproximadas de risco perioperatório baseadas em heurísticas inspiradas no NSQIP. "
        "Use a calculadora oficial do ACS-NSQIP para decisões clínicas definitivas."
    )

    refs = (
        "American College of Surgeons NSQIP Risk Calculator (oficial).",
        "Bilimoria KY et al. Development and evaluation of the ACS NSQIP Surgical Risk Calculator. J Am Coll Surg. 2013.",
    )

    return ScoreOutput(result=risks, interpretation=interpretation, references=refs)


# -----------------------
# RCRI (Revised Cardiac Risk Index)
# -----------------------

def rcri_score(
    *,
    high_risk_surgery: bool,
    ischemic_heart_disease: bool,
    congestive_heart_failure: bool,
    cerebrovascular_disease: bool,
    insulin_treated_diabetes: bool,
    creatinine_gt_2mg_dl: bool,
) -> ScoreOutput:
    """Calcula o RCRI com estratificação de risco e recomendações.

    Fatores (1 ponto cada):
      - Cirurgia de alto risco (intracavitária, intratorácica, vascular suprainguinal)
      - Doença cardíaca isquêmica
      - Insuficiência cardíaca congestiva
      - Doença cerebrovascular (AVC/AIT)
      - Diabetes em uso de insulina
      - Creatinina sérica pré-op > 2,0 mg/dL
    """
    details = {
        "high_risk_surgery": int(bool(high_risk_surgery)),
        "ischemic_heart_disease": int(bool(ischemic_heart_disease)),
        "congestive_heart_failure": int(bool(congestive_heart_failure)),
        "cerebrovascular_disease": int(bool(cerebrovascular_disease)),
        "insulin_treated_diabetes": int(bool(insulin_treated_diabetes)),
        "creatinine_gt_2mg_dl": int(bool(creatinine_gt_2mg_dl)),
    }
    score = sum(details.values())

    if score == 0:
        rclass = "Classe I"
        risk_pct = 0.4
        category = "Baixo"
    elif score == 1:
        rclass = "Classe II"
        risk_pct = 0.9
        category = "Intermediário"
    elif score == 2:
        rclass = "Classe III"
        risk_pct = 7.0
        category = "Intermediário"
    else:
        rclass = "Classe IV"
        risk_pct = 11.0
        category = "Alto"

    recommendations = (
        "Otimize comorbidades, controle rigoroso de glicemia e PA; para riscos moderados/altos, "
        "considerar estratificação adicional (ex.: avaliação funcional, ecocardiograma quando indicado) "
        "e discussão multidisciplinar."
    )

    result = {
        "score": score,
        "class": rclass,
        "risk_percent": risk_pct,
        "risk_category": category,
        "details": details,
        "recommendations": recommendations,
    }

    interpretation = (
        "RCRI estima o risco de eventos cardíacos maiores em cirurgias não cardíacas. "
        "A pontuação é a soma de 6 fatores de 1 ponto."
    )

    refs = (
        "Lee TH et al. Circulation. 1999;100(10):1043-1049.",
        "ACC/AHA perioperative guidelines.",
    )

    return ScoreOutput(result=result, interpretation=interpretation, references=refs)


# -----------------------
# ARISCAT (pulmonary complications)
# -----------------------

def ariscat_score(
    *,
    age_51_80: bool,
    age_gt_80: bool,
    spo2_le_95: bool,
    resp_infection_last_month: bool,
    anemia_hb_le_10: bool,
    incision_abd_upper: bool,
    incision_intrathoracic: bool,
    duration_2_to_3h: bool,
    duration_gt_3h: bool,
    emergency_surgery: bool,
) -> ScoreOutput:
    """Calcula escore ARISCAT e categoria de risco de complicações pulmonares."""
    details_points: Dict[str, int] = {}

    # Idade (não somar ambas)
    details_points["age_51_80"] = 3 if (age_51_80 and not age_gt_80) else 0
    details_points["age_gt_80"] = 16 if age_gt_80 else 0

    # SpO2 <= 95%
    details_points["spo2_le_95"] = 8 if spo2_le_95 else 0

    # Infecção respiratória < 1 mês
    details_points["resp_infection_last_month"] = 17 if resp_infection_last_month else 0

    # Anemia Hb <= 10 g/dL
    details_points["anemia_hb_le_10"] = 11 if anemia_hb_le_10 else 0

    # Incisão: escolha o maior aplicável
    incision_points = 0
    if incision_intrathoracic:
        incision_points = max(incision_points, 24)
    if incision_abd_upper:
        incision_points = max(incision_points, 15)
    details_points["incision"] = incision_points

    # Duração: escolha o maior aplicável
    duration_points = 0
    if duration_gt_3h:
        duration_points = max(duration_points, 23)
    elif duration_2_to_3h:
        duration_points = max(duration_points, 16)
    details_points["duration"] = duration_points

    # Emergência
    details_points["emergency_surgery"] = 8 if emergency_surgery else 0

    total = sum(details_points.values())

    if total < 26:
        risk_category = "Baixo"
        probability_cpp = 1.6
    elif total < 45:
        risk_category = "Intermediário"
        probability_cpp = 13.3
    else:
        risk_category = "Alto"
        probability_cpp = 42.1

    result = {
        "score": total,
        "risk_category": risk_category,
        "probability_cpp_percent": probability_cpp,
        "details": details_points,
    }

    interpretation = (
        "ARISCAT estima risco de complicações pulmonares pós-operatórias com base em idade, oxigenação, infecção recente, anemia, sítio da incisão, duração e emergência."
    )

    refs = (
        "Canet J et al. Prediction of postoperative pulmonary complications. Anesthesiology. 2010;113(6):1338-1350.",
    )

    return ScoreOutput(result=result, interpretation=interpretation, references=refs)


# -----------------------
# AKICS (Acute Kidney Injury after Cardiac Surgery)
# -----------------------

def akics_score(
    *,
    idade: int,
    sexo_feminino: bool,
    insuficiencia_cardiaca: bool,
    hipertensao: bool,
    emergencia: bool,
    tipo_cirurgia: str,
    creatinina_mg_dl: float,
    nao_cardiaca_complexidade: Optional[str] = None,
) -> ScoreOutput:
    """Calcula escore AKICS pré-operatório.

    tipo_cirurgia: "coronariana" | "valvar" | "combinada" | "nao_cardiaca"
    nao_cardiaca_complexidade (apenas quando tipo_cirurgia = "nao_cardiaca"): "baixa" | "media" | "alta"
    """
    # Validações básicas
    if idade < 0 or idade > 120:
        raise ValueError("Idade inválida")
    if creatinina_mg_dl < 0 or creatinina_mg_dl > 20:
        raise ValueError("Creatinina fora de faixa plausível")
    tipo = tipo_cirurgia.lower().strip()
    if tipo not in {"coronariana", "valvar", "combinada", "nao_cardiaca"}:
        raise ValueError("tipo_cirurgia inválido")
    if tipo == "nao_cardiaca" and nao_cardiaca_complexidade is not None:
        if nao_cardiaca_complexidade.lower() not in {"baixa", "media", "alta"}:
            raise ValueError("nao_cardiaca_complexidade inválida")

    # Pontuação
    points = 0.0
    components: Dict[str, float] = {}

    age_points = idade / 10.0
    points += age_points
    components["idade/10"] = round(age_points, 2)

    if sexo_feminino:
        points += 1.0
        components["sexo_feminino"] = 1.0
    else:
        components["sexo_feminino"] = 0.0

    if insuficiencia_cardiaca:
        points += 1.0
        components["insuficiencia_cardiaca"] = 1.0
    else:
        components["insuficiencia_cardiaca"] = 0.0

    if hipertensao:
        points += 1.0
        components["hipertensao"] = 1.0
    else:
        components["hipertensao"] = 0.0

    if emergencia:
        points += 2.0
        components["emergencia"] = 2.0
    else:
        components["emergencia"] = 0.0

    if tipo == "valvar":
        points += 1.0
        components["cirurgia_valvar"] = 1.0
    else:
        components["cirurgia_valvar"] = 0.0

    if tipo == "combinada":
        points += 2.0
        components["cirurgia_combinada"] = 2.0
    else:
        components["cirurgia_combinada"] = 0.0

    # Creatinina
    if 1.2 <= creatinina_mg_dl <= 2.0:
        points += 2.0
        components["creatinina_1.2_2.0"] = 2.0
    elif creatinina_mg_dl > 2.0:
        points += 5.0
        components["creatinina_>2.0"] = 5.0
    else:
        components["creatinina_1.2_2.0"] = 0.0
        components["creatinina_>2.0"] = 0.0

    # Adaptação para não-cardíacas
    if tipo == "nao_cardiaca":
        comp = (nao_cardiaca_complexidade or "baixa").lower()
        if comp == "alta":
            points += 1.0
            components["nao_cardiaca_complexidade_alta"] = 1.0
        elif comp == "media":
            points += 0.5
            components["nao_cardiaca_complexidade_media"] = 0.5
        else:
            components["nao_cardiaca_complexidade_baixa"] = 0.0

    # Estratificação
    if points <= 2:
        categoria = "Muito baixo"
        prob = 2.0
    elif points <= 5:
        categoria = "Baixo"
        prob = 8.0
    elif points <= 8:
        categoria = "Moderado"
        prob = 18.0
    elif points <= 13:
        categoria = "Alto"
        prob = 35.0
    else:
        categoria = "Muito alto"
        prob = 50.0

    result = {
        "pontuacao_total": round(points, 2),
        "categoria_risco": categoria,
        "probabilidade_percentual": prob,
        "detalhes": components,
    }

    interpretacao = (
        "AKICS pré-operatório estima risco de injúria renal aguda no pós-operatório de cirurgia cardíaca; "
        "os pontos refletem idade, comorbidades, urgência, tipo cirúrgico e creatinina. Adaptação não-cardíaca usa complexidade como proxy."
    )
    recomendacoes = (
        "Otimizar hemodinâmica e perfusão renal, evitar nefrotóxicos, balancear fluidos e considerar monitoração estreita em pacientes de risco."
    )
    refs = (
        "Wijeysundera DN et al., desenvolvimento de escores para IRA pós-cirurgia cardíaca (literatura de AKI).",
    )

    result_struct = {
        **result,
        "interpretacao_clinica": interpretacao,
        "recomendacoes": recomendacoes,
        "referencia_bibliografica": refs[0],
    }

    return ScoreOutput(result=result_struct, interpretation=interpretacao, references=refs)


# -----------------------
# PRE-DELIRIC (Delirium in ICU)
# -----------------------

def pre_deliric_score(
    *,
    idade: int,
    apache_ii: float,
    grupo_admissao: str,
    coma: bool,
    infeccao: bool,
    ph: float,
    hco3: Optional[float],
    sedativos: bool,
    morfina: bool,
    ureia_mg_dl: float,
    creatinina_mg_dl: float,
) -> ScoreOutput:
    """Calcula PRE-DELIRIC adaptado ao perioperatório (entrada inicial UTI)."""
    if idade < 0 or idade > 120:
        raise ValueError("Idade inválida")
    if not (0.0 <= apache_ii <= 71.0):
        raise ValueError("APACHE II fora de faixa (0-71)")
    if ph < 6.8 or ph > 7.8:
        raise ValueError("pH fora de faixa plausível (6.8-7.8)")
    if ureia_mg_dl < 0 or creatinina_mg_dl <= 0:
        raise ValueError("Ureia/Creatinina inválidas")

    # Idade
    age_pts = 0
    if idade >= 80:
        age_pts = 6
    elif idade >= 70:
        age_pts = 5
    elif idade >= 60:
        age_pts = 2
    elif idade >= 50:
        age_pts = 1

    # APACHE II
    apache_pts = 0
    if apache_ii >= 20:
        apache_pts = 5
    elif apache_ii >= 15:
        apache_pts = 3
    elif apache_ii >= 10:
        apache_pts = 2

    # Grupo de admissão
    grp = grupo_admissao.strip().lower()
    grp_map = {"clínico": 0, "clinico": 0, "cirúrgico": 1, "cirurgico": 1, "trauma": 2, "neuro": 5, "neurocirurgia": 5}
    if grp not in grp_map:
        raise ValueError("grupo_admissao inválido")
    adm_pts = grp_map[grp]

    coma_pts = 4 if coma else 0
    inf_pts = 1 if infeccao else 0
    acidose_pts = 2 if ph < 7.35 else 0  # usa pH; HCO3 pode refinar
    sed_pts = 1 if sedativos else 0
    morf_pts = 2 if morfina else 0

    ratio = ureia_mg_dl / creatinina_mg_dl
    ratio_pts = 0
    if ratio >= 10:
        ratio_pts = 2
    elif ratio >= 5:
        ratio_pts = 1

    details = {
        "idade_pts": age_pts,
        "apache_pts": apache_pts,
        "grupo_admissao_pts": adm_pts,
        "coma_pts": coma_pts,
        "infeccao_pts": inf_pts,
        "acidose_pts": acidose_pts,
        "sedativos_pts": sed_pts,
        "morfina_pts": morf_pts,
        "ureia_creatinina_ratio": round(ratio, 2),
        "ratio_pts": ratio_pts,
    }

    total = sum(v for k, v in details.items() if k.endswith("_pts"))

    if total <= 4:
        categoria = "Muito baixo"
        prob = 5.0
    elif total <= 9:
        categoria = "Baixo"
        prob = 15.0
    elif total <= 15:
        categoria = "Moderado"
        prob = 35.0
    else:
        categoria = "Alto"
        prob = 50.0

    result = {
        "pontuacao_total": int(total),
        "categoria_risco": categoria,
        "probabilidade_percentual": prob,
        "detalhes": details,
    }

    interpretacao = (
        "PRE-DELIRIC estima risco de delirium em UTI nas primeiras 24h. "
        "Este cálculo adaptado usa variáveis perioperatórias comuns (idade, APACHE II, admissão, coma, infecção, acidose, sedativos/opioides e relação ureia/creatinina)."
    )
    recomendacoes = (
        "Implementar medidas preventivas (reorientação, higiene do sono, mobilização precoce, evitar polifarmácia e sedação excessiva)."
    )
    refs = (
        "van den Boogaard M et al. The PRE-DELIRIC model. Intensive Care Med. 2012.",
    )

    result_struct = {
        **result,
        "interpretacao_clinica": interpretacao,
        "recomendacoes": recomendacoes,
        "referencia_bibliografica": refs[0],
    }

    return ScoreOutput(result=result_struct, interpretation=interpretacao, references=refs)
