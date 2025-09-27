from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src.config import load_config
from src.risk_scores import calculate_rcri, calculate_ariscat, calculate_stopbang
from src.ai_assistant import generate_recommendations
from src.reporting import build_pdf_report
from src.scores import (
    classify_asa,
    nsqip_proxy,
    rcri_score as rcri_score_full,
    ariscat_score as ariscat_score_full,
    akics_score,
    pre_deliric_score,
)


# --------- Page Config ---------
st.set_page_config(
    page_title="HelpAnest - Estratificação de Risco Perioperatório",
    page_icon="🩺",
    layout="wide",
)

# --------- Custom CSS ---------
CUSTOM_CSS = """
<style>
    :root {
        --primary: #0B5FA5;   /* azul médico */
        --primary-700: #094b82;
        --secondary: #E6F0FA; /* fundo suave */
        --bg: #F8FAFC;        /* cinza-azulado claro */
        --text: #0f172a;      /* cinza escuro legível */
        --card-bg: #ffffff;
        --muted: #64748b;
    }

    .app-container {
        background: var(--bg);
    }

    /* Header */
    .app-header {
        display: flex;
        align-items: center;
        gap: 16px;
        background: linear-gradient(90deg, var(--primary) 0%, var(--primary-700) 100%);
        color: #fff;
        padding: 16px 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        margin-bottom: 12px;
    }
    .app-header h1 {
        margin: 0;
        font-weight: 700;
        letter-spacing: .2px;
    }
    .app-header .subtitle {
        margin: 0;
        font-size: 13px;
        color: #cfe7ff;
    }

    /* Cards */
    .card {
        background: var(--card-bg);
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }

    /* Footer */
    .footer {
        margin-top: 24px;
        padding: 16px;
        color: var(--muted);
        text-align: center;
        border-top: 1px solid #e5e7eb;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------- Config & Session ---------
config = load_config()

if "patient" not in st.session_state:
    st.session_state["patient"] = {
        "demographics": {
            "nome": "",
            "idade": 60,
            "sexo": "Feminino",
            "peso_kg": 70.0,
            "altura_cm": 170.0,
            "imc": 24.22,
            "asa": "II",
            "asa_emergencia": False,
        },
        "comorbidities": {
            # Cardiovasculares
            "hipertensao": False,
            "doenca_cardiaca_isquemica": False,
            "insuficiencia_cardiaca": False,
            "arritmias": False,
            "doenca_cerebrovascular": False,
            # Endocrino-metabólicas
            "diabetes_tipo_1": False,
            "diabetes_tipo_2": False,
            "uso_insulina": False,
            "dislipidemia": False,
            "obesidade": False,
            # Pulmonares
            "dpoc": False,
            "asma": False,
            "infeccao_respiratoria_mes": False,
            "pneumopatia_restritiva": False,
            # Renais
            "insuficiencia_renal_cronica": False,
            "uso_diureticos": False,
            "uso_ieca_bra": False,
            # Neurológicas
            "demencia": False,
            "comprometimento_cognitivo": False,
            "deficits_sensoriais": False,
        },
        "medications": {
            "list_text": "",
            "classes": {
                "sedativos_benzos": False,
                "opioides": False,
                "anticoagulantes": False,
                "antidiabeticos_orais": False,
                "insulina": False,
            },
        },
        "labs": {
            "hemoglobina": 0.0,
            "hematocrito": 0.0,
            "creatinina": 0.0,
            "ureia": 0.0,
            "albumina": 0.0,
            "plaquetas": 0.0,
            "glicemia_jejum": 0.0,
            "ph": 7.4,
            "hco3": 24.0,
        },
        "surgical": {
            "tipo_cirurgia": "",
            "porte": "Médio",
            "urgencia": "Eletiva",
            "duracao_prevista_h": 2.0,
            "anestesia_planejada": "Geral",
        },
        "functional": {
            "mets": 4.0,
            "sobe_escadas": "Sim",
            "avd": [],
            "exercicio": "Sedentário",
        },
        "physical_exam": {
            # Sinais vitais pré-operatórios
            "spo2_ar_ambiente": 98.0,
            "pa_sistolica": 120,
            "pa_diastolica": 80,
            "fc": 75,
            # Exame físico adicional
            "ausculta_cardiaca": "Normal",
            "achados_cardiacos": "",
            "ausculta_pulmonar": "Normal",
            "achados_pulmonares": "",
            "edemas": False,
            "ingurgitamento_jugular": False,
        },
    }
if "results" not in st.session_state:
    st.session_state["results"] = {}
if "ai_summary" not in st.session_state:
    st.session_state["ai_summary"] = None

if st.session_state.get("disclaimer_ok", False):
    # --------- Header ---------
    col_logo, col_title = st.columns([1, 6])
    with col_logo:
        logo_path = Path("assets") / "logo.png"
        if logo_path.exists():
            st.image(str(logo_path), use_column_width=True)
        else:
            st.markdown("<div style='height:48px'></div>", unsafe_allow_html=True)
    with col_title:
        st.markdown(
            f"""
            <div class="app-header">
                <div>
                    <h1>{config.app_name}</h1>
                    <p class="subtitle">Estratificação de risco perioperatório com escores validados e apoio de IA</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------- Sidebar (Navegação e Config) ---------
    with st.sidebar:
        st.subheader("Navegação")
        section = st.radio(
            "Seções",
            options=["Dados do Paciente", "Cálculo de Riscos", "Relatório"],
            index=0,
        )

        st.markdown("---")
        st.caption("Relatórios serão salvos em 'reports/'. A IA usa a chave do .env.")

    # --------- Disclaimer (Gate) ---------
    if "disclaimer_ok" not in st.session_state:
        st.session_state["disclaimer_ok"] = False

    if not st.session_state["disclaimer_ok"]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("""
    **Aviso Importante**

    Essa plataforma foi desenvolvida para o Projeto de Iniciação Tecnológica "Desenvolvimento de Plataforma Digital para Estratificação de Risco
    Perioperatório como Ferramenta Auxiliar à Avaliação Pré-Anestésica". Ela foi submetida ao Comitê de Ética em Pesquisa do Hospital Universitário Onofre Lopes e está inscrita sob o CAAE XXXXXXXXXXX. O pesquisador responsável é Luis Felipe Barbosa da Silva, disponível no contato XXXXXXX. O projeto está orientado pelo Prof. Dr. Wallace Andrino da Silva.

    Clique no botão abaixo para confirmar que leu e entendeu este aviso.
    """)
        if st.button("Entendi", type="primary"):
            st.session_state["disclaimer_ok"] = True
            st.experimental_rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # Render normal app below
        pass

    # --------- Helpers ---------
    def _show_patient_form() -> None:
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Dados do Paciente")

            # DADOS DEMOGRÁFICOS
            st.markdown("**DADOS DEMOGRÁFICOS**")
            d1, d2, d3, d4, d5 = st.columns([2, 1, 1, 1, 1])
            with d1:
                st.session_state["patient"]["demographics"]["nome"] = st.text_input(
                    "Nome",
                    st.session_state["patient"]["demographics"].get("nome", ""),
                    help="Nome completo do paciente",
                )
            with d2:
                st.session_state["patient"]["demographics"]["idade"] = st.number_input(
                    "Idade",
                    min_value=0,
                    max_value=120,
                    value=int(st.session_state["patient"]["demographics"].get("idade", 60)),
                    help="Idade em anos",
                )
            with d3:
                st.session_state["patient"]["demographics"]["sexo"] = st.selectbox(
                    "Sexo",
                    ["Feminino", "Masculino"],
                    index=0 if st.session_state["patient"]["demographics"].get("sexo") == "Feminino" else 1,
                    help="Sexo biológico",
                )
            with d4:
                # ASA I-VI
                st.session_state["patient"]["demographics"]["asa"] = st.radio(
                    "ASA Physical Status",
                    options=["I", "II", "III", "IV", "V", "VI"],
                    index=["I", "II", "III", "IV", "V", "VI"].index(st.session_state["patient"]["demographics"].get("asa", "II")),
                    help="Classificação ASA (I a VI)",
                )
            with d5:
                st.session_state["patient"]["demographics"]["asa_emergencia"] = st.checkbox(
                    "Emergência (E)",
                    value=bool(st.session_state["patient"]["demographics"].get("asa_emergencia", False)),
                    help="Marque se o procedimento é em caráter de emergência (ASA-E)",
                )

            d6, d7, d8 = st.columns(3)
            with d6:
                peso = st.number_input(
                    "Peso (kg)",
                    min_value=20.0,
                    max_value=300.0,
                    value=float(st.session_state["patient"]["demographics"].get("peso_kg", 70.0)),
                    step=0.1,
                    help="Peso corporal em quilogramas",
                )
                st.session_state["patient"]["demographics"]["peso_kg"] = peso
            with d7:
                altura_cm = st.number_input(
                    "Altura (cm)",
                    min_value=100.0,
                    max_value=250.0,
                    value=float(st.session_state["patient"]["demographics"].get("altura_cm", 170.0)),
                    step=0.1,
                    help="Altura em centímetros",
                )
                st.session_state["patient"]["demographics"]["altura_cm"] = altura_cm
            with d8:
                altura_m = max(altura_cm / 100.0, 0.5)
                imc = round(peso / (altura_m ** 2), 2)
                st.session_state["patient"]["demographics"]["imc"] = imc
                st.metric("IMC", imc)

            # Validations
            if st.session_state["patient"]["demographics"]["idade"] > 110:
                st.warning("Idade acima do esperado; verifique o valor informado.")
            if imc < 16 or imc > 50:
                st.info("IMC fora da faixa usual; considerar avaliação nutricional.")

            st.markdown("---")
            # COMORBIDADES (por sistema)
            st.markdown("**COMORBIDADES**")

            st.caption("Cardiovasculares")
            cv1, cv2, cv3 = st.columns(3)
            with cv1:
                st.session_state["patient"]["comorbidities"]["hipertensao"] = st.checkbox("Hipertensão arterial", value=st.session_state["patient"]["comorbidities"]["hipertensao"], help="HAS diagnosticada")
                st.session_state["patient"]["comorbidities"]["doenca_cardiaca_isquemica"] = st.checkbox("Doença cardíaca isquêmica", value=st.session_state["patient"]["comorbidities"]["doenca_cardiaca_isquemica"], help="DAC/IAM/angina")
            with cv2:
                st.session_state["patient"]["comorbidities"]["insuficiencia_cardiaca"] = st.checkbox("Insuficiência cardíaca congestiva", value=st.session_state["patient"]["comorbidities"]["insuficiencia_cardiaca"], help="IC com FE reduzida/preservada")
                st.session_state["patient"]["comorbidities"]["arritmias"] = st.checkbox("Arritmias", value=st.session_state["patient"]["comorbidities"]["arritmias"], help="FA/flutter, etc.")
            with cv3:
                st.session_state["patient"]["comorbidities"]["doenca_cerebrovascular"] = st.checkbox("Doença cerebrovascular (AVC/AIT)", value=st.session_state["patient"]["comorbidities"]["doenca_cerebrovascular"], help="AVC/AIT prévio")

            st.caption("Endócrino-metabólicas")
            en1, en2, en3 = st.columns(3)
            with en1:
                st.session_state["patient"]["comorbidities"]["diabetes_tipo_1"] = st.checkbox("DM tipo 1", value=st.session_state["patient"]["comorbidities"]["diabetes_tipo_1"])
                st.session_state["patient"]["comorbidities"]["diabetes_tipo_2"] = st.checkbox("DM tipo 2", value=st.session_state["patient"]["comorbidities"]["diabetes_tipo_2"])
            with en2:
                st.session_state["patient"]["comorbidities"]["uso_insulina"] = st.checkbox("Uso de insulina", value=st.session_state["patient"]["comorbidities"]["uso_insulina"])
                st.session_state["patient"]["comorbidities"]["dislipidemia"] = st.checkbox("Dislipidemia", value=st.session_state["patient"]["comorbidities"]["dislipidemia"])
            with en3:
                st.session_state["patient"]["comorbidities"]["obesidade"] = st.checkbox("Obesidade", value=st.session_state["patient"]["comorbidities"]["obesidade"], help="IMC ≥ 30")

            st.caption("Pulmonares")
            pu1, pu2, pu3 = st.columns(3)
            with pu1:
                st.session_state["patient"]["comorbidities"]["dpoc"] = st.checkbox("DPOC", value=st.session_state["patient"]["comorbidities"]["dpoc"])
                st.session_state["patient"]["comorbidities"]["asma"] = st.checkbox("Asma brônquica", value=st.session_state["patient"]["comorbidities"]["asma"])
            with pu2:
                st.session_state["patient"]["comorbidities"]["infeccao_respiratoria_mes"] = st.checkbox("Infecção respiratória (< 1 mês)", value=st.session_state["patient"]["comorbidities"]["infeccao_respiratoria_mes"], help="Vias aéreas inferiores/superiores")
            with pu3:
                st.session_state["patient"]["comorbidities"]["pneumopatia_restritiva"] = st.checkbox("Pneumopatias restritivas", value=st.session_state["patient"]["comorbidities"]["pneumopatia_restritiva"])

            st.caption("Renais")
            re1, re2, re3 = st.columns(3)
            with re1:
                st.session_state["patient"]["comorbidities"]["insuficiencia_renal_cronica"] = st.checkbox("Insuficiência renal crônica", value=st.session_state["patient"]["comorbidities"]["insuficiencia_renal_cronica"])
            with re2:
                st.session_state["patient"]["comorbidities"]["uso_diureticos"] = st.checkbox("Uso de diuréticos", value=st.session_state["patient"]["comorbidities"]["uso_diureticos"])
            with re3:
                st.session_state["patient"]["comorbidities"]["uso_ieca_bra"] = st.checkbox("Uso de IECA/BRA", value=st.session_state["patient"]["comorbidities"]["uso_ieca_bra"])

            st.caption("Neurológicas")
            ne1, ne2, ne3 = st.columns(3)
            with ne1:
                st.session_state["patient"]["comorbidities"]["demencia"] = st.checkbox("Demência", value=st.session_state["patient"]["comorbidities"]["demencia"])
            with ne2:
                st.session_state["patient"]["comorbidities"]["comprometimento_cognitivo"] = st.checkbox("Comprometimento cognitivo", value=st.session_state["patient"]["comorbidities"]["comprometimento_cognitivo"])
            with ne3:
                st.session_state["patient"]["comorbidities"]["deficits_sensoriais"] = st.checkbox("Déficits sensoriais (visual/auditivo)", value=st.session_state["patient"]["comorbidities"]["deficits_sensoriais"])

            st.markdown("---")
            # MEDICAÇÕES EM USO
            st.markdown("**MEDICAÇÕES EM USO**")
            st.session_state["patient"]["medications"]["list_text"] = st.text_area(
                "Liste as medicações em uso (nome e dose)",
                value=st.session_state["patient"]["medications"].get("list_text", ""),
                placeholder="Ex.: AAS 100 mg/dia; Losartana 50 mg 12/12h; Metformina 850 mg 8/8h",
                help="Descreva os fármacos relevantes, incluindo dose e frequência",
            )
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.session_state["patient"]["medications"]["classes"]["sedativos_benzos"] = st.checkbox("Sedativos/Benzos", value=st.session_state["patient"]["medications"]["classes"]["sedativos_benzos"])
            with m2:
                st.session_state["patient"]["medications"]["classes"]["opioides"] = st.checkbox("Morfina/Opioides", value=st.session_state["patient"]["medications"]["classes"]["opioides"])
            with m3:
                st.session_state["patient"]["medications"]["classes"]["anticoagulantes"] = st.checkbox("Anticoagulantes", value=st.session_state["patient"]["medications"]["classes"]["anticoagulantes"])
            with m4:
                st.session_state["patient"]["medications"]["classes"]["antidiabeticos_orais"] = st.checkbox("Antidiabéticos orais", value=st.session_state["patient"]["medications"]["classes"]["antidiabeticos_orais"])
            st.session_state["patient"]["medications"]["classes"]["insulina"] = st.checkbox("Insulina", value=st.session_state["patient"]["medications"]["classes"]["insulina"])

            st.markdown("---")
            # EXAMES LABORATORIAIS
            st.markdown("**EXAMES LABORATORIAIS**")
            l1, l2, l3 = st.columns(3)
            with l1:
                st.session_state["patient"]["labs"]["hemoglobina"] = st.number_input("Hemoglobina (g/dL)", min_value=0.0, max_value=25.0, step=0.1, value=float(st.session_state["patient"]["labs"]["hemoglobina"]))
                st.session_state["patient"]["labs"]["hematocrito"] = st.number_input("Hematócrito (%)", min_value=0.0, max_value=70.0, step=0.1, value=float(st.session_state["patient"]["labs"]["hematocrito"]))
                st.session_state["patient"]["labs"]["plaquetas"] = st.number_input("Plaquetas (10³/µL)", min_value=0.0, max_value=1500.0, step=1.0, value=float(st.session_state["patient"]["labs"]["plaquetas"]))
            with l2:
                st.session_state["patient"]["labs"]["creatinina"] = st.number_input("Creatinina sérica (mg/dL)", min_value=0.0, max_value=20.0, step=0.1, value=float(st.session_state["patient"]["labs"]["creatinina"]))
                st.session_state["patient"]["labs"]["ureia"] = st.number_input("Ureia (mg/dL)", min_value=0.0, max_value=300.0, step=1.0, value=float(st.session_state["patient"]["labs"]["ureia"]))
                st.session_state["patient"]["labs"]["albumina"] = st.number_input("Albumina sérica (g/dL)", min_value=0.0, max_value=10.0, step=0.1, value=float(st.session_state["patient"]["labs"]["albumina"]))
            with l3:
                st.session_state["patient"]["labs"]["glicemia_jejum"] = st.number_input("Glicemia de jejum (mg/dL)", min_value=0.0, max_value=1000.0, step=1.0, value=float(st.session_state["patient"]["labs"]["glicemia_jejum"]))
                st.session_state["patient"]["labs"]["ph"] = st.number_input("Gasometria: pH", min_value=6.8, max_value=7.8, step=0.01, value=float(st.session_state["patient"]["labs"]["ph"]))
                st.session_state["patient"]["labs"]["hco3"] = st.number_input("Gasometria: HCO₃ (mEq/L)", min_value=0.0, max_value=60.0, step=0.5, value=float(st.session_state["patient"]["labs"]["hco3"]))

            # Real-time lab validations
            labs = st.session_state["patient"]["labs"]
            if labs["hemoglobina"] and (labs["hemoglobina"] < 8 or labs["hemoglobina"] > 18):
                st.warning("Hemoglobina fora da faixa comum (8–18 g/dL).")
            if labs["creatinina"] and labs["creatinina"] > 1.5:
                st.info("Creatinina elevada; considerar estratificação renal.")
            if labs["albumina"] and labs["albumina"] < 3.5:
                st.info("Albumina baixa; considerar risco nutricional.")
            if labs["ph"] and labs["hco3"] and (labs["ph"] < 7.35 and labs["hco3"] < 22):
                st.warning("Padrão compatível com acidose metabólica (pH<7,35 e HCO₃<22).")

            st.markdown("---")
            # DADOS CIRÚRGICOS
            st.markdown("**DADOS CIRÚRGICOS**")
            cir1, cir2, cir3 = st.columns(3)
            categorias = [
                "Cardíaca", "Vascular", "Torácica", "Abdominal", "Ortopédica", "Neurocirurgia", "Urologia", "Ginecologia", "Outras"
            ]
            subtipos = {
                "Cardíaca": ["Coronariana", "Valvar", "Combinada"],
                "Vascular": ["Suprainguinal", "Infrainguinal"],
                "Torácica": ["Pulmonar", "Esofágica", "Mediastinal"],
                "Abdominal": ["Alta (epigástrica)", "Baixa (pélvica)"],
                "Ortopédica": ["Grande porte", "Pequeno porte"],
                "Neurocirurgia": ["Craniana", "Espinal"],
                "Urologia": ["Prostatectomia", "Nefrectomia", "Outras"],
                "Ginecologia": ["Histerectomia", "Outras"],
                "Outras": ["Não especificado"],
            }
            with cir1:
                cat = st.selectbox(
                    "Tipo de cirurgia",
                    options=categorias,
                    index=categorias.index(st.session_state["patient"]["surgical"].get("tipo_cirurgia", "Outras")) if st.session_state["patient"]["surgical"].get("tipo_cirurgia") in categorias else categorias.index("Outras"),
                    help="Especialidade principal do procedimento",
                )
                st.session_state["patient"]["surgical"]["tipo_cirurgia"] = cat
                sub_opts = subtipos.get(cat, ["Não especificado"])
                st.session_state["patient"]["surgical"]["subtipo"] = st.selectbox(
                    "Subtipo",
                    options=sub_opts,
                    index=0,
                    help="Subcategoria relevante para estratificação",
                )
            with cir2:
                st.session_state["patient"]["surgical"]["porte"] = st.selectbox(
                    "Porte cirúrgico",
                    options=["Pequeno", "Médio", "Grande", "Especial"],
                    index=["Pequeno", "Médio", "Grande", "Especial"].index(st.session_state["patient"]["surgical"].get("porte", "Médio")),
                    help="Classificação do porte",
                )
                st.session_state["patient"]["surgical"]["urgencia"] = st.selectbox(
                    "Urgência",
                    options=["Eletiva", "Urgência", "Emergência"],
                    index=["Eletiva", "Urgência", "Emergência"].index(st.session_state["patient"]["surgical"].get("urgencia", "Eletiva")),
                    help="Caráter do procedimento",
                )
            with cir3:
                st.session_state["patient"]["surgical"]["duracao_cat"] = st.radio(
                    "Duração prevista",
                    options=["<2h", "2-3h", ">3h"],
                    index=["<2h", "2-3h", ">3h"].index(st.session_state["patient"]["surgical"].get("duracao_cat", "2-3h")),
                    help="Estimativa de duração",
                )
                st.session_state["patient"]["surgical"]["anestesia_planejada"] = st.selectbox(
                    "Anestesia planejada",
                    options=["Geral", "Peridural", "Raqui", "Bloqueio periférico", "Sedação", "Mista"],
                    index=["Geral", "Peridural", "Raqui", "Bloqueio periférico", "Sedação", "Mista"].index(st.session_state["patient"]["surgical"].get("anestesia_planejada", "Geral")),
                    help="Técnica anestésica prevista",
                )
            cir4, cir5 = st.columns([2, 1])
            with cir4:
                st.session_state["patient"]["surgical"]["incisao_site"] = st.selectbox(
                    "Local da incisão (impacta ARISCAT)",
                    options=["Intratorácica", "Abdome superior", "Abdome inferior", "Outras"],
                    index=["Intratorácica", "Abdome superior", "Abdome inferior", "Outras"].index(st.session_state["patient"]["surgical"].get("incisao_site", "Outras")),
                    help="Selecione o sítio principal da incisão",
                )
            with cir5:
                # Classificação automática de risco cirúrgico
                porte = st.session_state["patient"]["surgical"]["porte"]
                incisao = st.session_state["patient"]["surgical"].get("incisao_site", "Outras")
                subt = st.session_state["patient"]["surgical"].get("subtipo")
                risk = "Baixo"
                if porte == "Médio":
                    risk = "Intermediário"
                if porte in ("Grande", "Especial"):
                    risk = "Alto"
                if cat == "Cardíaca" or incisao == "Intratorácica" or (cat == "Vascular" and subt == "Suprainguinal"):
                    risk = "Alto"
                if cat == "Abdominal" and subt and subt.startswith("Alta") and risk == "Baixo":
                    risk = "Intermediário"
                st.session_state["patient"]["surgical"]["risco_cirurgico"] = risk
                if risk == "Baixo":
                    st.success("Risco cirúrgico: Baixo (<1% mortalidade)")
                elif risk == "Intermediário":
                    st.info("Risco cirúrgico: Intermediário (1–5% mortalidade)")
                else:
                    st.warning("Risco cirúrgico: Alto (>5% mortalidade)")

            st.markdown("---")
            # AVALIAÇÃO FUNCIONAL
            st.markdown("**AVALIAÇÃO FUNCIONAL**")
            f1, f2, f3 = st.columns(3)
            with f1:
                st.session_state["patient"]["functional"]["nsqip_status"] = st.radio(
                    "Status funcional (NSQIP)",
                    options=["Independente", "Parcialmente dependente", "Totalmente dependente"],
                    index=["Independente", "Parcialmente dependente", "Totalmente dependente"].index(st.session_state["patient"]["functional"].get("nsqip_status", "Independente")),
                    help="Capacidade basal para AVDs conforme NSQIP",
                )
            with f2:
                st.session_state["patient"]["functional"]["sobe_escadas_sem_parar"] = st.radio(
                    "Sobe 2 lances de escada sem parar?",
                    options=["Sim", "Não"],
                    index=["Sim", "Não"].index(st.session_state["patient"]["functional"].get("sobe_escadas_sem_parar", "Sim")),
                    help="Indicador prático de capacidade funcional",
                )
            with f3:
                st.session_state["patient"]["functional"]["avd_independencia"] = st.radio(
                    "Independência em AVDs",
                    options=["Independente", "Dependência parcial", "Dependência total"],
                    index=["Independente", "Dependência parcial", "Dependência total"].index(st.session_state["patient"]["functional"].get("avd_independencia", "Independente")),
                    help="Grau de independência nas atividades diárias",
                )
            # Avisos rápidos
            if st.session_state["patient"]["functional"]["nsqip_status"] != "Independente":
                st.info("Status funcional não-independente aumenta risco pós-operatório.")
            if st.session_state["patient"]["functional"].get("sobe_escadas_sem_parar") == "Não":
                st.warning("Baixa tolerância ao esforço (escadas).")

            st.markdown("---")
            # DADOS ESPECÍFICOS PARA SCORES
            st.markdown("**DADOS ESPECÍFICOS PARA SCORES**")
            if cat == "Cardíaca":
                ak1, ak2 = st.columns(2)
                with ak1:
                    st.session_state.setdefault("scores_specific", {})
                    st.session_state["scores_specific"].setdefault("akics", {})
                    st.session_state["scores_specific"]["akics"]["tipo_cardiaco"] = st.selectbox(
                        "AKICS - Tipo (cardíaca)",
                        options=["Coronariana", "Valvar", "Combinada"],
                        index=["Coronariana", "Valvar", "Combinada"].index(
                            st.session_state["scores_specific"]["akics"].get("tipo_cardiaco", "Coronariana")
                        ),
                        help="Tipo de cirurgia cardíaca",
                    )
                with ak2:
                    st.session_state["scores_specific"]["akics"]["funcao_ventricular"] = st.selectbox(
                        "AKICS - Função ventricular",
                        options=["Normal", "Disfunção leve", "Disfunção moderada", "Disfunção grave"],
                        index=0,
                        help="Classificação clínica/ecocardiográfica",
                    )
            pr1, pr2, pr3, pr4 = st.columns(4)
            with pr1:
                st.session_state.setdefault("scores_specific", {})
                st.session_state["scores_specific"].setdefault("pre_deliric", {})
                st.session_state["scores_specific"]["pre_deliric"]["apache_ii"] = st.number_input(
                    "PRE-DELIRIC - APACHE II",
                    min_value=0.0, max_value=71.0, step=0.5,
                    value=float(st.session_state["scores_specific"]["pre_deliric"].get("apache_ii", 0.0)),
                    help="Informe se disponível",
                )
            with pr2:
                st.session_state["scores_specific"]["pre_deliric"]["coma"] = st.checkbox(
                    "PRE-DELIRIC - Coma/não responsivo",
                    value=bool(st.session_state["scores_specific"]["pre_deliric"].get("coma", False)),
                )
            with pr3:
                st.session_state["scores_specific"]["pre_deliric"]["infeccao_ativa"] = st.checkbox(
                    "PRE-DELIRIC - Infecção ativa",
                    value=bool(st.session_state["scores_specific"]["pre_deliric"].get("infeccao_ativa", False)),
                )
            with pr4:
                st.session_state["scores_specific"]["pre_deliric"]["grupo_admissao"] = st.selectbox(
                    "PRE-DELIRIC - Grupo de admissão",
                    options=["Clínico", "Cirúrgico", "Trauma", "Neuro"],
                    index=["Clínico", "Cirúrgico", "Trauma", "Neuro"].index(
                        st.session_state["scores_specific"]["pre_deliric"].get("grupo_admissao", "Cirúrgico")
                    ),
                )

            st.markdown("---")
            # SINAIS VITAIS PRÉ-OPERATÓRIOS
            st.markdown("**SINAIS VITAIS PRÉ-OPERATÓRIOS**")
            v1, v2, v3, v4 = st.columns([1, 1, 1, 1])
            with v1:
                st.session_state["patient"]["physical_exam"]["spo2_ar_ambiente"] = st.number_input(
                    "SpO₂ em AA (%)",
                    min_value=50.0,
                    max_value=100.0,
                    step=0.1,
                    value=float(st.session_state["patient"]["physical_exam"].get("spo2_ar_ambiente", 98.0)),
                    help="Saturação de O₂ em ar ambiente",
                )
            with v2:
                st.session_state["patient"]["physical_exam"]["pa_sistolica"] = st.number_input(
                    "PA Sistólica (mmHg)",
                    min_value=50,
                    max_value=260,
                    value=int(st.session_state["patient"]["physical_exam"].get("pa_sistolica", 120)),
                )
            with v3:
                st.session_state["patient"]["physical_exam"]["pa_diastolica"] = st.number_input(
                    "PA Diastólica (mmHg)",
                    min_value=30,
                    max_value=160,
                    value=int(st.session_state["patient"]["physical_exam"].get("pa_diastolica", 80)),
                )
            with v4:
                st.session_state["patient"]["physical_exam"]["fc"] = st.number_input(
                    "Frequência Cardíaca (bpm)",
                    min_value=30,
                    max_value=220,
                    value=int(st.session_state["patient"]["physical_exam"].get("fc", 75)),
                )

            # Vital validations
            pa = st.session_state["patient"]["physical_exam"]
            if pa["pa_sistolica"] < 90 or pa["pa_sistolica"] > 180 or pa["pa_diastolica"] < 50 or pa["pa_diastolica"] > 120:
                st.warning("PA fora de faixas usuais; reavaliar antes do procedimento.")
            if pa["fc"] < 40 or pa["fc"] > 120:
                st.warning("Frequência cardíaca incomum; considerar avaliação adicional.")
            if pa["spo2_ar_ambiente"] < 92:
                st.info("SpO₂ < 92%: considerar oxigenação e investigação de causa.")

            st.markdown("</div>", unsafe_allow_html=True)


    def _show_risk_calculators() -> None:
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Cálculo de Riscos")

            tabs = st.tabs(["RCRI", "ARISCAT", "STOP-Bang"])

            with tabs[0]:
                r_col1, r_col2, r_col3 = st.columns(3)
                with r_col1:
                    high_risk_surgery = st.checkbox("Cirurgia de alto risco")
                    history_ischemic_heart_disease = st.checkbox("Doença cardíaca isquêmica")
                with r_col2:
                    history_congestive_heart_failure = st.checkbox("Insuficiência cardíaca")
                    history_cerebrovascular_disease = st.checkbox("Doença cerebrovascular")
                with r_col3:
                    insulin_therapy_diabetes = st.checkbox("Diabetes em insulina")
                    preop_creat_gt_2 = st.checkbox("Creatinina > 2 mg/dL")

                rcri_result = calculate_rcri(
                    high_risk_surgery=high_risk_surgery,
                    history_ischemic_heart_disease=history_ischemic_heart_disease,
                    history_congestive_heart_failure=history_congestive_heart_failure,
                    history_cerebrovascular_disease=history_cerebrovascular_disease,
                    insulin_therapy_diabetes=insulin_therapy_diabetes,
                    preoperative_creatinine_gt_2mg_dl=preop_creat_gt_2,
                )
                st.session_state["results"]["rcri"] = rcri_result
                st.success(f"RCRI: {rcri_result.score} (Risco {rcri_result.risk_category})")

            with tabs[1]:
                a1, a2, a3 = st.columns(3)
                with a1:
                    age_51_80 = st.checkbox("Idade 51-80")
                    age_gt_80 = st.checkbox("Idade > 80")
                    resp_inf = st.checkbox("Infecção respiratória < 1 mês")
                with a2:
                    low_spo2 = st.checkbox("SpO2 91–95%")
                    very_low_spo2 = st.checkbox("SpO2 ≤ 90%")
                    anemia = st.checkbox("Anemia")
                with a3:
                    surg_upper = st.checkbox("Cirurgia abdome superior")
                    surg_intrath = st.checkbox("Cirurgia intratorácica")
                    dur_2_3 = st.checkbox("Duração 2–3h")
                    dur_gt_3 = st.checkbox("Duração > 3h")
                    emerg = st.checkbox("Cirurgia de emergência")

                ariscat_score, ariscat_risk, ariscat_details = calculate_ariscat(
                    age_51_80=age_51_80,
                    age_gt_80=age_gt_80,
                    low_spo2=low_spo2,
                    very_low_spo2=very_low_spo2,
                    resp_infection_last_month=resp_inf,
                    anemia=anemia,
                    surgery_upper_abdominal=surg_upper,
                    surgery_intrathoracic=surg_intrath,
                    duration_2_to_3h=dur_2_3,
                    duration_gt_3h=dur_gt_3,
                    emergency_surgery=emerg,
                )
                st.session_state["results"]["ariscat"] = {
                    "score": ariscat_score,
                    "risk": ariscat_risk,
                    "details": ariscat_details,
                }
                st.success(f"ARISCAT: {ariscat_score} (Risco {ariscat_risk})")

            with tabs[2]:
                s1, s2, s3, s4 = st.columns(4)
                with s1:
                    snoring = st.checkbox("Ronco")
                    tired = st.checkbox("Cansaço diurno")
                with s2:
                    observed_apnea = st.checkbox("Apneia observada")
                    high_bp = st.checkbox("Hipertensão")
                with s3:
                    bmi_over_35 = st.checkbox("IMC > 35")
                    age_over_50 = st.checkbox("Idade > 50")
                with s4:
                    neck_circ_over_40 = st.checkbox("Circunf. pescoço > 40cm")
                    male = st.checkbox("Masculino")

                stopbang_result = calculate_stopbang(
                    snoring=snoring,
                    tired=tired,
                    observed_apnea=observed_apnea,
                    high_bp=high_bp,
                    bmi_over_35=bmi_over_35,
                    age_over_50=age_over_50,
                    neck_circ_over_40cm=neck_circ_over_40,
                    male=male,
                )
                st.session_state["results"]["stopbang"] = stopbang_result
                st.success(f"STOP-Bang: {stopbang_result.score} (Risco {stopbang_result.risk_category})")

            st.markdown("</div>", unsafe_allow_html=True)


    def _show_report_section() -> None:
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Relatório")

            # Resumo dos escores
            rcri = st.session_state["results"].get("rcri")
            ariscat = st.session_state["results"].get("ariscat")
            stopbang = st.session_state["results"].get("stopbang")

            st.write("Resumo atual dos escores:")
            colA, colB, colC = st.columns(3)
            with colA:
                if rcri:
                    st.metric("RCRI", rcri.score, rcri.risk_category)
                else:
                    st.info("RCRI não calculado")
            with colB:
                if ariscat:
                    st.metric("ARISCAT", ariscat["score"], ariscat["risk"])
                else:
                    st.info("ARISCAT não calculado")
            with colC:
                if stopbang:
                    st.metric("STOP-Bang", stopbang.score, stopbang.risk_category)
                else:
                    st.info("STOP-Bang não calculado")

            _show_interactive_visualizations()

            st.markdown("---")
            st.subheader("Resumo por IA (opcional)")
            custom_prompt = st.text_area(
                "Contexto clínico",
                "Paciente candidato a procedimento cirúrgico. Analise escores e descreva riscos e condutas.",
            )
            if st.button("Gerar resumo por IA"):
                patient = st.session_state["patient"]
                prompt = (
                    f"Nome: {patient['demographics']['nome']}, Idade: {patient['demographics']['idade']}, Sexo: {patient['demographics']['sexo']}.\n"
                    f"RCRI: {rcri.score if rcri else 'NA'} ({rcri.risk_category if rcri else 'NA'}).\n"
                    f"ARISCAT: {ariscat['score'] if ariscat else 'NA'} ({ariscat['risk'] if ariscat else 'NA'}).\n"
                    f"STOP-Bang: {stopbang.score if stopbang else 'NA'} ({stopbang.risk_category if stopbang else 'NA'}).\n\n"
                    f"Contexto: {custom_prompt}"
                )
                tmp_cfg = config
                ai_text = generate_recommendations(prompt, tmp_cfg)
                if ai_text:
                    st.session_state["ai_summary"] = ai_text
                    st.success("Resumo gerado.")
                    st.write(ai_text)
                else:
                    st.info("IA indisponível. Verifique a GOOGLE_API_KEY no .env.")

            st.markdown("---")
            st.subheader("Exportar PDF")
            file_name = st.text_input("Nome do arquivo", value="relatorio_paciente.pdf")
            if st.button("Gerar PDF"):
                patient = st.session_state["patient"]
                output_dir = Path(config.reports_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / file_name
                pdf_path = build_pdf_report(
                    output_path=output_path,
                    patient_info={"Nome": patient["demographics"]["nome"], "Idade": str(patient["demographics"]["idade"]), "Sexo": patient["demographics"]["sexo"]},
                    rcri={
                        "score": rcri.score if rcri else None,
                        "risk": rcri.risk_category if rcri else None,
                        "details": getattr(rcri, "details", {}),
                    },
                    ariscat=ariscat or {},
                    stopbang={
                        "score": stopbang.score if stopbang else None,
                        "risk": stopbang.risk_category if stopbang else None,
                        "details": getattr(stopbang, "details", {}),
                    },
                    ai_summary=st.session_state.get("ai_summary"),
                )
                with open(pdf_path, "rb") as f:
                    st.download_button("Baixar PDF", data=f, file_name=file_name, mime="application/pdf")
                st.success(f"PDF gerado em: {pdf_path}")

            st.markdown("</div>", unsafe_allow_html=True)


    def _show_interactive_visualizations() -> None:
        st.subheader("Visualizações Interativas")

        patient = st.session_state["patient"]
        demo = patient.get("demographics", {})
        surgical = patient.get("surgical", {})
        labs = patient.get("labs", {})
        comorb = patient.get("comorbidities", {})
        functional = patient.get("functional", {})

        rcri = st.session_state["results"].get("rcri")
        ariscat = st.session_state["results"].get("ariscat")

        # ASA
        try:
            asa_out = classify_asa(asa_class=str(demo.get("asa", "II")), emergency_modifier=bool(demo.get("asa_emergencia", False)))
        except Exception:
            asa_out = None

        # NSQIP (proxy)
        try:
            nsqip_out = nsqip_proxy(
                idade=int(demo.get("idade", 60)),
                sexo=str(demo.get("sexo", "Feminino")),
                status_funcional=str(functional.get("nsqip_status", "Independente")),
                emergencia=(str(surgical.get("urgencia", "Eletiva")) != "Eletiva"),
                asa=str(demo.get("asa", "II")),
                diabetes=bool(comorb.get("diabetes_tipo_1") or comorb.get("diabetes_tipo_2")),
                hipertensao=bool(comorb.get("hipertensao")),
                dpoc=bool(comorb.get("dpoc")),
                insuficiencia_cardiaca=bool(comorb.get("insuficiencia_cardiaca")),
                procedimento=f"{surgical.get('tipo_cirurgia','')} {surgical.get('subtipo','')}",
                hematocrito=float(labs.get("hematocrito", 0.0)),
                creatinina=float(labs.get("creatinina", 0.0)),
                albumina=float(labs.get("albumina", 0.0)),
                plaquetas=float(labs.get("plaquetas", 0.0)),
            )
            nsqip_res = nsqip_out.result
        except Exception:
            nsqip_res = {}

        # RCRI completo (% e label)
        try:
            rcri_details = getattr(rcri, "details", {}) if rcri else {}
            rcri_full = rcri_score_full(
                high_risk_surgery=bool(rcri_details.get("high_risk_surgery")) or (surgical.get("incisao_site") == "Intratorácica") or (surgical.get("tipo_cirurgia") == "Cardíaca") or (surgical.get("tipo_cirurgia") == "Vascular" and surgical.get("subtipo") == "Suprainguinal"),
                ischemic_heart_disease=bool(rcri_details.get("history_ischemic_heart_disease")) or bool(comorb.get("doenca_cardiaca_isquemica")),
                congestive_heart_failure=bool(rcri_details.get("history_congestive_heart_failure")) or bool(comorb.get("insuficiencia_cardiaca")),
                cerebrovascular_disease=bool(rcri_details.get("history_cerebrovascular_disease")) or bool(comorb.get("doenca_cerebrovascular")),
                insulin_treated_diabetes=bool(rcri_details.get("insulin_therapy_diabetes")) or bool(comorb.get("uso_insulina")),
                creatinine_gt_2mg_dl=bool(rcri_details.get("preoperative_creatinine_gt_2mg_dl")) or (float(labs.get("creatinina", 0.0)) > 2.0),
            )
            rcri_pct = rcri_full.result.get("risk_percent")
            rcri_class_text = f"{rcri_full.result.get('class','')} - {rcri_full.result.get('risk_category','')}"
        except Exception:
            rcri_pct = None
            rcri_class_text = ""

        # ARISCAT completo
        try:
            ariscat_full = ariscat_score_full(
                age_51_80=(int(demo.get("idade", 0)) >= 51 and int(demo.get("idade", 0)) <= 80),
                age_gt_80=(int(demo.get("idade", 0)) > 80),
                spo2_le_95=(float(patient.get("physical_exam", {}).get("spo2_ar_ambiente", 100.0)) <= 95.0),
                resp_infection_last_month=bool(comorb.get("infeccao_respiratoria_mes")),
                anemia_hb_le_10=(float(labs.get("hemoglobina", 100.0)) <= 10.0),
                incision_abd_upper=(surgical.get("incisao_site") == "Abdome superior"),
                incision_intrathoracic=(surgical.get("incisao_site") == "Intratorácica"),
                duration_2_to_3h=(surgical.get("duracao_cat") == "2-3h"),
                duration_gt_3h=(surgical.get("duracao_cat") == ">3h"),
                emergency_surgery=(str(surgical.get("urgencia", "Eletiva")) != "Eletiva"),
            )
            ariscat_pct = ariscat_full.result.get("probability_cpp_percent")
            ariscat_cat_text = ariscat_full.result.get("risk_category", "")
        except Exception:
            ariscat_pct = None
            ariscat_cat_text = ""

        # AKICS
        try:
            tipo = surgical.get("tipo_cirurgia", "Outras")
            if tipo == "Cardíaca":
                tipo_map = {"Coronariana": "coronariana", "Valvar": "valvar", "Combinada": "combinada"}
                tipo_card = tipo_map.get(surgical.get("subtipo", "Coronariana"), "coronariana")
                akics_out = akics_score(
                    idade=int(demo.get("idade", 60)),
                    sexo_feminino=(str(demo.get("sexo", "Feminino")).lower() == "feminino"),
                    insuficiencia_cardiaca=bool(comorb.get("insuficiencia_cardiaca")),
                    hipertensao=bool(comorb.get("hipertensao")),
                    emergencia=(str(surgical.get("urgencia", "Eletiva")) != "Eletiva"),
                    tipo_cirurgia=tipo_card,
                    creatinina_mg_dl=float(labs.get("creatinina", 0.0)),
                )
            else:
                comp = {"Pequeno": "baixa", "Médio": "media", "Grande": "alta", "Especial": "alta"}.get(surgical.get("porte", "Médio"), "media")
                akics_out = akics_score(
                    idade=int(demo.get("idade", 60)),
                    sexo_feminino=(str(demo.get("sexo", "Feminino")).lower() == "feminino"),
                    insuficiencia_cardiaca=bool(comorb.get("insuficiencia_cardiaca")),
                    hipertensao=bool(comorb.get("hipertensao")),
                    emergencia=(str(surgical.get("urgencia", "Eletiva")) != "Eletiva"),
                    tipo_cirurgia="nao_cardiaca",
                    creatinina_mg_dl=float(labs.get("creatinina", 0.0)),
                    nao_cardiaca_complexidade=comp,
                )
            akics_pct = akics_out.result.get("probabilidade_percentual")
            akics_cat = akics_out.result.get("categoria_risco", "")
        except Exception:
            akics_pct = None
            akics_cat = ""

        # PRE-DELIRIC
        try:
            pre_spec = st.session_state.get("scores_specific", {}).get("pre_deliric", {})
            pred_out = pre_deliric_score(
                idade=int(demo.get("idade", 60)),
                apache_ii=float(pre_spec.get("apache_ii", 0.0)),
                grupo_admissao=str(pre_spec.get("grupo_admissao", "Cirúrgico")),
                coma=bool(pre_spec.get("coma", False)),
                infeccao=bool(pre_spec.get("infeccao_ativa", False)),
                ph=float(labs.get("ph", 7.4)),
                hco3=float(labs.get("hco3", 24.0)),
                sedativos=bool(patient.get("medications", {}).get("classes", {}).get("sedativos_benzos", False)),
                morfina=bool(patient.get("medications", {}).get("classes", {}).get("opioides", False)),
                ureia_mg_dl=float(labs.get("ureia", 0.0)),
                creatinina_mg_dl=float(labs.get("creatinina", 0.0)),
            )
            pred_pct = pred_out.result.get("probabilidade_percentual")
            pred_cat = pred_out.result.get("categoria_risco", "")
        except Exception:
            pred_pct = None
            pred_cat = ""

        # Normalizações
        def asa_to_pct(asa_label: str) -> float:
            map_pct = {"I": 5.0, "II": 10.0, "III": 25.0, "IV": 50.0, "V": 75.0, "VI": 90.0}
            return map_pct.get(asa_label, 10.0)

        asa_pct = asa_to_pct(str(demo.get("asa", "II")))
        ns_mort = float(nsqip_res.get("mortality_30d_pct", 0.0)) if nsqip_res else 0.0
        rcri_pct = float(rcri_pct) if rcri_pct is not None else 0.0
        ariscat_pct = float(ariscat_pct) if ariscat_pct is not None else 0.0
        akics_pct = float(akics_pct) if akics_pct is not None else 0.0
        pred_pct = float(pred_pct) if pred_pct is not None else 0.0

        # Radar
        radar_categories = ["Mortalidade (NSQIP)", "Cardíaco (RCRI)", "Pulmonar (ARISCAT)", "Renal (AKICS)", "Delirium (PRE-DELIRIC)", "Geral (ASA)"]
        radar_values = [ns_mort, rcri_pct, ariscat_pct, akics_pct, pred_pct, asa_pct]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=radar_values, theta=radar_categories, fill='toself', name='Riscos (%)'))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False)
        st.plotly_chart(fig_radar, use_container_width=True)

        # Barras
        def risk_color(pct: float) -> str:
            if pct >= 35:
                return "#dc2626"
            if pct >= 10:
                return "#f59e0b"
            return "#16a34a"
        bars = []
        rcri_class_text = f"{rcri_full.result.get('class','')} - {rcri_full.result.get('risk_category','')}" if 'rcri_full' in locals() else "RCRI"
        ariscat_cat_text = ariscat_full.result.get("risk_category", "") if 'ariscat_full' in locals() else ""
        akics_cat = akics_out.result.get("categoria_risco", "") if 'akics_out' in locals() else ""
        pred_cat = pred_out.result.get("categoria_risco", "") if 'pred_out' in locals() else ""
        bars.append({"Escore": "NSQIP Mort.", "Valor": ns_mort, "Categoria": "-", "Label": f"NSQIP Mortalidade {ns_mort:.1f}%", "color": risk_color(ns_mort)})
        bars.append({"Escore": "RCRI", "Valor": rcri_pct, "Categoria": rcri_class_text or "-", "Label": f"{rcri_class_text or 'RCRI'} ({rcri_pct:.1f}%)", "color": risk_color(rcri_pct)})
        bars.append({"Escore": "ARISCAT", "Valor": ariscat_pct, "Categoria": ariscat_cat_text or "-", "Label": f"ARISCAT {ariscat_cat_text} ({ariscat_pct:.1f}%)", "color": risk_color(ariscat_pct)})
        bars.append({"Escore": "AKICS", "Valor": akics_pct, "Categoria": akics_cat or "-", "Label": f"AKICS {akics_cat} ({akics_pct:.1f}%)", "color": risk_color(akics_pct)})
        bars.append({"Escore": "PRE-DELIRIC", "Valor": pred_pct, "Categoria": pred_cat or "-", "Label": f"Delirium {pred_cat} ({pred_pct:.1f}%)", "color": risk_color(pred_pct)})
        bars_fig = go.Figure(go.Bar(x=[b["Escore"] for b in bars], y=[b["Valor"] for b in bars], text=[b["Label"] for b in bars], marker_color=[b["color"] for b in bars], textposition='outside'))
        bars_fig.update_yaxes(title_text="%", range=[0, max(40, max([b["Valor"] for b in bars] + [10]) * 1.2)])
        st.plotly_chart(bars_fig, use_container_width=True)

        # Gauges
        st.markdown("### Indicadores por Sistema")
        g1, g2, g3, g4 = st.columns(4)
        ns_card = float(nsqip_res.get("cardiac_complication_pct", rcri_pct)) if nsqip_res else rcri_pct
        cv_val = max(0.0, min(100.0, (rcri_pct + ns_card) / 2.0))
        with g1:
            fig_cv = go.Figure(go.Indicator(mode="gauge+number", value=cv_val, title={"text": "Cardio %"}, gauge={"axis": {"range": [0, 100]}, "steps": [{"range": [0, 10], "color": "#dcfce7"}, {"range": [10, 35], "color": "#fef9c3"}, {"range": [35, 100], "color": "#fee2e2"}]},))
            st.plotly_chart(fig_cv, use_container_width=True)
        ns_pulm = float(nsqip_res.get("pneumonia_pct", ariscat_pct)) if nsqip_res else ariscat_pct
        pulm_val = max(0.0, min(100.0, (ariscat_pct + ns_pulm) / 2.0))
        with g2:
            fig_pulm = go.Figure(go.Indicator(mode="gauge+number", value=pulm_val, title={"text": "Pulmonar %"}, gauge={"axis": {"range": [0, 100]}, "steps": [{"range": [0, 10], "color": "#dcfce7"}, {"range": [10, 35], "color": "#fef9c3"}, {"range": [35, 100], "color": "#fee2e2"}]},))
            st.plotly_chart(fig_pulm, use_container_width=True)
        ns_renal = float(nsqip_res.get("renal_failure_pct", akics_pct)) if nsqip_res else akics_pct
        renal_val = max(0.0, min(100.0, (akics_pct + ns_renal) / 2.0))
        with g3:
            fig_renal = go.Figure(go.Indicator(mode="gauge+number", value=renal_val, title={"text": "Renal %"}, gauge={"axis": {"range": [0, 100]}, "steps": [{"range": [0, 10], "color": "#dcfce7"}, {"range": [10, 35], "color": "#fef9c3"}, {"range": [35, 100], "color": "#fee2e2"}]},))
            st.plotly_chart(fig_renal, use_container_width=True)
        with g4:
            fig_neuro = go.Figure(go.Indicator(mode="gauge+number", value=pred_pct, title={"text": "Delirium %"}, gauge={"axis": {"range": [0, 100]}, "steps": [{"range": [0, 10], "color": "#dcfce7"}, {"range": [10, 35], "color": "#fef9c3"}, {"range": [35, 100], "color": "#fee2e2"}]},))
            st.plotly_chart(fig_neuro, use_container_width=True)

        # Tabela interativa
        st.markdown("### Tabela Interativa de Escores")
        table_rows = []
        if asa_out:
            table_rows.append({"Escore": "ASA", "Pontuação": demo.get("asa"), "Categoria": asa_out.result.get("risk", ""), "Risco %": asa_pct, "Interpretação": asa_out.result.get("description", "")})
        if nsqip_res:
            table_rows.append({"Escore": "NSQIP Mortalidade", "Pontuação": "-", "Categoria": "-", "Risco %": ns_mort, "Interpretação": "Probabilidade de mortalidade 30d"})
        if rcri_pct is not None:
            table_rows.append({"Escore": "RCRI", "Pontuação": (rcri.score if rcri else "-"), "Categoria": rcri_class_text, "Risco %": rcri_pct, "Interpretação": "Risco cardíaco (RCRI)"})
        if ariscat_pct is not None:
            table_rows.append({"Escore": "ARISCAT", "Pontuação": (ariscat["score"] if ariscat else "-"), "Categoria": ariscat_cat_text, "Risco %": ariscat_pct, "Interpretação": "Risco de complicações pulmonares"})
        if akics_pct is not None:
            table_rows.append({"Escore": "AKICS", "Pontuação": "-", "Categoria": akics_cat, "Risco %": akics_pct, "Interpretação": "Risco de IRA pós-operatória"})
        if pred_pct is not None:
            table_rows.append({"Escore": "PRE-DELIRIC", "Pontuação": "-", "Categoria": pred_cat, "Risco %": pred_pct, "Interpretação": "Risco de delirium em UTI"})
        import pandas as pd
        df_scores = pd.DataFrame(table_rows)
        filter_choice = st.selectbox("Filtrar por categoria de risco", options=["Todos", "Baixo", "Intermediário", "Alto", "Muito baixo", "Muito alto"]) 
        if filter_choice != "Todos":
            df_scores = df_scores[df_scores["Categoria"].astype(str).str.contains(filter_choice, case=False, na=False)]
        df_scores = df_scores.sort_values(by=["Risco %"], ascending=False, na_position="last")
        st.dataframe(df_scores, use_container_width=True)

        # Heatmap de fatores
        st.markdown("### Heatmap de Fatores de Risco")
        def factor_matrix() -> "pd.DataFrame":
            import pandas as pd
            rows: Dict[str, Dict[str, float]] = {}
            if rcri:
                for k, v in (getattr(rcri, "details", {}) or {}).items():
                    rows.setdefault(k, {})["RCRI"] = float(v)
            if ariscat:
                for k, v in (ariscat.get("details", {}) or {}).items():
                    rows.setdefault(k, {})["ARISCAT"] = float(v)
            if 'akics_out' in locals() and akics_out:
                for k, v in (akics_out.result.get("detalhes", {}) or {}).items():
                    rows.setdefault(k, {})["AKICS"] = float(v)
            if 'pred_out' in locals() and pred_out:
                for k, v in (pred_out.result.get("detalhes", {}) or {}).items():
                    rows.setdefault(k, {})["PRE-DELIRIC"] = float(v)
            if not rows:
                return pd.DataFrame()
            return pd.DataFrame(rows).T.fillna(0.0)
        df_heat = factor_matrix()
        if not df_heat.empty:
            fig_heat = px.imshow(df_heat, color_continuous_scale="Blues", aspect="auto")
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("Fatores insuficientes para heatmap.")

        # Timeline de recomendações (IA)
        st.markdown("### Timeline de Recomendações (Medicações)")
        ai_meds = st.session_state.get("ai_meds") or {}
        susp_list = ai_meds.get("suspender", []) if isinstance(ai_meds, dict) else []
        import re
        items = []
        for s in susp_list:
            m = re.search(r"(\d+)\s*(d|dia|dias|h|hora|horas)", str(s), flags=re.I)
            days = 0
            if m:
                val = int(m.group(1))
                unit = m.group(2).lower()
                if unit.startswith("h"):
                    days = max(0, val / 24)
                else:
                    days = val
            items.append({"Medicação": s, "DiasAntes": days})
        if items:
            import pandas as pd
            from datetime import datetime
            df_tl = pd.DataFrame(items)
            now = datetime.now()
            df_tl["start"] = now - pd.to_timedelta((df_tl["DiasAntes"] * 24).astype(int), unit="h")
            df_tl["finish"] = now
            fig_tl = px.timeline(df_tl, x_start="start", x_end="finish", y="Medicação", color="DiasAntes", color_continuous_scale="Bluered")
            fig_tl.update_layout(showlegend=False)
            st.plotly_chart(fig_tl, use_container_width=True)
        else:
            st.info("Sem recomendações de suspensão de medicações para timeline.")


# --------- Main Layout (Tabs/Sections) ---------
if section == "Dados do Paciente":
    _show_patient_form()
elif section == "Cálculo de Riscos":
    _show_risk_calculators()
else:
    _show_report_section()

# --------- Footer ---------
st.markdown(
    """
    <div class="footer">
        Desenvolvido por HelpAnest • Feito com Streamlit • © 2025
    </div>
    """,
    unsafe_allow_html=True,
)
