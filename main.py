from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import streamlit as st

from src.config import load_config
from src.risk_scores import calculate_rcri, calculate_ariscat, calculate_stopbang
from src.ai_assistant import generate_recommendations
from src.reporting import build_pdf_report


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
        },
        "comorbidities": {
            "hipertensao": False,
            "diabetes_tipo_1": False,
            "diabetes_tipo_2": False,
            "cardiopatia_isquemica": False,
            "insuficiencia_cardiaca": False,
            "arritmias": False,
            "valvulopatias": False,
            "dpoc": False,
            "asma": False,
            "pneumopatia_restritiva": False,
            "insuficiencia_renal": False,
            "hepatopatia": False,
            "avc_previo": False,
            "demencia": False,
            "depressao": False,
        },
        "medications": {
            "list_text": "",
            "classes": {
                "anticoagulantes": False,
                "antiagregantes": False,
                "betabloqueadores": False,
                "ieca_bra": False,
                "diureticos": False,
                "estatinas": False,
                "insulina": False,
                "antidiabeticos_orais": False,
            },
        },
        "labs": {
            "hemoglobina": 0.0,
            "hematocrito": 0.0,
            "leucocitos": 0.0,
            "plaquetas": 0.0,
            "creatinina": 0.0,
            "ureia": 0.0,
            "tfg": 0.0,
            "glicemia": 0.0,
            "hba1c": 0.0,
            "albumina": 0.0,
            "proteinas_totais": 0.0,
        },
    }
if "results" not in st.session_state:
    st.session_state["results"] = {}
if "ai_summary" not in st.session_state:
    st.session_state["ai_summary"] = None
if "reports_dir" not in st.session_state:
    st.session_state["reports_dir"] = config.reports_dir

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
    st.subheader("Configurações")
    api_key_input = st.text_input("Google API Key", value=config.google_api_key or "", type="password")
    model_name = st.text_input("Modelo Gemini", value=config.default_model)
    reports_dir = st.text_input("Pasta de relatórios", value=config.reports_dir)
    if st.button("Salvar sessão"):
        st.session_state["api_key"] = api_key_input.strip() or None
        st.session_state["model_name"] = model_name.strip()
        st.session_state["reports_dir"] = reports_dir.strip() or config.reports_dir
        st.success("Configurações salvas na sessão.")

# --------- Helpers ---------
def _show_patient_form() -> None:
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Dados do Paciente")

        # DADOS DEMOGRÁFICOS
        st.markdown("**DADOS DEMOGRÁFICOS**")
        d1, d2, d3, d4 = st.columns([2, 1, 1, 1])
        with d1:
            st.session_state["patient"]["demographics"]["nome"] = st.text_input(
                "Nome", st.session_state["patient"]["demographics"].get("nome", "")
            )
        with d2:
            st.session_state["patient"]["demographics"]["idade"] = st.number_input(
                "Idade", min_value=0, max_value=120, value=int(st.session_state["patient"]["demographics"].get("idade", 60))
            )
        with d3:
            st.session_state["patient"]["demographics"]["sexo"] = st.selectbox(
                "Sexo", ["Feminino", "Masculino"],
                index=0 if st.session_state["patient"]["demographics"].get("sexo") == "Feminino" else 1
            )
        with d4:
            st.session_state["patient"]["demographics"]["asa"] = st.radio(
                "ASA Physical Status", options=["I", "II", "III", "IV", "V", "VI"],
                index=["I", "II", "III", "IV", "V", "VI"].index(st.session_state["patient"]["demographics"].get("asa", "II"))
            )

        d5, d6, d7 = st.columns(3)
        with d5:
            peso = st.number_input(
                "Peso (kg)", min_value=20.0, max_value=300.0,
                value=float(st.session_state["patient"]["demographics"].get("peso_kg", 70.0)), step=0.1
            )
            st.session_state["patient"]["demographics"]["peso_kg"] = peso
        with d6:
            altura_cm = st.number_input(
                "Altura (cm)", min_value=100.0, max_value=250.0,
                value=float(st.session_state["patient"]["demographics"].get("altura_cm", 170.0)), step=0.1
            )
            st.session_state["patient"]["demographics"]["altura_cm"] = altura_cm
        with d7:
            altura_m = max(altura_cm / 100.0, 0.5)
            imc = round(peso / (altura_m ** 2), 2)
            st.session_state["patient"]["demographics"]["imc"] = imc
            st.metric("IMC", imc)

        # Quick validations
        if st.session_state["patient"]["demographics"]["idade"] > 110:
            st.warning("Idade acima do esperado; verifique o valor informado.")
        if imc < 16 or imc > 50:
            st.info("IMC fora da faixa usual; considerar avaliação nutricional.")

        st.markdown("---")
        # COMORBIDADES
        st.markdown("**COMORBIDADES**")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.session_state["patient"]["comorbidities"]["hipertensao"] = st.checkbox("Hipertensão", value=st.session_state["patient"]["comorbidities"]["hipertensao"])        
            st.session_state["patient"]["comorbidities"]["diabetes_tipo_1"] = st.checkbox("Diabetes tipo 1", value=st.session_state["patient"]["comorbidities"]["diabetes_tipo_1"])        
            st.session_state["patient"]["comorbidities"]["diabetes_tipo_2"] = st.checkbox("Diabetes tipo 2", value=st.session_state["patient"]["comorbidities"]["diabetes_tipo_2"])        
            st.session_state["patient"]["comorbidities"]["cardiopatia_isquemica"] = st.checkbox("Cardiopatia isquêmica", value=st.session_state["patient"]["comorbidities"]["cardiopatia_isquemica"])        
        with c2:
            st.session_state["patient"]["comorbidities"]["insuficiencia_cardiaca"] = st.checkbox("Insuficiência cardíaca", value=st.session_state["patient"]["comorbidities"]["insuficiencia_cardiaca"])        
            st.session_state["patient"]["comorbidities"]["arritmias"] = st.checkbox("Arritmias", value=st.session_state["patient"]["comorbidities"]["arritmias"])        
            st.session_state["patient"]["comorbidities"]["valvulopatias"] = st.checkbox("Valvulopatias", value=st.session_state["patient"]["comorbidities"]["valvulopatias"])        
            st.session_state["patient"]["comorbidities"]["dpoc"] = st.checkbox("DPOC", value=st.session_state["patient"]["comorbidities"]["dpoc"])        
        with c3:
            st.session_state["patient"]["comorbidities"]["asma"] = st.checkbox("Asma", value=st.session_state["patient"]["comorbidities"]["asma"])        
            st.session_state["patient"]["comorbidities"]["pneumopatia_restritiva"] = st.checkbox("Pneumopatias restritivas", value=st.session_state["patient"]["comorbidities"]["pneumopatia_restritiva"])        
            st.session_state["patient"]["comorbidities"]["insuficiencia_renal"] = st.checkbox("Insuficiência renal", value=st.session_state["patient"]["comorbidities"]["insuficiencia_renal"])        
            st.session_state["patient"]["comorbidities"]["hepatopatia"] = st.checkbox("Hepatopatias", value=st.session_state["patient"]["comorbidities"]["hepatopatia"])        
        c4, c5 = st.columns(2)
        with c4:
            st.session_state["patient"]["comorbidities"]["avc_previo"] = st.checkbox("AVC prévio", value=st.session_state["patient"]["comorbidities"]["avc_previo"])        
            st.session_state["patient"]["comorbidities"]["demencia"] = st.checkbox("Demência", value=st.session_state["patient"]["comorbidities"]["demencia"])        
        with c5:
            st.session_state["patient"]["comorbidities"]["depressao"] = st.checkbox("Depressão", value=st.session_state["patient"]["comorbidities"]["depressao"])        

        st.markdown("---")
        # MEDICAÇÕES EM USO
        st.markdown("**MEDICAÇÕES EM USO**")
        st.session_state["patient"]["medications"]["list_text"] = st.text_area(
            "Liste as medicações em uso (nome e dose)",
            value=st.session_state["patient"]["medications"].get("list_text", ""),
            placeholder="Ex.: AAS 100 mg/dia; Losartana 50 mg 12/12h; Metformina 850 mg 8/8h"
        )
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.session_state["patient"]["medications"]["classes"]["anticoagulantes"] = st.checkbox("Anticoagulantes", value=st.session_state["patient"]["medications"]["classes"]["anticoagulantes"])        
            st.session_state["patient"]["medications"]["classes"]["antiagregantes"] = st.checkbox("Antiagregantes", value=st.session_state["patient"]["medications"]["classes"]["antiagregantes"])        
        with m2:
            st.session_state["patient"]["medications"]["classes"]["betabloqueadores"] = st.checkbox("Betabloqueadores", value=st.session_state["patient"]["medications"]["classes"]["betabloqueadores"])        
            st.session_state["patient"]["medications"]["classes"]["ieca_bra"] = st.checkbox("IECA/BRA", value=st.session_state["patient"]["medications"]["classes"]["ieca_bra"])        
        with m3:
            st.session_state["patient"]["medications"]["classes"]["diureticos"] = st.checkbox("Diuréticos", value=st.session_state["patient"]["medications"]["classes"]["diureticos"])        
            st.session_state["patient"]["medications"]["classes"]["estatinas"] = st.checkbox("Estatinas", value=st.session_state["patient"]["medications"]["classes"]["estatinas"])        
        with m4:
            st.session_state["patient"]["medications"]["classes"]["insulina"] = st.checkbox("Insulina", value=st.session_state["patient"]["medications"]["classes"]["insulina"])        
            st.session_state["patient"]["medications"]["classes"]["antidiabeticos_orais"] = st.checkbox("Antidiabéticos orais", value=st.session_state["patient"]["medications"]["classes"]["antidiabeticos_orais"])        

        st.markdown("---")
        # EXAMES LABORATORIAIS
        st.markdown("**EXAMES LABORATORIAIS**")
        l1, l2, l3 = st.columns(3)
        with l1:
            st.session_state["patient"]["labs"]["hemoglobina"] = st.number_input("Hemoglobina (g/dL)", min_value=0.0, max_value=25.0, step=0.1, value=float(st.session_state["patient"]["labs"]["hemoglobina"]))
            st.session_state["patient"]["labs"]["hematocrito"] = st.number_input("Hematócrito (%)", min_value=0.0, max_value=70.0, step=0.1, value=float(st.session_state["patient"]["labs"]["hematocrito"]))
            st.session_state["patient"]["labs"]["leucocitos"] = st.number_input("Leucócitos (10³/µL)", min_value=0.0, max_value=200.0, step=0.1, value=float(st.session_state["patient"]["labs"]["leucocitos"]))
            st.session_state["patient"]["labs"]["plaquetas"] = st.number_input("Plaquetas (10³/µL)", min_value=0.0, max_value=1500.0, step=1.0, value=float(st.session_state["patient"]["labs"]["plaquetas"]))
        with l2:
            st.session_state["patient"]["labs"]["creatinina"] = st.number_input("Creatinina (mg/dL)", min_value=0.0, max_value=20.0, step=0.1, value=float(st.session_state["patient"]["labs"]["creatinina"]))
            st.session_state["patient"]["labs"]["ureia"] = st.number_input("Ureia (mg/dL)", min_value=0.0, max_value=300.0, step=1.0, value=float(st.session_state["patient"]["labs"]["ureia"]))
            st.session_state["patient"]["labs"]["tfg"] = st.number_input("TFG (mL/min/1.73m²)", min_value=0.0, max_value=200.0, step=1.0, value=float(st.session_state["patient"]["labs"]["tfg"]))
        with l3:
            st.session_state["patient"]["labs"]["glicemia"] = st.number_input("Glicemia (mg/dL)", min_value=0.0, max_value=1000.0, step=1.0, value=float(st.session_state["patient"]["labs"]["glicemia"]))
            st.session_state["patient"]["labs"]["hba1c"] = st.number_input("HbA1c (%)", min_value=0.0, max_value=25.0, step=0.1, value=float(st.session_state["patient"]["labs"]["hba1c"]))
            st.session_state["patient"]["labs"]["albumina"] = st.number_input("Albumina (g/dL)", min_value=0.0, max_value=10.0, step=0.1, value=float(st.session_state["patient"]["labs"]["albumina"]))
            st.session_state["patient"]["labs"]["proteinas_totais"] = st.number_input("Proteínas totais (g/dL)", min_value=0.0, max_value=12.0, step=0.1, value=float(st.session_state["patient"]["labs"]["proteinas_totais"]))

        # Simple range hints
        labs = st.session_state["patient"]["labs"]
        if labs["hemoglobina"] and (labs["hemoglobina"] < 8 or labs["hemoglobina"] > 18):
            st.warning("Hemoglobina fora da faixa comum (8–18 g/dL).")
        if labs["creatinina"] and labs["creatinina"] > 1.5:
            st.info("Creatinina elevada; considerar classificação de risco renal.")

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
            tmp_cfg = config.model_copy(update={
                "google_api_key": st.session_state.get("api_key") or config.google_api_key,
                "default_model": st.session_state.get("model_name") or config.default_model,
            })
            ai_text = generate_recommendations(prompt, tmp_cfg)
            if ai_text:
                st.session_state["ai_summary"] = ai_text
                st.success("Resumo gerado.")
                st.write(ai_text)
            else:
                st.info("Configure a Google API Key na barra lateral para habilitar IA.")

        st.markdown("---")
        st.subheader("Exportar PDF")
        file_name = st.text_input("Nome do arquivo", value="relatorio_paciente.pdf")
        if st.button("Gerar PDF"):
            patient = st.session_state["patient"]
            output_dir = Path(st.session_state.get("reports_dir", config.reports_dir))
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
