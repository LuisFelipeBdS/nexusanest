from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Tuple

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image,
)
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart


PRIMARY_COLOR = colors.HexColor("#0B5FA5")
PRIMARY_DARK = colors.HexColor("#094b82")
ACCENT_GREEN = colors.HexColor("#16a34a")
ACCENT_YELLOW = colors.HexColor("#f59e0b")
ACCENT_RED = colors.HexColor("#dc2626")


class ReportGenerator:
    def __init__(
        self,
        *,
        title: str = "Relatório de Avaliação Perioperatória",
        author: str = "HelpAnest",
        institution: str = "Instituição",
        logo_path: str | Path = "assets/logo.png",
    ) -> None:
        self.title = title
        self.author = author
        self.institution = institution
        self.logo_path = Path(logo_path)
        self.styles = getSampleStyleSheet()
        self.styles.add(
            ParagraphStyle(
                name="Section",
                parent=self.styles["Heading2"],
                textColor=PRIMARY_DARK,
                spaceBefore=12,
                spaceAfter=6,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="Small",
                parent=self.styles["BodyText"],
                fontSize=9,
                leading=11,
            )
        )

    # ---------------------- Header / Footer ----------------------
    def _header_footer(self, canvas, doc) -> None:  # type: ignore[no-untyped-def]
        canvas.saveState()
        width, height = A4
        header_h = 1.2 * cm
        canvas.setFillColor(PRIMARY_COLOR)
        canvas.rect(0, height - header_h, width, header_h, fill=1, stroke=0)

        x = 1 * cm
        y = height - 0.9 * cm
        if self.logo_path.exists():
            try:
                canvas.drawImage(str(self.logo_path), x, height - 1.1 * cm, width=1.0 * cm, height=1.0 * cm, preserveAspectRatio=True, mask='auto')
                x += 1.2 * cm
            except Exception:
                pass
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 11)
        canvas.drawString(x, y, self.institution)
        canvas.setFont("Helvetica", 10)
        canvas.drawRightString(width - 1 * cm, y, self.title)

        # Footer with page number
        canvas.setFillColor(colors.grey)
        canvas.setFont("Helvetica", 9)
        canvas.drawRightString(width - 1 * cm, 0.75 * cm, f"Página {doc.page}")
        canvas.restoreState()

    # ---------------------- Helpers ----------------------
    def _risk_color(self, category: str | None) -> colors.Color:
        if not category:
            return colors.black
        cat = category.lower()
        if "alto" in cat:
            return ACCENT_RED
        if "inter" in cat:
            return ACCENT_YELLOW
        if "baixo" in cat or "muito baixo" in cat:
            return ACCENT_GREEN
        return colors.black

    def _scores_table(self, scores: Dict[str, Any]) -> Table:
        data: List[List[Any]] = [["Escore", "Resultado", "Interpretação/Risco"]]
        # ASA
        asa = scores.get("asa") or {}
        data.append([
            "ASA",
            asa.get("label") or asa.get("asa") or "-",
            (asa.get("description") or "") + (" | Risco: " + str(asa.get("risk")) if asa.get("risk") else ""),
        ])
        # RCRI
        rcri = scores.get("rcri") or {}
        if rcri:
            rc_text = f"{rcri.get('class','')} ({rcri.get('risk_percent','-')}%)"
            data.append(["RCRI", str(rcri.get("score", "-")), rc_text])
        # ARISCAT
        ariscat = scores.get("ariscat") or {}
        if ariscat:
            ar_text = f"{ariscat.get('risk_category','')} ({ariscat.get('probability_cpp_percent','-')}%)"
            data.append(["ARISCAT", str(ariscat.get("score", "-")), ar_text])
        # NSQIP (proxy) - mostrar mortalidade e LOS
        nsqip = scores.get("nsqip") or {}
        if nsqip:
            ns_text = f"Mort: {nsqip.get('mortality_30d_pct','-')}% | LOS: {nsqip.get('length_of_stay_days','-')}d"
            data.append(["NSQIP (proxy)", "-", ns_text])
        # AKICS
        akics = scores.get("akics") or {}
        if akics:
            data.append(["AKICS", str(akics.get("pontuacao_total", "-")), f"{akics.get('categoria_risco','')} ({akics.get('probabilidade_percentual','-')}%)"])
        # PRE-DELIRIC
        pred = scores.get("pre_deliric") or {}
        if pred:
            data.append(["PRE-DELIRIC", str(pred.get("pontuacao_total", "-")), f"{pred.get('categoria_risco','')} ({pred.get('probabilidade_percentual','-')}%)"])

        table = Table(data, colWidths=[4.2 * cm, 3.0 * cm, 9.8 * cm])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                    ("TEXTCOLOR", (0, 0), (-1, 0), PRIMARY_DARK),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.lightgrey),
                    ("FONT", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONT", (0, 1), (-1, -1), "Helvetica"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.Color(0.98, 0.99, 1.0)]),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        return table

    def _risk_barchart(self, risks: List[Tuple[str, float]]) -> Drawing:
        # Build a simple vertical bar chart for risk percentages
        labels = [k for k, _ in risks]
        values = [float(v) for _, v in risks]
        max_val = max(values + [10.0])

        drawing = Drawing(400, 180)
        bc = VerticalBarChart()
        bc.x = 40
        bc.y = 30
        bc.height = 120
        bc.width = 320
        bc.data = [values]
        bc.strokeColor = colors.lightgrey
        bc.valueAxis.valueMin = 0
        bc.valueAxis.valueMax = max(10, int(max_val * 1.2))
        bc.valueAxis.valueStep = max(5, int(max_val / 5) or 1)
        bc.categoryAxis.categoryNames = labels
        bc.categoryAxis.labels.boxAnchor = 'ne'
        bc.categoryAxis.labels.angle = 20
        bc.barWidth = 14
        bc.groupSpacing = 6
        # Color by threshold
        fills = []
        for v in values:
            if v >= 35:
                fills.append(ACCENT_RED)
            elif v >= 10:
                fills.append(ACCENT_YELLOW)
            else:
                fills.append(ACCENT_GREEN)
        bc.bars[0].fillColor = colors.white  # default
        for i, color in enumerate(fills):
            bc.bars[0].fillColor = colors.white  # reset
            try:
                bc.bars[0][i].fillColor = color
            except Exception:
                pass
        drawing.add(bc)
        return drawing

    # ---------------------- Public API ----------------------
    def build(
        self,
        *,
        output_path: Path,
        patient: Dict[str, Any],
        scores: Dict[str, Any],
        ai_general: Dict[str, Any] | None = None,
        ai_meds: Dict[str, Any] | None = None,
        references: List[str] | None = None,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            topMargin=2.2 * cm,
            bottomMargin=2.0 * cm,
            leftMargin=1.8 * cm,
            rightMargin=1.8 * cm,
            title=self.title,
            author=self.author,
        )

        story: List[Any] = []

        # Header section (title on page)
        story.append(Paragraph(self.title, self.styles["Title"]))
        story.append(Paragraph(self.institution, self.styles["Small"]))
        story.append(Spacer(1, 8))

        # Patient header
        story.append(Paragraph("Dados do Paciente", self.styles["Section"]))
        demo = patient.get("demographics", {})
        surgical = patient.get("surgical", {})
        patient_table = Table(
            [
                ["Nome", demo.get("nome", "-"), "Idade", demo.get("idade", "-"), "Sexo", demo.get("sexo", "-")],
                ["IMC", demo.get("imc", "-"), "ASA", f"{demo.get('asa','-')}{' -E' if demo.get('asa_emergencia') else ''}", "Urgência", surgical.get("urgencia", "-")],
                ["Cirurgia", surgical.get("tipo_cirurgia", "-"), "Subtipo", surgical.get("subtipo", "-"), "Anestesia", surgical.get("anestesia_planejada", "-")],
            ],
            colWidths=[2.2 * cm, 4.3 * cm, 2.0 * cm, 3.1 * cm, 2.0 * cm, 3.4 * cm],
        )
        patient_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONT", (0, 0), (-1, -1), "Helvetica"),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        story.append(patient_table)
        story.append(Spacer(1, 12))

        # Executive summary (AI general)
        story.append(Paragraph("Resumo Executivo dos Riscos", self.styles["Section"]))
        resumo_exec = (ai_general or {}).get("resumo_executivo") or "-"
        story.append(Paragraph(str(resumo_exec), self.styles["BodyText"]))
        story.append(Spacer(1, 8))

        # By systems
        story.append(Paragraph("Análise por Sistemas", self.styles["Section"]))
        por_sist = (ai_general or {}).get("por_sistemas") or {}
        for sist in ("cardiovascular", "pulmonar", "renal", "delirium"):
            items = por_sist.get(sist) or []
            story.append(Paragraph(sist.capitalize(), self.styles["Heading3"]))
            if items:
                for it in items:
                    story.append(Paragraph(f"• {it}", self.styles["BodyText"]))
            else:
                story.append(Paragraph("• -", self.styles["BodyText"]))
            story.append(Spacer(1, 4))

        # Scores table
        story.append(Spacer(1, 6))
        story.append(Paragraph("Escores Calculados", self.styles["Section"]))
        story.append(self._scores_table(scores))

        # Risk bar chart (collect key percents)
        risks: List[Tuple[str, float]] = []
        rcri = scores.get("rcri") or {}
        if rcri.get("risk_percent") is not None:
            risks.append(("RCRI%", float(rcri["risk_percent"])) )
        ariscat = scores.get("ariscat") or {}
        if ariscat.get("probability_cpp_percent") is not None:
            risks.append(("ARISCAT%", float(ariscat["probability_cpp_percent"])) )
        nsqip = scores.get("nsqip") or {}
        if nsqip.get("mortality_30d_pct") is not None:
            risks.append(("NSQIP Mort%", float(nsqip["mortality_30d_pct"])) )
        pred = scores.get("pre_deliric") or {}
        if pred.get("probabilidade_percentual") is not None:
            risks.append(("Delirium%", float(pred["probabilidade_percentual"])) )
        if risks:
            story.append(Spacer(1, 6))
            story.append(Paragraph("Riscos Percentuais (Resumo)", self.styles["Heading3"]))
            story.append(self._risk_barchart(risks))

        story.append(PageBreak())

        # IA detailed
        story.append(Paragraph("Análise da IA (Estruturada)", self.styles["Section"]))
        if ai_general:
            recs = ai_general.get("recomendacoes") or []
            story.append(Paragraph("Recomendações", self.styles["Heading3"]))
            for r in recs:
                story.append(Paragraph(f"• {r}", self.styles["BodyText"]))
            story.append(Spacer(1, 6))
            mon = ai_general.get("monitorizacao") or []
            story.append(Paragraph("Monitorização sugerida", self.styles["Heading3"]))
            for m in mon:
                story.append(Paragraph(f"• {m}", self.styles["BodyText"]))
        else:
            story.append(Paragraph("IA não disponível.", self.styles["BodyText"]))

        # Medications recommendations
        story.append(Spacer(1, 10))
        story.append(Paragraph("Recomendações de Medicações", self.styles["Section"]))
        meds = ai_meds or {}
        meds_table = Table(
            [
                ["Suspender", "Manter", "Ajustar"],
                ["\n".join(meds.get("suspender", [])) or "-", "\n".join(meds.get("manter", [])) or "-", "\n".join(meds.get("ajustar", [])) or "-"],
            ],
            colWidths=[6.0 * cm, 5.0 * cm, 6.0 * cm],
        )
        meds_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONT", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(meds_table)

        # Bibliography
        story.append(Spacer(1, 12))
        story.append(Paragraph("Bibliografia / Referências", self.styles["Section"]))
        refs: List[str] = list(references or [])
        # Gather from scores if present
        def _refs_from(obj: Any) -> List[str]:
            if isinstance(obj, dict):
                cand = obj.get("references") or obj.get("referencias")
                if isinstance(cand, (list, tuple)):
                    return [str(x) for x in cand]
            return []
        for key in ["asa", "rcri", "ariscat", "nsqip", "akics", "pre_deliric"]:
            refs.extend(_refs_from(scores.get(key) or {}))
        seen = set()
        for ref in refs:
            if ref and ref not in seen:
                story.append(Paragraph(f"- {ref}", self.styles["Small"]))
                seen.add(ref)

        doc.build(story, onFirstPage=self._header_footer, onLaterPages=self._header_footer)
        return output_path


def export_with_timestamp(
    *,
    base_dir: Path,
    base_name: str = "relatorio_perioperatorio",
    patient_name: str | None = None,
    patient: Dict[str, Any],
    scores: Dict[str, Any],
    ai_general: Dict[str, Any] | None = None,
    ai_meds: Dict[str, Any] | None = None,
    references: List[str] | None = None,
) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = (patient_name or patient.get("demographics", {}).get("nome") or "paciente").strip() or "paciente"
    safe_name = "".join(c for c in safe_name if c.isalnum() or c in ("_", "-"))
    outfile = base_dir / f"{base_name}_{safe_name}_{ts}.pdf"
    gen = ReportGenerator()
    return gen.build(
        output_path=outfile,
        patient=patient,
        scores=scores,
        ai_general=ai_general,
        ai_meds=ai_meds,
        references=references,
    )
