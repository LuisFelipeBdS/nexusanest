from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, List

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


def _format_score_section(title: str, data: Dict) -> List:
    styles = getSampleStyleSheet()
    elems: List = [Paragraph(title, styles["Heading2"])]
    rows = []
    # Flatten common fields
    if "score" in data:
        rows.append(["Pontuação", str(data.get("score"))])
    if "class" in data:
        rows.append(["Classe", str(data.get("class"))])
    if "risk" in data:
        rows.append(["Categoria", str(data.get("risk"))])
    if "risk_category" in data:
        rows.append(["Categoria", str(data.get("risk_category"))])
    if "risk_percent" in data:
        rows.append(["Risco %", f"{data.get('risk_percent')}%"])
    if "probability_cpp_percent" in data:
        rows.append(["Prob. CPP %", f"{data.get('probability_cpp_percent')}%"])
    if "recommendations" in data:
        rows.append(["Recomendações", str(data.get("recommendations"))])

    details = data.get("details") or data.get("detalhes") or {}
    if isinstance(details, dict) and details:
        for k, v in details.items():
            rows.append([k.replace("_", " ").capitalize(), str(v)])

    if not rows:
        elems.append(Paragraph("-", styles["BodyText"]))
        return elems + [Spacer(1, 8)]

    table = Table(rows, colWidths=[5 * cm, 10 * cm])
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
            ]
        )
    )
    elems.append(table)
    elems.append(Spacer(1, 8))
    return elems


def _markdown_like_to_flowable(text: str) -> List:
    styles = getSampleStyleSheet()
    out: List = []
    if not text:
        return out
    # Simple conversions: headings, bold/italics, bullets
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            out.append(Spacer(1, 4))
            continue
        if line.startswith("### "):
            out.append(Paragraph(line[4:], styles["Heading3"]))
            continue
        if line.startswith("## "):
            out.append(Paragraph(line[3:], styles["Heading2"]))
            continue
        if line.startswith("# "):
            out.append(Paragraph(line[2:], styles["Heading1"]))
            continue
        if line.startswith("- ") or line.startswith("* ") or line.startswith("• "):
            out.append(Paragraph(f"• {line[2:]}", styles["BodyText"]))
            continue
        # inline bold/italics
        line = line.replace("**", "<b>", 1).replace("**", "</b>", 1)
        line = line.replace("*", "<i>", 1).replace("*", "</i>", 1)
        out.append(Paragraph(line, styles["BodyText"]))
    return out


def build_pdf_report(
    output_path: Path,
    patient_info: Dict[str, str],
    rcri: Dict,
    ariscat: Dict,
    stopbang: Dict,
    ai_summary: Optional[str] = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(output_path), pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()

    story = []

    story.append(Paragraph("Relatório de Risco Perioperatório", styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Dados do Paciente", styles["Heading2"]))
    for k, v in patient_info.items():
        story.append(Paragraph(f"<b>{k}:</b> {v}", styles["BodyText"]))
    story.append(Spacer(1, 12))

    # RCRI / ARISCAT / STOP-Bang com formatação legível
    story += _format_score_section("RCRI", rcri or {})
    story += _format_score_section("ARISCAT", ariscat or {})
    story += _format_score_section("STOP-Bang", stopbang or {})

    if ai_summary:
        story.append(Paragraph("Resumo IA", styles["Heading2"]))
        story += _markdown_like_to_flowable(ai_summary)

    doc.build(story)
    return output_path
