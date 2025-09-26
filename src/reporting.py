from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer


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

    story.append(Paragraph("RCRI", styles["Heading2"]))
    story.append(Paragraph(str(rcri), styles["Code"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("ARISCAT", styles["Heading2"]))
    story.append(Paragraph(str(ariscat), styles["Code"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("STOP-Bang", styles["Heading2"]))
    story.append(Paragraph(str(stopbang), styles["Code"]))
    story.append(Spacer(1, 12))

    if ai_summary:
        story.append(Paragraph("Resumo IA", styles["Heading2"]))
        story.append(Paragraph(ai_summary.replace("\n", "<br/>"), styles["BodyText"]))

    doc.build(story)
    return output_path
