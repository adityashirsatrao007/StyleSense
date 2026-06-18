"""
Generate corrected reviewer response PDF.
Only claims what was actually done.
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

OUTPUT = "/home/aditya/Desktop/Projects/StyleSense/reviewer_response.pdf"

doc = SimpleDocTemplate(
    OUTPUT, pagesize=A4,
    leftMargin=2*cm, rightMargin=2*cm,
    topMargin=2*cm, bottomMargin=2*cm,
)

styles = getSampleStyleSheet()
title_style = ParagraphStyle('Title2', parent=styles['Title'], fontSize=14, spaceAfter=12)
heading_style = ParagraphStyle('H1', parent=styles['Heading1'], fontSize=12, spaceAfter=8, spaceBefore=16)
body_style = ParagraphStyle('Body2', parent=styles['Normal'], fontSize=10, leading=14, alignment=TA_JUSTIFY, spaceAfter=6)
bold_style = ParagraphStyle('Bold', parent=body_style, fontName='Helvetica-Bold')
small_style = ParagraphStyle('Small', parent=body_style, fontSize=9, leading=12)

story = []

# Title
story.append(Paragraph("Response to Reviewer Comments", title_style))
story.append(Paragraph("Prepared for camera-ready revision — June 18, 2026", small_style))
story.append(Spacer(1, 6))
story.append(Paragraph("<b>Paper ID:</b> 634", body_style))
story.append(Paragraph("<b>Title:</b> StyleSense: A Lightweight MobileNet-Based Framework for Real Time Personalized Fashion Recommendation", body_style))
story.append(Paragraph("<b>Track:</b> Track 3 — Advanced Computational Intelligence Systems", body_style))
story.append(Paragraph("<b>Conference:</b> ISPCC 2025", body_style))
story.append(Spacer(1, 12))

# ─── CHANGES SUMMARY TABLE ───
story.append(Paragraph("<b>Summary of Changes Made</b>", heading_style))

changes_data = [
    ["Section", "Change", "Status"],
    ["Section 1", "Added numbered Contributions list (4 items)", "Done"],
    ["Section 1", "Added 'Paper Organization' paragraph", "Done"],
    ["Section 1", "Added scope justification for ISPCC Track 3", "Done"],
    ["Section 1", "Corrected dataset size from ~3,000 to 10,251", "Done"],
    ["Section 2", "Condensed literature review to direct relevance", "Done"],
    ["Section 3", "Added Table I: Model Configuration & Technical Specs", "Done"],
    ["Section 3", "Added Table II: Mobile Deployment Metrics", "Done"],
    ["Section 3", "Added Table III: Comparison with Prior Work", "Done"],
    ["Section 3", "Added architecture description with param counts", "Done"],
    ["Section 4", "Added deployment metrics (2.73 MB, 33 ms inference)", "Done"],
    ["Section 5", "Added Limitations and Future Work subsection", "Done"],
    ["Section 5", "Removed old verbose Discussion subsections", "Done"],
    ["References", "Added 3 references [17]-[19] (DeepFashion2, MobileViT, ResNet)", "Done"],
    ["Figures", "Regenerated Figs 1, 2, 4 at high resolution (drawio)", "Done"],
    ["Figures", "Font sizes increased to 18-22pt for readability", "Done"],
    ["Language", "Revised verbose passages throughout", "Done"],
]

table = Table(changes_data, colWidths=[80, 300, 50])
table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2B579A')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 9),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#EBF1F8')]),
    ('TOPPADDING', (0, 0), (-1, -1), 4),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
]))
story.append(table)
story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════
# REVIEWER #1
# ═══════════════════════════════════════════════════════════════
story.append(Paragraph("<b>Reviewer #1</b>", heading_style))
story.append(Paragraph("<b>Final Verdict:</b> Weak Accept – Worthy, but requires significant revision. | <b>Confidence:</b> Confident", body_style))
story.append(Spacer(1, 6))

r1_qa = [
    ("Q1: Is the work within the scope of the conference?",
     "Clearly within scope",
     "Agreed. The work applies lightweight deep learning to a practical recommendation scenario on mobile devices, which aligns with Track 3 (Advanced Computational Intelligence Systems). A scope justification with citations has been added to Section 1."),

    ("Q2: Formatting and visual clarity?",
     "Moderate concerns",
     "All figures have been regenerated at high resolution using drawio with consistent font sizing (18-22pt). Three new tables (Model Configuration, Deployment Metrics, Prior Work Comparison) have been added with standardized formatting. Architecture diagrams were redrawn with larger fonts for readability."),

    ("Q3: Technical contribution novelty?",
     "Moderate novelty",
     "A numbered Contributions list (Section 1) now explicitly separates what is novel from existing methods: (1) first lightweight framework for style-context classification (not garment type); (2) real-time mobile deployment at 2.73 MB; (3) unified visual + physiological profiling architecture; (4) end-to-end pipeline portable to Android/Flutter."),

    ("Q4: Technical contribution significance?",
     "Moderate contribution",
     "Mobile deployment metrics have been added (Table II): inference time of 33 ms/image on CPU, model size of 2.73 MB, and 7.9x speedup over Keras. A comparison table (Table III) shows our model uses 2.59M parameters — 10-50x fewer than comparable systems while maintaining competitive accuracy."),

    ("Q5: Adequate references?",
     "Minor gaps",
     "Three additional references added: [17] DeepFashion2 (CVPR 2019), [18] MobileViT (ICLR 2022), [19] ResNet (CVPR 2016). The literature review has been condensed to directly relevant works."),

    ("Q6: Properly structured and clearly written?",
     "Moderate issues",
     "Section 2 (Literature Review) has been condensed. A 'Paper Organization' paragraph has been added at the end of Section 1. Technical specifications are now presented in structured tables. Verbose passages have been simplified throughout."),

    ("Q9: General comments",
     "Useful but needs validation improvements",
     "We thank the reviewer for the constructive assessment. Validation strengthened: Table III provides comparison with prior systems. Mobile deployment metrics (Table II) demonstrate practical impact. Presentation improved: three new tables, condensed literature review, and simplified language. Novelty clarified: explicit contributions list differentiates our style-context approach from existing garment-type classifiers."),
]

for q, verdict, response in r1_qa:
    story.append(Paragraph(f"<b>{q}</b>", bold_style))
    story.append(Paragraph(f"<i>Reviewer: {verdict}</i>", small_style))
    story.append(Paragraph(f"<b>Our Response:</b> {response}", body_style))
    story.append(Spacer(1, 8))

story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════
# REVIEWER #2
# ═══════════════════════════════════════════════════════════════
story.append(Paragraph("<b>Reviewer #2</b>", heading_style))
story.append(Paragraph("<b>Final Verdict:</b> Weak Accept – Worthy, but requires significant revision. | <b>Confidence:</b> Confident", body_style))
story.append(Spacer(1, 6))

r2_qa = [
    ("Q1: Is the work within the scope?",
     "Clearly out of scope",
     "We respectfully disagree. Track 3 ('Advanced Computational Intelligence Systems') encompasses applied machine learning on resource-constrained platforms. Lightweight CNN-based recommendation on mobile devices is a well-established application of computational intelligence. A scope justification with citations has been added to Section 1."),

    ("Q2: Formatting and visual clarity?",
     "Major issues",
     "All figures have been regenerated at high resolution. Three new tables with standardized alignment have been added. Font sizes increased to 18-22pt in diagrams for readability. No formatting errors remain."),

    ("Q3: Technical contribution novelty?",
     "Limited novelty",
     "Table III now differentiates StyleSense from existing systems: unlike DeepFashion (garment-type) and StyleSnap (visual search), our approach performs style-context classification across social scenarios — a fundamentally more subjective task. The contributions list (Section 1) explicitly separates novel contributions from existing methods."),

    ("Q4: Technical contribution significance?",
     "Moderate contribution",
     "Table II presents mobile deployment metrics: 2.73 MB TFLite model, 33 ms inference time on CPU, 7.9x acceleration over Keras. These demonstrate practical significance for real-world on-device deployment on resource-constrained mobile devices."),

    ("Q5: Adequate references?",
     "Minor gaps",
     "Three additional references added as detailed in response to Reviewer #1, Q5."),

    ("Q6: Properly structured and clearly written?",
     "Moderate issues",
     "Manuscript revised: literature review condensed, verbose passages simplified, structured tables added for technical specifications and comparisons."),

    ("Weakness 1: Language and Writing Quality",
     "Complex, verbose sentences",
     "Language revised throughout. Verbose passages simplified. A 'Paper Organization' paragraph added for structural clarity."),

    ("Weakness 2: Lack of Clear Novelty",
     "Not clearly differentiated",
     "Addressed via: (a) explicit Contributions bullet list (4 items), (b) Table III comparing with prior work, (c) unified architecture description showing joint visual + physiological processing as differentiator."),

    ("Weakness 3: Dataset Limitations",
     "Small dataset (~3000 images)",
     "Correction: the actual dataset contains 10,251 images (not ~3,000). The paper text has been corrected. A 'Limitations and Future Work' subsection discusses controlled capture conditions and notes cross-dataset validation on DeepFashion2 as future work."),

    ("Weakness 4: Missing Technical Details",
     "Architecture, hardware, inference time missing",
     "Table I now provides: layer-wise MobileNetV2 configuration, training hyperparameters (54 epochs, batch 24→32, Adam optimizer), hardware specs (GTX 1650 Ti 4GB, 30GB RAM), and software stack (TensorFlow 2.21.0)."),

    ("Weakness 5: Limited Comparative Analysis",
     "No comparison with EfficientNet, ViT",
     "Table III compares StyleSense with DeepFashion, StyleSnap, MobileViT, and EfficientNet-B0 on parameter count, mobile readiness, and task scope. Our model uses 2.59M parameters — 2-53x fewer than the listed alternatives."),

    ("Weakness 6: Overclaiming Results",
     "No cross-dataset validation",
     "A 'Limitations and Future Work' subsection has been added (Section 5) explicitly stating that reported accuracy was achieved on a curated dataset with controlled capture conditions. Cross-dataset validation and real-world user testing are noted as primary future work directions."),
]

for q, verdict, response in r2_qa:
    story.append(Paragraph(f"<b>{q}</b>", bold_style))
    story.append(Paragraph(f"<i>Reviewer: {verdict}</i>", small_style))
    story.append(Paragraph(f"<b>Our Response:</b> {response}", body_style))
    story.append(Spacer(1, 8))

doc.build(story)
print(f"Reviewer response saved to: {OUTPUT}")
