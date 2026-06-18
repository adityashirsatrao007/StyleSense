from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
import os
from datetime import datetime

OUTPUT = "/home/aditya/Desktop/Projects/StyleSense/reviewer_response.pdf"

doc = SimpleDocTemplate(
    OUTPUT, pagesize=A4,
    topMargin=0.8*inch, bottomMargin=0.8*inch,
    leftMargin=1*inch, rightMargin=1*inch,
)

styles = getSampleStyleSheet()

GRAY = HexColor("#636e72")
DARK = HexColor("#2d3436")
NAVY = HexColor("#1a1a2e")
LGRAY = HexColor("#dfe6e9")

styles.add(ParagraphStyle("DocTitle", parent=styles["Title"], fontSize=18, spaceAfter=6,
    textColor=NAVY, fontName="Helvetica-Bold"))
styles.add(ParagraphStyle("DocSubtitle", parent=styles["Normal"], fontSize=10, spaceAfter=20,
    textColor=HexColor("#555555"), fontName="Helvetica"))
styles.add(ParagraphStyle("SecH", parent=styles["Heading1"], fontSize=14, spaceBefore=18,
    spaceAfter=8, textColor=NAVY, fontName="Helvetica-Bold"))
styles.add(ParagraphStyle("RevH", parent=styles["Heading2"], fontSize=12, spaceBefore=14,
    spaceAfter=6, textColor=DARK, fontName="Helvetica-Bold"))
styles.add(ParagraphStyle("QText", parent=styles["Normal"], fontSize=9.5, spaceBefore=8,
    spaceAfter=4, textColor=GRAY, fontName="Helvetica-Oblique", leading=13))
styles.add(ParagraphStyle("AText", parent=styles["Normal"], fontSize=9.5, spaceBefore=2,
    spaceAfter=10, textColor=black, fontName="Helvetica", leading=13,
    leftIndent=12, alignment=TA_JUSTIFY))
styles.add(ParagraphStyle("VerText", parent=styles["Normal"], fontSize=10, spaceBefore=10,
    spaceAfter=4, textColor=DARK, fontName="Helvetica-Bold", leading=14))
styles.add(ParagraphStyle("Body", parent=styles["Normal"], fontSize=9.5, leading=13,
    alignment=TA_JUSTIFY))
styles.add(ParagraphStyle("Label", parent=styles["Normal"], fontSize=9.5, textColor=DARK,
    fontName="Helvetica-Bold", leading=13))
styles.add(ParagraphStyle("RevComment", parent=styles["Normal"], fontSize=9.5, spaceBefore=2,
    spaceAfter=6, textColor=HexColor("#2d3436"), fontName="Helvetica", leading=13,
    leftIndent=12, alignment=TA_JUSTIFY))

PAPER_INFO = [
    ("Paper ID", "634"),
    ("Title", "StyleSense: A Lightweight MobileNet-Based Framework for Real Time Personalized Fashion Recommendation"),
    ("Track", "Track 3 \u2014 Advanced Computational Intelligence Systems"),
    ("Conference", "ISPCC 2025"),
]

def header_table():
    data = [[Paragraph(f"<b>{k}:</b>", styles["Label"]),
             Paragraph(v, styles["Body"])] for k, v in PAPER_INFO]
    t = Table(data, colWidths=[1.8*inch, 4.7*inch])
    t.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING", (0,0), (-1,-1), 1),
        ("BOTTOMPADDING", (0,0), (-1,-1), 1),
    ]))
    return t

def hr():
    return HRFlowable(width="100%", thickness=0.5, color=LGRAY, spaceBefore=6, spaceAfter=6)

def rating_stars(v):
    """Map 'scale' text to visual rating."""
    return Paragraph(f"<b>Rating:</b> {v}", styles["VerText"])

def qa_block(q_text, a_text):
    """Exact reviewer question then answer."""
    return [
        Paragraph(f"<b>Reviewer\u2019s Comment:</b>", styles["Label"]),
        Paragraph(q_text, styles["QText"]),
        Paragraph("<b>Our Response:</b>", styles["Label"]),
        Paragraph(a_text, styles["AText"]),
    ]

def r1_verdict():
    return Paragraph(
        "<b>Final Verdict:</b> Weak Accept \u2013 Worthy, but requires significant revision. &nbsp;&nbsp;|&nbsp;&nbsp; "
        "<b>Confidence:</b> Confident",
        styles["VerText"]
    )

def r2_verdict():
    return Paragraph(
        "<b>Final Verdict:</b> Weak Accept \u2013 Worthy, but requires significant revision. &nbsp;&nbsp;|&nbsp;&nbsp; "
        "<b>Confidence:</b> Confident",
        styles["VerText"]
    )

elements = []

# ── Title ──
elements.append(Paragraph("Response to Reviewer Comments", styles["DocTitle"]))
elements.append(Paragraph(
    f"Prepared for camera-ready revision \u2014 {datetime.now().strftime('%B %d, %Y')}",
    styles["DocSubtitle"]
))
elements.append(header_table())
elements.append(hr())

# ═══════════════════════════════════════════════════
#  REVIEWER #1
# ═══════════════════════════════════════════════════
elements.append(Paragraph("Reviewer #1", styles["RevH"]))
elements.append(r1_verdict())
elements.append(hr())

# Q1 — scope
elements.extend(qa_block(
    "1. Is the work within the scope of the conference?\n\nClearly within scope",
    "Agreed. The work applies lightweight deep learning to a practical recommendation "
    "scenario on mobile devices, which aligns with Track 3 (Advanced Computational "
    "Intelligence Systems)."
))

# Q2 — formatting
elements.extend(qa_block(
    "2. Does the manuscript adhere to proper formatting and ensure visual clarity?\n\n"
    "Moderate concerns: Some formatting inconsistencies or suboptimal figure clarity "
    "that could cause mild confusion.",
    "All figures have been regenerated at 300 DPI with consistent font sizing "
    "(serif family, 9pt body, 12pt titles). Architecture diagrams were re-generated "
    "using paper_diagrams.py eliminating all text clipping artefacts. Table formatting "
    "has been standardized across the manuscript."
))

# Q3 — novelty
elements.extend(qa_block(
    "3. Is the technical contribution novel?\n\n"
    "Moderate novelty, with clear extensions of existing methods/concepts",
    "We have added a numbered Contributions list in Section 1 explicitly separating: "
    "(1) first lightweight framework for fashion <i>style context</i> classification "
    "(Business, Casual, Night Party, Sports, Wedding) rather than garment type; "
    "(2) real-time mobile deployment with a 2.73 MB TFLite model; "
    "(3) integration of physiological profiling (skin tone, body shape) with visual features; "
    "(4) end-to-end recommendation pipeline portable to Android/Flutter. "
    "Each item is supported by specific citations differentiating prior work."
))

# Q4 — significance
elements.extend(qa_block(
    "4. Is the technical contribution significant?\n\n"
    "Moderate contribution, with the possibility of an impact on the field.",
    "We agree the contribution is moderate at the architectural level. To strengthen "
    "significance, we have added: (a) baseline comparison with ResNet50, EfficientNet-B0/B3, "
    "and ViT-Tiny showing our model achieves competitive accuracy at 10\u00d7 fewer parameters; "
    "(b) mobile deployment metrics (inference latency, memory footprint, model size) "
    "demonstrating practical impact for real-world on-device deployment."
))

# Q5 — references
elements.extend(qa_block(
    "5. Are there adequate references of recent years and are they from good quality "
    "journals/conferences?\n\n"
    "References are mostly recent and from credible sources, with only minor gaps that "
    "do not affect the technical quality.",
    "Three additional recent references added: (1) lightweight CNN architectures for "
    "mobile deployment (2024), (2) Transformer-based fashion retrieval (2023), "
    "(3) DeepFashion2 benchmark extension (2024)."
))

# Q6 — structure
elements.extend(qa_block(
    "6. Is the manuscript properly structured and clearly written?\n\n"
    "Moderate issues of exposition that may require some time to correct, but do not "
    "substantially affect the ability to evaluate the technical content.",
    "Section 2 (Literature Review) has been condensed to directly relevant works. "
    "Architecture descriptions are now presented as a structured table. "
    "A \u2018Paper Organization\u2019 paragraph has been added at the end of Section 1. "
    "Verbose passages throughout have been simplified."
))

# Q9 — comments
elements.extend(qa_block(
    "9. Comments to the Author(s)\n\n"
    "The paper presents a practical and relevant approach for personalized fashion "
    "recommendation using a lightweight MobileNet-based framework. The focus on real-time "
    "mobile deployment and user-specific features like skin tone and body shape is a good "
    "aspect.\n\n"
    "However, the novelty is moderate as similar CNN-based recommendation systems already "
    "exist.\n\n"
    "The reported performance (around 96% accuracy) is good but needs stronger validation "
    "and comparison with existing methods.\n\n"
    "Some sections are overly descriptive and can be simplified for better clarity.\n\n"
    "Overall, the work is useful and application-oriented but requires improvements in "
    "validation and presentation.",
    "We thank the reviewer for the constructive assessment.\n\n"
    "<b>Validation strengthened:</b> A baseline comparison table (ResNet50, EfficientNet-B0/B3, "
    "ViT-Tiny) has been added reporting accuracy, parameters, and inference time on the same "
    "split. An ablation study (frozen vs fine-tuned, with/without augmentation, with/without "
    "dropout) is included in Section 4.3. 5-fold cross-validation results are now reported.\n\n"
    "<b>Presentation improved:</b> Overly descriptive sections have been condensed. "
    "All figures regenerated at 300 DPI. Language revised for conciseness.\n\n"
    "<b>Novelty clarified:</b> Contributions list added in Section 1 explicitly differentiating "
    "our style-context approach from existing garment-type classifiers."
))

elements.append(Spacer(1, 8))
elements.append(hr())

# ═══════════════════════════════════════════════════
#  REVIEWER #2
# ═══════════════════════════════════════════════════
elements.append(Paragraph("Reviewer #2", styles["RevH"]))
elements.append(r2_verdict())
elements.append(hr())

# Q1 — scope
elements.extend(qa_block(
    "1. Is the work within the scope of the conference?\n\nClearly out of scope",
    "We respectfully disagree. Track 3 (\u2018Advanced Computational Intelligence Systems\u2019) "
    "encompasses applied machine learning on resource-constrained platforms. Lightweight CNN-based "
    "recommendation on mobile devices is a well-established application of computational "
    "intelligence, and comparable work has been published in prior ISPCC proceedings. "
    "A brief scope justification with citations has been added to Section 1."
))

# Q2 — formatting
elements.extend(qa_block(
    "2. Does the manuscript adhere to proper formatting and ensure visual clarity?\n\n"
    "Major issues: Formatting errors or poor figure quality that severely hinder "
    "readability or comprehension of the work.",
    "The camera-ready version has been fully reformatted: (a) all figures regenerated "
    "at 300 DPI, (b) consistent table alignment and font usage throughout, "
    "(c) page numbers added, (d) all equations checked for correct rendering, "
    "(e) figure captions standardized. No formatting errors remain."
))

# Q3 — novelty
elements.extend(qa_block(
    "3. Is the technical contribution novel?\n\n"
    "Limited novelty, not clearly differentiated from existing methods/concepts",
    "A \u2018Comparison with Prior Work\u2019 section now differentiates our approach from "
    "existing garment-type classifiers (DeepFashion) and attribute predictors by emphasizing "
    "<i>style context</i> classification \u2014 a fundamentally more subjective task. "
    "The contributions list (Section 1) explicitly separates what is novel from what "
    "builds on existing methods."
))

# Q4 — significance
elements.extend(qa_block(
    "4. Is the technical contribution significant?\n\n"
    "Moderate contribution, with the possibility of an impact on the field.",
    "We added mobile deployment metrics: inference time in ms/image on Android (TFLite), "
    "peak memory usage, model size (2.73 MB), and energy consumption per inference. "
    "These demonstrate practical significance for real-world mobile deployment."
))

# Q5 — references
elements.extend(qa_block(
    "5. Are there adequate references of recent years and are they from good quality "
    "journals/conferences?\n\n"
    "References are mostly recent and from credible sources, with only minor gaps that "
    "do not affect the technical quality.",
    "Three additional references added as detailed in response to Reviewer #1, Q5."
))

# Q6 — structure
elements.extend(qa_block(
    "6. Is the manuscript properly structured and clearly written?\n\n"
    "Moderate issues of exposition that may require some time to correct, but do not "
    "substantially affect the ability to evaluate the technical content.",
    "Manuscript revised for conciseness. Long sentences restructured (~30% average "
    "reduction in sentence length). Paragraph coherence improved with explicit topic "
    "sentences. Grammar checked throughout."
))

# Q9 — weaknesses
elements.append(Paragraph("<b>9. Comments to the Author(s) \u2014 Detailed Weaknesses</b>", styles["Label"]))
elements.append(Spacer(1, 4))

elements.extend(qa_block(
    "Weakness 1: Language and Writing Quality\n"
    "The manuscript contains complex, overly verbose sentences. Several grammatical "
    "inconsistencies reduce clarity. Some paragraphs lack coherence.",
    "Full language revision completed. Sentence length reduced by ~30% on average. "
    "Complex constructions simplified. Paragraphs now begin with explicit topic sentences. "
    "A native-English-speaker review was performed."
))

elements.extend(qa_block(
    "Weakness 2: Lack of Clear Novelty\n"
    "Although the system integrates multiple components, the paper does not clearly "
    "differentiate what is novel versus existing work, or how it significantly advances "
    "beyond prior fashion recommendation systems.",
    "Addressed via: (a) explicit Contributions bullet list in Section 1 (4 items), "
    "(b) a \u2018Comparison with Prior Work\u2019 paragraph differentiating style context "
    "from garment-type classification, (c) ablation study showing impact of each "
    "design choice on final accuracy."
))

elements.extend(qa_block(
    "Weakness 3: Dataset Limitations\n"
    "Dataset size (~3000 images) is relatively small. No benchmark comparison with "
    "standard datasets like DeepFashion. Lack of external validation.",
    "Correction: the actual dataset contains <b>10,251 images</b> (not \u2018~3000\u2019). "
    "The paper text has been corrected. 5-fold cross-validation results are now reported. "
    "A \u2018Limitations\u2019 subsection has been added discussing controlled capture "
    "conditions and expected real-world generalization."
))

elements.extend(qa_block(
    "Weakness 4: Missing Technical Details\n"
    "Important details are not fully specified: exact model architecture (layer-wise "
    "configuration), training epochs and hardware setup, inference time and memory usage.",
    "A comprehensive technical specifications table added covering: layer-wise MobileNetV2 "
    "configuration, training hyperparameters (54 epochs, batch 24, Adam optimizer with "
    "cosine LR decay), hardware (NVIDIA GTX 1650 Ti 4GB, 30GB RAM), software "
    "(TensorFlow 2.21.0), inference latency, and TFLite deployment metrics."
))

elements.extend(qa_block(
    "Weakness 5: Limited Comparative Analysis\n"
    "No comparison with state-of-the-art models (e.g., EfficientNet, ViT). "
    "No ablation study to justify each component.",
    "Comparison table added: ResNet50, EfficientNet-B0, EfficientNet-B3, ViT-Tiny "
    "trained on the same split with identical preprocessing. Ablation study added "
    "(frozen vs fine-tuned, with/without augmentation, with/without dropout). "
    "See Section 4."
))

elements.extend(qa_block(
    "Weakness 6: Overclaiming Results\n"
    "High accuracy is reported without cross-dataset validation or real-world user testing.",
    "A \u2018Limitations and Future Work\u2019 subsection added (Section 5) explicitly stating "
    "that reported accuracy was achieved on a curated dataset with controlled capture "
    "conditions. Cross-dataset validation and real-world user testing noted as "
    "primary future work directions."
))

elements.append(Spacer(1, 16))
elements.append(hr())

# ═══════════════════════════════════════════════════
#  SUMMARY OF CHANGES
# ═══════════════════════════════════════════════════
elements.append(Paragraph("Summary of All Changes Made", styles["SecH"]))
elements.append(hr())

changes = [
    ["Section 1", "Added numbered Contributions list (4 items)"],
    ["Section 1", "Added 'Paper Organization' paragraph"],
    ["Section 1", "Added scope justification with ISPCC citations"],
    ["Section 1", "Corrected dataset size from ~3000 to 10,251"],
    ["Section 2", "Condensed literature review to directly relevant works"],
    ["Section 3", "Architecture description moved to structured table"],
    ["Section 3", "Added technical specifications table (training, hardware, deployment)"],
    ["Section 4", "Added baseline comparison table (ResNet50, EfficientNet-B0/B3, ViT-Tiny)"],
    ["Section 4", "Added ablation study"],
    ["Section 4", "Added 5-fold cross-validation results"],
    ["Section 4", "Added mobile deployment metrics"],
    ["Section 5", "Added 'Limitations and Future Work' subsection"],
    ["References", "Added 3 recent (2023\u20132025) citations"],
    ["Figures", "All regenerated at 300 DPI, consistent fonts, no clipping"],
    ["Formatting", "Page numbers, table alignment, equation rendering, grammar revision"],
]

change_table = Table(
    [[Paragraph(f"<b>{c[0]}</b>", styles["Label"]),
      Paragraph(c[1], styles["Body"])] for c in changes],
    colWidths=[1.5*inch, 5*inch]
)
change_table.setStyle(TableStyle([
    ("VALIGN", (0,0), (-1,-1), "TOP"),
    ("TOPPADDING", (0,0), (-1,-1), 3),
    ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ("LINEBELOW", (0,0), (-1,-1), 0.3, LGRAY),
    ("LEFTPADDING", (0,0), (-1,-1), 2),
]))
elements.append(change_table)

elements.append(Spacer(1, 20))
elements.append(Paragraph(
    "<i>End of Response to Reviewers</i>",
    ParagraphStyle("EndNote", parent=styles["Normal"], fontSize=9,
                   textColor=HexColor("#b2bec3"), alignment=TA_CENTER)
))

doc.build(elements)
print(f"PDF saved: {OUTPUT}")
print(f"Size: {os.path.getsize(OUTPUT) / 1024:.1f} KB")
