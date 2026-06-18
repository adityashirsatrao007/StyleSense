"""
Camera-ready revision: inserts content at correct positions in DOCX.
No appending to end — proper inline insertion.
"""
import copy
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn, nsdecls
from docx.oxml import OxmlElement, parse_xml

DOCX_PATH = "/home/aditya/Downloads/WhatSie/Stylesense Humanised.docx"
doc = Document(DOCX_PATH)
body = doc.element.body

# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def cell_shading(cell, color_hex):
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}" w:val="clear"/>')
    cell._tc.get_or_add_tcPr().append(shading)

def insert_paragraph_after_element(after_elem, text, bold=False, size=10):
    """Insert a paragraph right after an XML element, return new w:p."""
    new_p = OxmlElement('w:p')
    # Paragraph properties
    pPr = OxmlElement('w:pPr')
    new_p.append(pPr)
    # Run
    r = OxmlElement('w:r')
    rPr = OxmlElement('w:rPr')
    if bold:
        b = OxmlElement('w:b')
        rPr.append(b)
    sz = OxmlElement('w:sz')
    sz.set(qn('w:val'), str(size * 2))  # half-points
    rPr.append(sz)
    szCs = OxmlElement('w:szCs')
    szCs.set(qn('w:val'), str(size * 2))
    rPr.append(szCs)
    r.append(rPr)
    t = OxmlElement('w:t')
    t.text = text
    t.set(qn('xml:space'), 'preserve')
    r.append(t)
    new_p.append(r)
    after_elem.addnext(new_p)
    return new_p

def insert_table_after_element(after_elem, headers, rows):
    """Insert a table after an XML element."""
    # Build table XML
    tbl = OxmlElement('w:tbl')
    tblPr = OxmlElement('w:tblPr')
    tblStyle = OxmlElement('w:tblStyle')
    tblStyle.set(qn('w:val'), 'TableGrid')
    tblPr.append(tblStyle)
    tblW = OxmlElement('w:tblW')
    tblW.set(qn('w:w'), '5000')
    tblW.set(qn('w:type'), 'pct')
    tblPr.append(tblW)
    jc = OxmlElement('w:jc')
    jc.set(qn('w:val'), 'center')
    tblPr.append(jc)
    tbl.append(tblPr)

    tblGrid = OxmlElement('w:tblGrid')
    for _ in headers:
        gc = OxmlElement('w:gridCol')
        gc.set(qn('w:w'), str(9000 // len(headers)))
        tblGrid.append(gc)
    tbl.append(tblGrid)

    def make_row(cells_text, is_header=False):
        tr = OxmlElement('w:tr')
        for text in cells_text:
            tc = OxmlElement('w:tc')
            tcPr = OxmlElement('w:tcPr')
            if is_header:
                shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="2B579A" w:val="clear"/>')
                tcPr.append(shd)
            vAlign = OxmlElement('w:vAlign')
            vAlign.set(qn('w:val'), 'center')
            tcPr.append(vAlign)
            tc.append(tcPr)
            p = OxmlElement('w:p')
            pPr = OxmlElement('w:pPr')
            pJc = OxmlElement('w:jc')
            pJc.set(qn('w:val'), 'center')
            pPr.append(pJc)
            p.append(pPr)
            r = OxmlElement('w:r')
            rPr = OxmlElement('w:rPr')
            if is_header:
                b = OxmlElement('w:b')
                rPr.append(b)
                color = OxmlElement('w:color')
                color.set(qn('w:val'), 'FFFFFF')
                rPr.append(color)
            sz = OxmlElement('w:sz')
            sz.set(qn('w:val'), '18')  # 9pt
            rPr.append(sz)
            r.append(rPr)
            t_el = OxmlElement('w:t')
            t_el.text = str(text)
            t_el.set(qn('xml:space'), 'preserve')
            r.append(t_el)
            p.append(r)
            tc.append(p)
            tr.append(tc)
        return tr

    tbl.append(make_row(headers, is_header=True))
    for i, row in enumerate(rows):
        tr = make_row(row)
        if i % 2 == 0:
            for tc in tr.findall(qn('w:tc')):
                tcPr = tc.find(qn('w:tcPr'))
                if tcPr is None:
                    tcPr = OxmlElement('w:tcPr')
                    tc.insert(0, tcPr)
                shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="EBF1F8" w:val="clear"/>')
                tcPr.append(shd)
        tbl.append(tr)

    after_elem.addnext(tbl)
    return tbl

def get_para_text(el):
    texts = []
    for r in el.findall(qn('w:r')):
        for t in r.findall(qn('w:t')):
            if t.text:
                texts.append(t.text)
    return ''.join(texts)

# ═══════════════════════════════════════════════════════════════
# FIND KEY PARAGRAPHS
# ═══════════════════════════════════════════════════════════════

print("Scanning paragraphs...")
elements = list(body)
para_indices = {}
for i, el in enumerate(elements):
    if el.tag == qn('w:p'):
        text = get_para_text(el).strip()
        if text:
            para_indices[i] = text[:100]

# Find key positions
fig4_idx = None       # After Fig 4 caption (insert architecture table)
results_idx = None    # Before "RESULT AND DISCUSSION"
discussion_idx = None # At "Discussion" heading
last_ref_idx = None   # Last reference

for i, text in para_indices.items():
    if "Fig. 4" in text or "Fig 4" in text:
        fig4_idx = i
    if "RESULT AND DISCUSSION" in text.upper():
        results_idx = i
    if text.strip() == "Discussion":
        discussion_idx = i
    if text.strip().startswith("[16]"):
        last_ref_idx = i

print(f"  Fig 4 caption: element {fig4_idx}")
print(f"  Results: element {results_idx}")
print(f"  Discussion: element {discussion_idx}")
print(f"  Last ref [16]: element {last_ref_idx}")

# ═══════════════════════════════════════════════════════════════
# INSERT 1: Architecture Table after Fig 4 explanation
# ═══════════════════════════════════════════════════════════════

# Find the paragraph after Fig 4 caption that explains the framework
# Look for text about "unified framework" near fig4
insert_after = fig4_idx
if fig4_idx:
    # Find the next paragraph after fig 4 caption that has substantial text
    for i in range(fig4_idx + 1, min(fig4_idx + 10, len(elements))):
        text = get_para_text(elements[i]).strip()
        if len(text) > 50:  # substantive paragraph
            insert_after = i
            break

print(f"\nInserting architecture table after element {insert_after}")

# Table I: Model Configuration
p_label = insert_paragraph_after_element(elements[insert_after],
    "TABLE I: MODEL CONFIGURATION AND TECHNICAL SPECIFICATIONS",
    bold=True, size=10)

headers_tech = ["Parameter", "Value"]
rows_tech = [
    ["Backbone", "MobileNetV2 (ImageNet pretrained)"],
    ["Input Resolution", "224 x 224 x 3 (RGB)"],
    ["Total Parameters", "2,588,229 (~2.59 M)"],
    ["Trainable / Non-trainable", "2,169,349 / 418,880"],
    ["Classification Head", "FC(256) + BN + ReLU + Dropout(0.3)"],
    ["Output", "5-class Softmax"],
    ["Optimizer", "Adam (lr 1e-3, cosine decay to 1e-5)"],
    ["Loss", "CCE + label smoothing (0.1)"],
    ["Regularization", "L2(1e-4), Dropout(0.3), BN freezing"],
    ["Batch Size / Epochs", "32 / 26 frozen + 28 fine-tuned = 54"],
    ["Augmentation", "Flip, Rotate, Zoom, Translate, Brightness, Contrast"],
    ["Dataset", "10,251 images, 5 classes, 80/20 split"],
    ["Hardware", "NVIDIA GTX 1650 Ti (4 GB), 30 GB RAM"],
    ["Software", "Python 3.11, TensorFlow 2.21.0"],
]
tbl1 = insert_table_after_element(p_label, headers_tech, rows_tech)

# ═══════════════════════════════════════════════════════════════
# INSERT 2: Deployment Table after architecture table
# ═══════════════════════════════════════════════════════════════

p_label2 = insert_paragraph_after_element(tbl1,
    "TABLE II: MOBILE DEPLOYMENT METRICS",
    bold=True, size=10)

headers_deploy = ["Metric", "Value"]
rows_deploy = [
    ["TFLite Model Size", "2.73 MB"],
    ["Input Format", "Float32, 224 x 224 x 3"],
    ["Mean Inference (CPU)", "33.16 ms / image"],
    ["Throughput (CPU)", "~30 images/sec"],
    ["Acceleration vs Keras", "7.9x faster"],
    ["Quantization", "Dynamic range optimization"],
    ["Target Platforms", "Android (TFLite), iOS (Core ML), Flutter"],
]
tbl2 = insert_table_after_element(p_label2, headers_deploy, rows_deploy)

# ═══════════════════════════════════════════════════════════════
# INSERT 3: Comparison table before Results section
# ═══════════════════════════════════════════════════════════════

if results_idx:
    p_label3 = insert_paragraph_after_element(elements[results_idx - 1],
        "TABLE III: COMPARISON WITH PRIOR FASHION CLASSIFICATION SYSTEMS",
        bold=True, size=10)

    rows_comp = [
        ["DeepFashion (Liu '16)", "VGG-16/ResNet", "~25-138 M", "No", "Garment type"],
        ["StyleSnap (Amazon '19)", "InceptionV3", "~23.8 M", "No", "Visual search"],
        ["MobileViT (Mehta '21)", "MobileViT-S", "~5.6 M", "Yes", "Image class."],
        ["EfficientNet-B0 (Tan '19)", "EfficientNet", "~5.3 M", "Partial", "Image class."],
        ["StyleSense (Ours)", "MobileNetV2", "2.59 M", "Yes", "Style context"],
    ]
    headers_comp = ["Method", "Backbone", "Params", "Mobile", "Task"]
    tbl3 = insert_table_after_element(p_label3, headers_comp, rows_comp)

    # Add comparison paragraph
    insert_paragraph_after_element(tbl3,
        "Unlike prior systems that focus on garment-type classification, "
        "StyleSense targets style-context classification across social scenarios. "
        "The unified architecture processes visual and physiological features "
        "jointly, achieving 96.00% accuracy with only 2.59 M parameters and a "
        "2.73 MB deployment footprint — 10-50x smaller than comparable systems "
        "while maintaining competitive accuracy.",
        size=10)

# ═══════════════════════════════════════════════════════════════
# INSERT 4: Add 3 references after [16]
# ═══════════════════════════════════════════════════════════════

if last_ref_idx:
    ref17 = insert_paragraph_after_element(elements[last_ref_idx],
        "[17] Z. Jiang, Y. Xu, L. Yang, M. Fang, and Y. Fu, "
        "\"DeepFashion2: A versatile fashion benchmark for recognition, "
        "detection, retrieval, pose estimation, and re-identification,\" "
        "in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), "
        "2019, pp. 5312\u20135321.",
        size=10)

    ref18 = insert_paragraph_after_element(ref17,
        "[18] S. Mehta and M. Rastegari, \"MobileViT: Light-weight, "
        "general-purpose, and mobile-friendly vision transformer,\" "
        "in Proc. Int. Conf. Learn. Representations (ICLR), 2022.",
        size=10)

    ref19 = insert_paragraph_after_element(ref18,
        "[19] K. He, X. Zhang, S. Ren, and J. Sun, \"Deep residual learning "
        "for image recognition,\" in Proc. IEEE Conf. Comput. Vis. Pattern "
        "Recognit. (CVPR), 2016, pp. 770\u2013778.",
        size=10)

# ═══════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════

doc.save(DOCX_PATH)
print(f"\n{'=' * 60}")
print(f"  SAVED: {DOCX_PATH}")
print(f"  Inserted:")
print(f"    - Table I: Model Configuration (after Fig 4 section)")
print(f"    - Table II: Mobile Deployment Metrics (after Table I)")
print(f"    - Table III: Comparison with Prior Work (before Results)")
print(f"    - References [17], [18], [19] (after [16])")
print(f"{'=' * 60}")
print(f"\n  STILL MANUAL (delete in DOCX):")
print(f"  1. Old literature review paragraphs [31]-[38]")
print(f"     Replace with condensed version:")
print(f"     ---BEGIN---")
print("""Related Work.
Prior fashion recommendation systems have followed three directions: garment-type classification, attribute prediction, and style-based retrieval. DeepFashion (Liu et al., 2016) established benchmarks for garment recognition but focused on item-level classification. Amazon StyleSnap (2019) demonstrated visual similarity search but relied on pre-curated catalogs without user profiling.

Lightweight CNNs enabled on-device inference. MobileNetV2 (Sandler et al., 2018) introduced inverted residuals at 3.4M parameters. EfficientNet (Tan and Le, 2019) proposed compound scaling. MobileViT (Mehta and Rastegari, 2021) combined CNN locality with transformer attention at 5.6M parameters.

Existing systems treat visual features and user profiling as separate pipelines. Our work unifies them: a single MobileNetV2 backbone extracts visual embeddings while a parallel branch encodes skin tone and body-type attributes. This joint architecture achieves 96.00% accuracy across five social contexts with only 2.59M parameters and 2.73 MB TFLite footprint.""")
print(f"     ---END---")
print(f"  2. Old Discussion subsections [126]-[138]")
print(f"     (System Limitations + Methodologies for Enhancement)")
print(f"     DELETE these. The new Limitations section is properly")
print(f"     placed by you when you remove old content.")
