"""
Generate architecture & workflow diagrams for the StyleSense paper.
Produces Fig 1 (System Architecture), Fig 2 (Workflow), Fig 4 (Unified Framework).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "paper_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

# ─── Color scheme ─────────────────────────────────────────────
C_INPUT = "#4472C4"
C_PREPROCESS = "#5B9BD5"
C_BACKBONE = "#2F5597"
C_CLASSIFIER = "#C55A11"
C_OUTPUT = "#548235"
C_FLOW = "#808080"
C_BG = "#F2F2F2"


def draw_rounded_box(ax, xy, width, height, color, label, sublabel=None,
                     fc_alpha=0.15, lw=1.5):
    """Draw a rounded rectangle with label."""
    x, y = xy
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.08",
        facecolor=color, alpha=fc_alpha,
        edgecolor=color, linewidth=lw,
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + height / 2, label,
            ha="center", va="center", fontsize=9, fontweight="bold", color=color)
    if sublabel:
        ax.text(x + width / 2, y + height * 0.25, sublabel,
                ha="center", va="center", fontsize=7, color="#666666")
    return box


def draw_arrow(ax, x1, y1, x2, y2, color=C_FLOW, lw=1.5):
    """Draw a directed arrow between points."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="->", color=color, lw=lw,
            connectionstyle="arc3,rad=0",
        ),
    )


# ═══════════════════════════════════════════════════════════════
#  Fig 1 — Architecture of the High-Fidelity StyleSense
#           Recommendation Engine
# ═══════════════════════════════════════════════════════════════
def fig1_architecture():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_facecolor("white")

    # Title
    ax.text(5, 4.7, "Architecture of the High-Fidelity StyleSense Recommendation Engine",
            ha="center", va="center", fontsize=12, fontweight="bold")

    # Block 1: Input Image
    draw_rounded_box(ax, (0.3, 3.0), 1.6, 1.2, C_INPUT,
                     "Input Image", "224×224×3\nRaw Fashion Data")

    # Block 2: Preprocessing
    draw_rounded_box(ax, (2.4, 3.0), 1.6, 1.2, C_PREPROCESS,
                     "Preprocessing", "Normalization\nStochastic Augmentation")

    # Block 3: MobileNetV2 Backbone
    draw_rounded_box(ax, (4.5, 3.0), 2.0, 1.2, C_BACKBONE,
                     "MobileNetV2 Backbone", "Inverted Residuals\nLinear Bottlenecks")

    # Block 4: Global Pooling + Flatten
    draw_rounded_box(ax, (7.0, 3.0), 1.6, 1.2, C_CLASSIFIER,
                     "GAP + Flatten", "Global Avg Pool\n→ Feature Vector")

    # Block 5: Dense Head
    draw_rounded_box(ax, (4.5, 1.2), 2.0, 1.2, C_CLASSIFIER,
                     "Dense Classifier", "FC(128) + ReLU\nDropout(0.3)")

    # Block 6: Softmax Output
    draw_rounded_box(ax, (7.0, 1.2), 1.6, 1.2, C_OUTPUT,
                     "Softmax Output", "5 Classes\nProbability Dist.")

    # Block 7: Sidebar — Skin Tone + Body Shape
    draw_rounded_box(ax, (0.3, 1.2), 1.6, 1.2, "#7030A0",
                     "Physiological\nProfiling", "Skin Tone Mapping\nBody Shape Analysis")

    # Arrows — main pipeline
    draw_arrow(ax, 1.9, 3.6, 2.4, 3.6)
    draw_arrow(ax, 4.0, 3.6, 4.5, 3.6)
    draw_arrow(ax, 6.5, 3.6, 7.0, 3.6)
    draw_arrow(ax, 7.8, 3.0, 7.8, 2.4)
    draw_arrow(ax, 6.5, 2.4, 6.5, 1.8)
    draw_arrow(ax, 7.8, 1.8, 7.0, 1.8)

    # Arrow from physiological profiling to classifier
    draw_arrow(ax, 1.1, 1.2, 1.1, 0.6, C_FLOW, lw=1.0)
    ax.plot([1.1, 6.5], [0.6, 0.6], color=C_FLOW, lw=1.0, linestyle="--")
    arr = FancyArrowPatch((6.5, 0.6), (6.5, 1.2),
                          arrowstyle="->", color=C_FLOW, lw=1.0,
                          connectionstyle="arc3,rad=0")
    ax.add_patch(arr)

    # Legend
    legend_elements = [
        mpatches.Patch(color=C_INPUT, alpha=0.3, label="Input"),
        mpatches.Patch(color=C_PREPROCESS, alpha=0.3, label="Preprocessing"),
        mpatches.Patch(color=C_BACKBONE, alpha=0.3, label="Feature Extraction"),
        mpatches.Patch(color=C_CLASSIFIER, alpha=0.3, label="Classification"),
        mpatches.Patch(color=C_OUTPUT, alpha=0.3, label="Output"),
    ]
    ax.legend(handles=legend_elements, loc="lower center",
              ncol=5, fontsize=7, framealpha=0.8,
              bbox_to_anchor=(0.5, -0.05))

    save_path = OUTPUT_DIR / "Fig1_architecture.png"
    fig.savefig(str(save_path))
    print(f"Fig 1 saved to {save_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  Fig 2 — Workflow of the High-Fidelity StyleSense
#           Recommendation System
# ═══════════════════════════════════════════════════════════════
def fig2_workflow():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(5, 5.7, "Workflow of the High-Fidelity StyleSense Recommendation System",
            ha="center", va="center", fontsize=12, fontweight="bold")

    steps = [
        ("Input Image\nAcquisition", C_INPUT, 0.3, 4.0),
        ("Image\nPreprocessing", C_PREPROCESS, 2.3, 4.0),
        ("MobileNetV2\nFeature Extraction", C_BACKBONE, 4.3, 4.0),
        ("Skin Tone &\nBody Shape Profiling", "#7030A0", 4.3, 2.0),
    ]

    for label, color, x, y in steps:
        draw_rounded_box(ax, (x, y), 1.6, 1.2, color, label)

    # Decision diamond (simplified as box)
    draw_rounded_box(ax, (6.3, 3.0), 1.6, 1.2, C_CLASSIFIER,
                     "Fusion &\nClassification", "FC(128) → Softmax")

    draw_rounded_box(ax, (8.3, 3.0), 1.4, 1.2, C_OUTPUT,
                     "Style\nPrediction", "5 Categories")

    # Arrows
    draw_arrow(ax, 1.9, 4.6, 2.3, 4.6)
    draw_arrow(ax, 3.9, 4.6, 4.3, 4.6)
    draw_arrow(ax, 5.9, 4.6, 6.3, 3.6)
    draw_arrow(ax, 5.9, 2.6, 6.3, 3.0)
    draw_arrow(ax, 7.1, 4.6, 7.5, 4.6)
    ax.plot([7.5, 7.5], [4.6, 3.6], color=C_FLOW, lw=1.5)
    arr = FancyArrowPatch((7.5, 3.6), (6.3, 3.6),
                          arrowstyle="->", color=C_FLOW, lw=1.0,
                          connectionstyle="arc3,rad=0")
    ax.add_patch(arr)
    draw_arrow(ax, 6.3, 4.6, 4.3, 4.0)

    # Bottom arrow
    ax.plot([5.1, 5.1], [2.0, 1.2], color=C_FLOW, lw=1.0, linestyle="--")
    ax.plot([5.1, 7.1], [1.2, 1.2], color=C_FLOW, lw=1.0)
    arr2 = FancyArrowPatch((7.1, 1.2), (7.9, 3.0),
                           arrowstyle="->", color=C_FLOW, lw=1.0,
                           connectionstyle="arc3,rad=-0.3")
    ax.add_patch(arr2)

    # Output boxes
    categories = ["Business", "Casual", "Night Party", "Sports", "Wedding"]
    for i, cat in enumerate(categories):
        y_cat = 0.1 + i * 0.2
        ax.text(9.2, y_cat, f"▪ {cat}", fontsize=8, color=C_OUTPUT,
                fontweight="bold", va="bottom")

    save_path = OUTPUT_DIR / "Fig2_workflow.png"
    fig.savefig(str(save_path))
    print(f"Fig 2 saved to {save_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  Fig 4 — Architecture of the Unified StyleSense Framework
# ═══════════════════════════════════════════════════════════════
def fig4_unified_framework():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    ax.text(5, 5.2, "Architecture of the Unified StyleSense Framework",
            ha="center", va="center", fontsize=12, fontweight="bold")

    # Input
    draw_rounded_box(ax, (0.2, 3.5), 1.4, 1.2, C_INPUT,
                     "Input Image\n(224×224×3)", "Raw Pixels")

    # Preprocess
    draw_rounded_box(ax, (2.0, 3.5), 1.4, 1.2, C_PREPROCESS,
                     "Preprocess", "Normalize\nAugment")

    # MobileNetV2 encoder — larger block showing layers
    enc_x, enc_y, enc_w, enc_h = 3.8, 2.8, 2.0, 2.2
    enc_box = FancyBboxPatch(
        (enc_x, enc_y), enc_w, enc_h,
        boxstyle="round,pad=0.1",
        facecolor=C_BACKBONE, alpha=0.12,
        edgecolor=C_BACKBONE, linewidth=1.5,
    )
    ax.add_patch(enc_box)
    ax.text(enc_x + enc_w / 2, enc_y + enc_h * 0.85,
            "MobileNetV2 Encoder", ha="center", fontsize=9, fontweight="bold",
            color=C_BACKBONE)
    encoder_layers = [
        "Conv2D (32, 3×3, stride=2)",
        "bottleneck (t=1, 16)",
        "bottleneck (t=6, 24)",
        "bottleneck (t=6, 32)",
        "bottleneck (t=6, 64)",
        "bottleneck (t=6, 96)",
        "bottleneck (t=6, 160)",
        "Conv2D (1280, 1×1)",
    ]
    for i, layer in enumerate(encoder_layers):
        ax.text(enc_x + enc_w / 2, enc_y + enc_h * (0.70 - i * 0.075),
                layer, ha="center", fontsize=6, color="#444444")

    # GAP + Flatten
    draw_rounded_box(ax, (6.2, 3.5), 1.4, 1.2, C_CLASSIFIER,
                     "GAP + Flatten", "Pool → 1D Vector")

    # Dense + Dropout
    draw_rounded_box(ax, (8.0, 3.5), 1.4, 1.2, C_CLASSIFIER,
                     "FC(128) + Dropout(0.3)", "ReLU Activation")

    # Softmax
    draw_rounded_box(ax, (8.0, 1.5), 1.4, 1.1, C_OUTPUT,
                     "Softmax (5)", "Class Probabilities")

    # Arrows
    draw_arrow(ax, 1.6, 4.1, 2.0, 4.1)
    draw_arrow(ax, 3.4, 4.1, 3.8, 4.1)
    draw_arrow(ax, 5.8, 3.9, 6.2, 3.9)
    draw_arrow(ax, 7.6, 3.9, 8.0, 3.9)
    draw_arrow(ax, 8.7, 3.5, 8.7, 2.6)

    # Side path: user metrics
    draw_rounded_box(ax, (0.2, 1.5), 1.4, 0.8, "#7030A0",
                     "User Metrics", "Skin Tone\nBody Shape")
    ax.plot([0.9, 0.9], [1.5, 1.0], color=C_FLOW, lw=1.0, linestyle="--")
    ax.plot([0.9, 5.0], [1.0, 1.0], color=C_FLOW, lw=1.0, linestyle="--")
    arr = FancyArrowPatch((5.0, 1.0), (5.0, 2.8),
                          arrowstyle="->", color=C_FLOW, lw=1.0,
                          connectionstyle="arc3,rad=0")
    ax.add_patch(arr)

    # Label: Feature Concatenation
    ax.text(5.0, 2.6, "Feature\nConcatenation", ha="center",
            fontsize=7, fontweight="bold", color="#7030A0",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="#7030A0", alpha=0.7))

    save_path = OUTPUT_DIR / "Fig4_unified_framework.png"
    fig.savefig(str(save_path))
    print(f"Fig 4 saved to {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating StyleSense paper architecture diagrams...")
    fig1_architecture()
    fig2_workflow()
    fig4_unified_framework()
    print(f"\nAll 3 architecture diagrams saved to: {OUTPUT_DIR}/")
    print("  Fig1_architecture.png  — System Architecture")
    print("  Fig2_workflow.png       — Recommendation Workflow")
    print("  Fig4_unified_framework.png  — Unified Framework")
