"""
Clean, readable IEEE paper diagrams for StyleSense.
No cramming — proper spacing, big text, clear arrows.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "paper_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.pad_inches": 0.05,
})

# Colors
C = {
    "input":  "#2C5F8A",
    "prep":   "#4A86C8",
    "back":   "#1A3A5C",
    "head":   "#D45D2B",
    "out":    "#2E7D32",
    "prof":   "#7B4EA0",
}

def box(ax, x, y, w, h, color, text, sub=""):
    r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                        facecolor=color, edgecolor="#ddd", lw=0.5, zorder=2)
    ax.add_patch(r)
    ax.text(x+w/2, y+h/2+0.5, text, ha="center", va="center",
            fontsize=8, fontweight="bold", color="white", zorder=3)
    if sub:
        ax.text(x+w/2, y+h*0.22, sub, ha="center", va="center",
                fontsize=6.5, color="white", alpha=0.85, zorder=3)

def arrow(ax, x1, y1, x2, y2, c="#888", lw=1.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=lw,
                                mutation_scale=12), zorder=2)


def fig1_architecture():
    """
    3.44" x 1.06" — simple horizontal pipeline.
    4 boxes, clean, readable.
    """
    W, H = 3.44, 1.06
    fig, ax = plt.subplots(1, 1, figsize=(W, H))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    bh = 0.55
    y0 = (H - bh) / 2
    n = 4
    gap = 0.12
    bw = (W - gap * (n-1)) / n

    items = [
        (C["input"], "Input\n224x224"),
        (C["back"], "MobileNetV2\nBackbone"),
        (C["head"], "Classifier\nFC+Softmax"),
        (C["out"], "Prediction\n5 Classes"),
    ]
    for i, (clr, txt) in enumerate(items):
        x = i * (bw + gap)
        box(ax, x, y0, bw, bh, clr, txt, "")
        if i < n-1:
            arrow(ax, x+bw+0.02, y0+bh/2, x+bw+gap-0.02, y0+bh/2, clr)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    fig.savefig(OUTPUT_DIR / "Fig1_architecture.png")
    plt.close(fig)
    print(f"Fig1: {W}\"x{H}\"")


def fig2_workflow():
    """
    3.44" x 3.42" — vertical system flow.
    5 clean stages, good spacing.
    """
    W, H = 3.44, 3.42
    fig, ax = plt.subplots(1, 1, figsize=(W, H))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    bw, bh = 2.2, 0.45
    cx = (W - bw) / 2
    gap = 0.15
    steps = [
        (C["input"], "Input Image", "User captures fashion photo"),
        (C["prep"], "Preprocessing", "Resize, normalize, augment"),
        (C["back"], "Feature Extraction", "MobileNetV2 encoder"),
        (C["head"], "Classification", "FC layer + Softmax"),
        (C["out"], "Style Prediction", "Business / Casual / Party / Sport / Wedding"),
    ]
    total_h = len(steps) * bh + (len(steps)-1) * gap
    y0 = (H - total_h) / 2 + total_h

    for i, (clr, lbl, sub) in enumerate(steps):
        y = y0 - i * (bh + gap)
        box(ax, cx, y, bw, bh, clr, lbl, sub)
        if i < len(steps) - 1:
            arrow(ax, cx+bw/2, y-0.02, cx+bw/2, y-bh-gap+0.02, C["back"])

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    fig.savefig(OUTPUT_DIR / "Fig2_workflow.png")
    plt.close(fig)
    print(f"Fig2: {W}\"x{H}\"")


def fig4_unified_framework():
    """
    3.44" x 3.38" — two-branch framework.
    Left: classifier path. Right: user profiling. Merge at top.
    """
    W, H = 3.44, 3.38
    fig, ax = plt.subplots(1, 1, figsize=(W, H))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    bw, bh = 1.3, 0.4
    gap = 0.08

    # Center column
    cx = (W - bw) / 2

    # Input at top
    box(ax, cx, H-0.50, bw, bh, C["input"], "Input Image")
    arrow(ax, cx+bw/2, H-0.50-0.02, cx+bw/2, H-0.50-bh-gap+0.02, C["input"])

    # Preprocessing
    box(ax, cx, H-0.98, bw, bh, C["prep"], "Preprocessing")
    arrow(ax, cx+bw/2, H-0.98-0.02, cx+bw/2, H-0.98-bh-gap+0.02, C["prep"])

    # MobileNetV2 - wide box, full width
    mw = 2.8
    mx = (W - mw) / 2
    box(ax, mx, H-1.55, mw, 0.6, C["back"], "MobileNetV2 Backbone", "Inverted Residual Blocks")
    arrow(ax, cx+bw/2, H-1.55-0.02, cx+bw/2, H-1.55-bh-gap+0.02, C["back"])

    # GAP
    box(ax, cx, H-2.35, bw, bh, C["head"], "GAP + Flatten")
    ax.plot([cx+bw/2, cx+bw/2], [H-2.35-0.02, H-2.35-0.08], color="#888", lw=1)

    # Two branches side by side
    bw2 = 1.10
    gap2 = 0.08
    total2 = 2 * bw2 + gap2
    left_x = (W - total2) / 2
    right_x = left_x + bw2 + gap2

    # Branch arrows
    ax.annotate("", xy=(left_x+bw2/2, H-2.35-0.43), xytext=(cx+bw/2, H-2.35-0.08),
                arrowprops=dict(arrowstyle="-|>", color=C["head"], lw=1, mutation_scale=10))
    ax.annotate("", xy=(right_x+bw2/2, H-2.35-0.43), xytext=(cx+bw/2, H-2.35-0.08),
                arrowprops=dict(arrowstyle="-|>", color=C["head"], lw=1, mutation_scale=10))

    # Classifier branch
    box(ax, left_x, H-2.78, bw2, bh, C["head"], "FC(128)+Dropout")
    arrow(ax, left_x+bw2/2, H-2.78-0.02, left_x+bw2/2, H-2.78-bh-gap+0.02, C["head"])

    box(ax, left_x, H-3.26, bw2, bh, C["out"], "Softmax", "5 Class")
    ax.text(left_x+bw2/2, H-3.26-bh/2-0.06, "Path A: Classifier", ha="center", fontsize=6,
            color="#666", fontstyle="italic")

    # Profiling branch
    box(ax, right_x, H-2.78, bw2, bh, C["prof"], "User Metrics", "Skin+Bodyshape")
    arrow(ax, right_x+bw2/2, H-2.78-0.02, right_x+bw2/2, H-2.78-bh-gap+0.02, C["prof"])

    # Feature concat label
    cx_concat = W/2
    ax.text(cx_concat, H-2.35-0.25, "Feature Concatenation", ha="center", fontsize=7,
            fontweight="bold", color=C["prof"])
    ax.text(cx_concat, H-2.35-0.35, "\u2192 Concat at classifier input", ha="center",
            fontsize=6, color="#666", fontstyle="italic")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    fig.savefig(OUTPUT_DIR / "Fig4_unified_framework.png")
    plt.close(fig)
    print(f"Fig4: {W}\"x{H}\"")


if __name__ == "__main__":
    print("Generating clean paper figures...")
    fig1_architecture()
    fig2_workflow()
    fig4_unified_framework()
    print("Done.")
