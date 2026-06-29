"""
Generate Research PDF for Template Matching System
====================================================
Creates a comprehensive PDF document with:
- Mind map / Concept map
- System architecture diagram
- Algorithm flowcharts
- Multi-scale and NMS illustrations
- Deployment diagram
- Step-by-step instructions
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc, Polygon
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "diagrams"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
})

# ─────────────────────────────────────────────────────────────────
# Color palette
# ─────────────────────────────────────────────────────────────────
C = {
    "bg":       "#0d1117",
    "surface":  "#161b22",
    "accent":   "#58a6ff",
    "green":    "#3fb950",
    "orange":   "#d2991d",
    "red":      "#f85149",
    "purple":   "#a371f7",
    "cyan":     "#39d2c0",
    "pink":     "#db61a2",
    "white":    "#e6edf3",
    "grey":     "#8b949e",
    "border":   "#30363d",
}

# ─────────────────────────────────────────────────────────────────
# Diagram 1: Mind Map / Concept Map
# ─────────────────────────────────────────────────────────────────

def generate_mind_map():
    """Generate a mind map showing the Template Matching system overview."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 11), facecolor=C["bg"])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11)
    ax.axis("off")

    def draw_node(x, y, text, color, size="large"):
        w, h = (3.2, 1.0) if size == "large" else (2.8, 0.8)
        box = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.15",
            facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.9
        )
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=9 if size == "large" else 8,
                fontweight="bold", color="white")

    def draw_arrow(x1, y1, x2, y2, color=C["grey"]):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.8, connectionstyle="arc3,rad=0.05"))

    # Central node
    draw_node(8, 5.5, "Template\nMatching\nSystem", C["accent"], "large")

    # Level 1 - branches
    branches = [
        (2.5, 8.8, "Matching\nMethods", C["purple"]),
        (13.5, 8.8, "Scale\nHandling", C["green"]),
        (2.5, 2.2, "User\nInterfaces", C["orange"]),
        (13.5, 2.2, "Deployment\n& Pipeline", C["cyan"]),
    ]
    for bx, by, bt, bc in branches:
        draw_node(bx, by, bt, bc)
        draw_arrow(8, 5.5, bx, by)

    # Level 2 - leaves
    leaves = [
        # Methods sub-branches
        (0.8, 10.2, "CCOEFF\n(Normalized)", C["purple"], "small"),
        (4.2, 10.2, "CCORR\n(Normalized)", C["purple"], "small"),
        (0.8, 7.8, "SQDIFF\n(Normalized)", C["purple"], "small"),
        (4.2, 7.8, "Comparison\nGrid", C["purple"], "small"),

        # Scale sub-branches
        (12.0, 10.2, "Multi-Scale\nMatching", C["green"], "small"),
        (15.0, 10.2, "Scale\nRange 0.3x-2.5x", C["green"], "small"),
        (12.0, 7.8, "NMS\nSuppression", C["green"], "small"),
        (15.0, 7.8, "Confidence\nThreshold", C["green"], "small"),

        # UI sub-branches
        (0.8, 3.5, "Web Interface\n(Flask)", C["orange"], "small"),
        (4.2, 3.5, "Desktop GUI\n(Tkinter)", C["orange"], "small"),
        (0.8, 0.9, "CLI\nCommands", C["orange"], "small"),
        (4.2, 0.9, "REST API\nEndpoints", C["orange"], "small"),

        # Deployment sub-branches
        (12.0, 3.5, "Render\nCloud", C["cyan"], "small"),
        (15.0, 3.5, "Gunicorn\nWSGI", C["cyan"], "small"),
        (12.0, 0.9, "GitHub\nCI/CD", C["cyan"], "small"),
        (15.0, 0.9, "Docker\n(optional)", C["cyan"], "small"),
    ]

    for lx, ly, lt, lc, ls in leaves:
        draw_node(lx, ly, lt, lc, ls)
        parent_y = ly + 1.0 if ly > 5.5 else ly - 1.0 if ly < 5.5 else ly
        parent_x = lx + 1.5 if lx < 8 else lx - 1.5
        draw_arrow(parent_x, parent_y, lx, ly)

    # Arrows from central to L1
    for bx, by, _, _ in branches:
        draw_arrow(8, 5.5, bx, by)

    ax.set_title("Template Matching System — Concept Map", fontsize=16, fontweight="bold",
                color=C["white"], pad=15)
    path = str(OUTPUT_DIR / "01_mind_map.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"✅ {path}")
    return path


# ─────────────────────────────────────────────────────────────────
# Diagram 2: System Architecture
# ─────────────────────────────────────────────────────────────────

def generate_architecture_diagram():
    """System architecture block diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 9), facecolor=C["bg"])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    def box(x, y, w, h, text, color, fontsize=9):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                          facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.85)
        ax.add_patch(b)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
               fontsize=fontsize, fontweight="bold", color="white")

    def arrow_h(x1, x2, y, color=C["grey"]):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                   arrowprops=dict(arrowstyle="->", color=color, lw=2))

    def arrow_v(x, y1, y2, color=C["grey"]):
        ax.annotate("", xy=(x, y2), xytext=(x, y1),
                   arrowprops=dict(arrowstyle="->", color=color, lw=2))

    # Layer labels
    ax.text(0.3, 7.7, "UI Layer", fontsize=11, color=C["grey"], fontweight="bold", rotation=90, va="center")
    ax.text(0.3, 5.3, "API Layer", fontsize=11, color=C["grey"], fontweight="bold", rotation=90, va="center")
    ax.text(0.3, 2.6, "Core Engine", fontsize=11, color=C["grey"], fontweight="bold", rotation=90, va="center")
    ax.text(0.3, 0.5, "Infrastructure", fontsize=11, color=C["grey"], fontweight="bold", rotation=90, va="center")

    # UI Layer
    box(1.2, 7.0, 3.0, 1.2, "Web Interface\nHTML/CSS/JS + Flask", C["orange"])
    box(4.8, 7.0, 3.0, 1.2, "Desktop GUI\nTkinter + Matplotlib", C["orange"])
    box(8.4, 7.0, 2.8, 1.2, "CLI\nargparse", C["orange"])

    # API Layer
    box(1.2, 4.7, 4.5, 1.0, "REST API Endpoints\n/api/match | /api/compare | /api/upload", C["purple"])
    box(6.3, 4.7, 4.5, 1.0, "Image Processing Pipeline\nload → decode → grayscale → match", C["purple"])

    # Core Engine
    box(1.2, 2.2, 3.3, 1.3, "Template Matcher\n6 Methods + NMS", C["accent"])
    box(4.9, 2.2, 3.3, 1.3, "Multi-Scale Matcher\nScale range + resampling", C["accent"])
    box(8.6, 2.2, 3.3, 1.3, "Visualization\nDraw bboxes + labels", C["accent"])

    # Infrastructure
    box(1.2, 0.2, 3.3, 1.0, "OpenCV 4.8+\nmatchTemplate()", C["green"])
    box(4.9, 0.2, 3.3, 1.0, "Gunicorn + Flask\nWSGI Server", C["green"])
    box(8.6, 0.2, 3.3, 1.0, "Render Cloud\nFree Tier", C["green"])

    # Data flow arrows
    arrow_v(2.7, 8.2, 5.7)
    arrow_v(6.3, 8.2, 5.7)
    arrow_v(3.5, 5.7, 3.5)
    arrow_v(6.5, 3.5, 1.2)
    arrow_v(9.5, 3.5, 1.2)

    ax.set_title("System Architecture Diagram", fontsize=16, fontweight="bold",
                color=C["white"], pad=12)
    path = str(OUTPUT_DIR / "02_architecture.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"✅ {path}")
    return path


# ─────────────────────────────────────────────────────────────────
# Diagram 3: Algorithm Flowchart
# ─────────────────────────────────────────────────────────────────

def generate_flowchart():
    """Generate the matching algorithm flowchart."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 14), facecolor=C["bg"])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")

    def node(x, y, w, h, text, color, shape="round", fs=8):
        if shape == "round":
            b = FancyBboxPatch((x - w/2, y - h/2), w, h,
                              boxstyle="round,pad=0.15", facecolor=color,
                              edgecolor="white", linewidth=1.5, alpha=0.9)
        elif shape == "diamond":
            pts = [(x, y + h/2), (x + w/2, y), (x, y - h/2), (x - w/2, y)]
            b = Polygon(pts, facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.9)
        else:
            b = FancyBboxPatch((x - w/2, y - h/2), w, h,
                              boxstyle="square,pad=0.1", facecolor=color,
                              edgecolor="white", linewidth=1.5, alpha=0.9)
        ax.add_patch(b)
        ax.text(x, y, text, ha="center", va="center", fontsize=fs,
               fontweight="bold", color="white")

    def arrow_d(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle="->", color=C["grey"], lw=2))

    # Start
    node(5, 13.2, 2.4, 0.8, "START", C["green"], "round", 10)
    arrow_d(5, 12.8, 5, 12.1)

    # Load
    node(5, 11.5, 4.2, 0.8, "Load Source & Template Images", C["accent"], "round", 9)
    arrow_d(5, 11.1, 5, 10.5)

    # Grayscale
    node(5, 9.9, 3.8, 0.8, "Convert to Grayscale", C["accent"], "round", 9)
    arrow_d(5, 9.5, 5, 8.9)

    # Multi-scale?
    node(5, 8.3, 4.0, 0.8, "Multi-Scale Enabled?", C["orange"], "diamond", 8)

    # Yes branch -> right
    ax.text(7.6, 8.7, "YES", fontsize=8, color=C["green"], fontweight="bold")
    node(9, 7.5, 3.0, 0.8, "Resize Template\nat N scales", C["purple"], "round", 8)
    arrow_d(7.0, 8.3, 8.0, 7.9)
    arrow_d(9, 7.1, 9, 6.5)
    node(9, 5.9, 3.0, 0.8, "Match at\neach scale", C["purple"], "round", 8)
    arrow_d(8.0, 5.9, 7.0, 5.9)

    # No branch -> straight down
    ax.text(5.5, 7.7, "NO", fontsize=8, color=C["red"], fontweight="bold")
    arrow_d(5, 7.9, 5, 7.3)

    # cv2.matchTemplate
    node(5, 6.7, 4.0, 0.8, "cv2.matchTemplate()", C["accent"], "round", 10)
    arrow_d(5, 6.3, 5, 5.9)

    # Threshold filter
    node(5, 5.3, 4.0, 0.8, "Filter by Threshold", C["accent"], "round", 9)
    arrow_d(5, 4.9, 5, 4.5)

    # Matches found?
    node(5, 3.9, 3.8, 0.8, "Matches Found?", C["orange"], "diamond", 8)
    ax.text(7.4, 3.9, "YES", fontsize=8, color=C["green"], fontweight="bold")
    ax.text(3.4, 3.4, "NO", fontsize=8, color=C["red"], fontweight="bold")

    arrow_d(5, 3.5, 5, 3.1)

    # NMS
    node(5, 2.5, 3.6, 0.8, "Apply NMS\n(Non-Max Suppression)", C["purple"], "round", 8)
    arrow_d(5, 2.1, 5, 1.7)

    # Draw
    node(5, 1.1, 3.6, 0.8, "Draw Bounding Boxes\n& Return Results", C["green"], "round", 8)
    arrow_d(5, 0.7, 5, 0.3)
    node(5, 0.15, 1.5, 0.5, "END", C["red"], "round", 10)

    # No matches path
    node(2.5, 3.9, 1.8, 0.5, "Return\nEmpty", C["red"], "round", 7)
    arrow_d(3.1, 3.9, 2.0, 3.9)

    ax.set_title("Template Matching Algorithm Flowchart", fontsize=14, fontweight="bold",
                color=C["white"], pad=10)
    path = str(OUTPUT_DIR / "03_flowchart.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"✅ {path}")
    return path


# ─────────────────────────────────────────────────────────────────
# Diagram 4: Multi-Scale + NMS Illustration
# ─────────────────────────────────────────────────────────────────

def generate_multiscale_diagram():
    """Illustrate multi-scale matching and NMS."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), facecolor=C["bg"])
    fig.suptitle("Multi-Scale Matching & Non-Maximum Suppression", fontsize=14,
                fontweight="bold", color=C["white"], y=0.98)

    np.random.seed(42)

    for i, (scale, title) in enumerate([
        (0.5, "Scale 0.5x"), (1.0, "Scale 1.0x (Original)"), (2.0, "Scale 2.0x"),
    ]):
        ax = axes[0, i]
        ax.set_facecolor(C["bg"])

        # Simulate a heatmap
        heatmap = np.zeros((80, 80))
        cx, cy = 30 + int(i * 15), 35 + int(i * 5)
        for y in range(80):
            for x in range(80):
                heatmap[y, x] = np.exp(-((x-cx)**2 + (y-cy)**2) / 150) + 0.05 * np.random.random()
                if i == 1:
                    heatmap[y, x] += 0.3 * np.exp(-((x-50)**2 + (y-60)**2) / 80)

        im = ax.imshow(heatmap, cmap="hot", aspect="auto")
        ax.set_title(f"{title}\nTemplate: {int(47*scale)}x{int(73*scale)} px", color=C["white"], fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

        # Mark peak
        if i == 0:
            ax.plot(cx, cy, "go", markersize=10, markeredgecolor="white", markeredgewidth=1.5)
        elif i == 1:
            ax.plot(cx, cy, "go", markersize=10, markeredgecolor="white", markeredgewidth=1.5)
            ax.plot(50, 60, "co", markersize=8, markeredgecolor="white", markeredgewidth=1)

    # NMS illustration
    ax_nms = axes[1, 0]
    ax_nms.set_facecolor(C["bg"])
    ax_nms.set_xlim(0, 10)
    ax_nms.set_ylim(0, 10)
    ax_nms.axis("off")
    ax_nms.set_title("Before NMS\n(Overlapping detections)", color=C["white"], fontsize=9)

    # Draw overlapping boxes
    boxes_before = [
        (2, 3, 4, 3, C["red"]),
        (3, 4, 4, 3, C["orange"]),
        (2.5, 3.5, 3.5, 2.5, C["red"]),
        (7, 6, 2, 2, C["green"]),
    ]
    for x, y, w, h, c in boxes_before:
        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor=c, linewidth=2)
        ax_nms.add_patch(rect)

    ax_nms_after = axes[1, 1]
    ax_nms_after.set_facecolor(C["bg"])
    ax_nms_after.set_xlim(0, 10)
    ax_nms_after.set_ylim(0, 10)
    ax_nms_after.axis("off")
    ax_nms_after.set_title("After NMS\n(Only best kept)", color=C["white"], fontsize=9)

    boxes_after = [
        (2.5, 3.5, 3.5, 2.5, C["green"]),
        (7, 6, 2, 2, C["green"]),
    ]
    for x, y, w, h, c in boxes_after:
        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor=c, linewidth=2.5)
        ax_nms_after.add_patch(rect)

    # NMS formula
    ax_formula = axes[1, 2]
    ax_formula.set_facecolor(C["bg"])
    ax_formula.axis("off")
    ax_formula.set_title("IoU Formula", color=C["white"], fontsize=9)

    formula_text = (
        "IoU(A,B) = |A ∩ B| / |A ∪ B|\n\n"
        "If IoU > threshold (0.3):\n"
        "  → Keep box with\n"
        "    higher confidence\n\n"
        "NMS reduces duplicate\n"
        "detections by 50-80%"
    )
    ax_formula.text(0.5, 0.5, formula_text, transform=ax_formula.transAxes,
                   ha="center", va="center", fontsize=10, color=C["white"],
                   fontfamily="monospace",
                   bbox=dict(boxstyle="round", facecolor=C["surface"], edgecolor=C["border"], pad=1.0))

    plt.tight_layout()
    path = str(OUTPUT_DIR / "04_multiscale_nms.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"✅ {path}")
    return path


# ─────────────────────────────────────────────────────────────────
# Diagram 5: Matching Methods Comparison Chart
# ─────────────────────────────────────────────────────────────────

def generate_methods_chart():
    """Generate a comparison chart for the 6 matching methods."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6), facecolor=C["bg"])
    ax.set_facecolor(C["bg"])

    methods = [
        "TM_CCOEFF", "TM_CCOEFF\n_NORMED",
        "TM_CCORR", "TM_CCORR\n_NORMED",
        "TM_SQDIFF", "TM_SQDIFF\n_NORMED",
    ]
    # Simulated performance metrics
    accuracy = [0.65, 0.95, 0.55, 0.88, 0.42, 0.90]
    speed =    [0.90, 0.88, 0.85, 0.83, 0.95, 0.92]
    robustness=[0.60, 0.93, 0.50, 0.85, 0.35, 0.88]

    x = np.arange(len(methods))
    w = 0.25

    bars1 = ax.bar(x - w, accuracy, w, label="Accuracy", color=C["accent"], alpha=0.85)
    bars2 = ax.bar(x, speed, w, label="Speed", color=C["green"], alpha=0.85)
    bars3 = ax.bar(x + w, robustness, w, label="Robustness", color=C["purple"], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=8, color=C["white"])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score (normalized)", color=C["white"], fontsize=10)
    ax.legend(facecolor=C["surface"], edgecolor=C["border"], labelcolor=C["white"], fontsize=9)

    # Style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C["border"])
    ax.spines["bottom"].set_color(C["border"])
    ax.tick_params(colors=C["grey"])
    ax.yaxis.grid(True, alpha=0.2, color=C["grey"])

    # Highlight best
    ax.annotate("★ Best Overall", xy=(1, accuracy[1]), xytext=(0, 1.07),
               arrowprops=dict(arrowstyle="->", color=C["orange"], lw=2),
               fontsize=9, color=C["orange"], fontweight="bold", ha="center")

    ax.set_title("Matching Methods — Performance Comparison", fontsize=14,
                fontweight="bold", color=C["white"], pad=12)
    plt.tight_layout()
    path = str(OUTPUT_DIR / "05_methods_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"✅ {path}")
    return path


# ─────────────────────────────────────────────────────────────────
# Diagram 6: Deployment Pipeline
# ─────────────────────────────────────────────────────────────────

def generate_deployment_diagram():
    """Render deployment pipeline diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 6), facecolor=C["bg"])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.axis("off")

    steps = [
        (1.0, "Local\nDevelopment", "Python + OpenCV\nFlask Dev Server", C["purple"]),
        (4.0, "Git Push\nGitHub", "git push origin main\nSource Control", C["accent"]),
        (7.0, "Render\nDetects Push", "Auto-deploy trigger\nrender.yaml", C["orange"]),
        (10.0, "Build &\nInstall", "pip install -r\nrequirements.txt", C["cyan"]),
        (13.0, "Run with\nGunicorn", "gunicorn wsgi:app\n--bind 0.0.0.0:$PORT", C["green"]),
    ]

    for x, title, desc, color in steps:
        # Circle
        circle = plt.Circle((x + 1.0, 3.5), 1.1, facecolor=color, edgecolor="white", linewidth=2, alpha=0.9)
        ax.add_patch(circle)
        ax.text(x + 1.0, 3.9, title, ha="center", va="center", fontsize=10,
               fontweight="bold", color="white")
        ax.text(x + 1.0, 3.0, desc, ha="center", va="center", fontsize=7,
               color="white", alpha=0.85)

    # Arrows between circles
    for i in range(len(steps) - 1):
        x1 = steps[i][0] + 2.1
        x2 = steps[i+1][0] - 0.1
        ax.annotate("", xy=(x2, 3.5), xytext=(x1, 3.5),
                   arrowprops=dict(arrowstyle="->", color=C["grey"], lw=2.5,
                                  connectionstyle="arc3,rad=0"))

    ax.set_title("Render Deployment Pipeline", fontsize=16, fontweight="bold",
                color=C["white"], pad=10)

    # Labels at bottom
    labels = ["1. Develop", "2. Commit", "3. Auto-detect", "4. Build", "5. Deploy"]
    for i, (x, _, _, _) in enumerate(steps):
        ax.text(x + 1.0, 1.2, labels[i], ha="center", fontsize=9, color=C["grey"], fontweight="bold")

    path = str(OUTPUT_DIR / "06_deployment.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"✅ {path}")
    return path


# ─────────────────────────────────────────────────────────────────
# Diagram 7: Result Visualization Example
# ─────────────────────────────────────────────────────────────────

def generate_result_example():
    """Generate a synthetic result visualization example."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), facecolor=C["bg"])
    ax.set_facecolor(C["bg"])

    # Synthetic "source" image
    np.random.seed(123)
    img = np.random.randint(30, 80, (400, 600, 3), dtype=np.uint8)

    # Draw shapes
    shapes = [
        (50, 60, 80, 60, (200, 80, 80)),
        (250, 100, 60, 70, (80, 200, 80)),
        (450, 50, 70, 50, (80, 80, 200)),
        (150, 250, 90, 80, (200, 200, 80)),
        (380, 280, 75, 65, (200, 80, 200)),
        (70, 180, 55, 55, (80, 200, 200)),
    ]

    # Template to match (green rectangle)
    template_color = (80, 200, 80)

    for x, y, w, h, color in shapes:
        img[y:y+h, x:x+w] = color

    ax.imshow(img)

    # Draw "detected" bounding boxes (simulating results)
    detections = [
        (50, 60, 80, 60, 0.99, "#3fb950"),
        (250, 100, 60, 70, 0.92, "#3fb950"),
        (148, 248, 94, 84, 0.78, "#d2991d"),
        (383, 282, 69, 59, 0.65, "#d2991d"),
    ]

    for i, (x, y, w, h, conf, color) in enumerate(detections):
        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=2.5)
        ax.add_patch(rect)
        ax.annotate(f"#{i+1} {conf:.2f}", (x, y - 8),
                   color=color, fontsize=9, fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.2", facecolor=C["bg"], edgecolor=color, alpha=0.9))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Result Visualization Example\nDetected matches with confidence scores",
                fontsize=13, color=C["white"], pad=10)

    plt.tight_layout()
    path = str(OUTPUT_DIR / "07_result_example.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"✅ {path}")
    return path


# ─────────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🎨 Generating diagrams for PDF...\n")
    diagrams = {
        "mind_map": generate_mind_map(),
        "architecture": generate_architecture_diagram(),
        "flowchart": generate_flowchart(),
        "multiscale": generate_multiscale_diagram(),
        "methods_chart": generate_methods_chart(),
        "deployment": generate_deployment_diagram(),
        "result_example": generate_result_example(),
    }
    print(f"\n✅ All {len(diagrams)} diagrams generated in: {OUTPUT_DIR}")
