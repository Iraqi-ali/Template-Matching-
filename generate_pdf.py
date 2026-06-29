"""
Generate Research PDF - Template Matching System
"""

from fpdf import FPDF
from pathlib import Path

DIAGRAMS_DIR = Path("diagrams")
OUTPUT = Path("Template_Matching_Research.pdf")
M = 20  # margin mm
FW = 210 - 2 * M  # full text width


class PDF(FPDF):
    def header(self):
        if self.page_no() <= 1:
            return
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(100, 100, 100)
        self.cell(95, 5, "Template Matching System - Research Documentation")
        self.cell(0, 5, f"Page {self.page_no()}", align="R")
        self.ln(8)

    def footer(self):
        if self.page_no() <= 1:
            return
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(130, 130, 130)
        self.cell(0, 10, "github.com/Iraqi-ali/Template-Matching-", align="C")

    def cover(self):
        self.add_page()
        self.ln(35)
        self.set_draw_color(58, 166, 255)
        self.set_line_width(1.5)
        self.line(30, 55, 180, 55)
        self.ln(20)
        self.set_font("Helvetica", "B", 30)
        self.set_text_color(30, 30, 30)
        self.cell(0, 12, "Template Matching", align="C")
        self.ln(14)
        self.cell(0, 12, "System", align="C")
        self.ln(18)
        self.set_font("Helvetica", "", 16)
        self.set_text_color(58, 166, 255)
        self.cell(0, 10, "Research & Technical Documentation", align="C")
        self.ln(16)
        self.set_draw_color(58, 166, 255)
        self.line(30, self.get_y(), 180, self.get_y())
        self.ln(16)
        self.set_font("Helvetica", "", 11)
        self.set_text_color(60, 60, 60)
        for t in [
            "Computer Vision  |  Image Processing  |  OpenCV  |  Python",
            "",
            "Version 1.0  -  June 2026",
            "Author: Ali  |  github.com/Iraqi-ali/Template-Matching-",
            "",
            "Includes: Architecture Diagrams, Flowcharts, Algorithm Details,",
            "Multi-Scale Matching, NMS, Web/GUI/CLI Interfaces, Render Deployment",
        ]:
            self.cell(0, 7, t, align="C")
            self.ln(8)
        self.ln(14)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(130, 130, 130)
        self.cell(0, 5, "Ready for deployment on Render Cloud", align="C")

    def ch(self, n, t):
        self.ln(6)
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(58, 166, 255)
        self.cell(0, 10, f"Chapter {n}: {t}")
        self.ln(12)
        self.set_draw_color(58, 166, 255)
        self.line(M, self.get_y(), self.w - M, self.get_y())
        self.ln(6)

    def sec(self, t):
        self.ln(4)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(30, 30, 30)
        self.cell(0, 8, t)
        self.ln(10)

    def p(self, t):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(50, 50, 50)
        self.set_x(M)
        self.multi_cell(FW, 5.5, t)
        self.ln(2)

    def b(self, t):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(50, 50, 50)
        self.set_x(M + 6)
        self.multi_cell(FW - 6, 5.5, "- " + t)

    def code(self, code):
        self.ln(2)
        self.set_fill_color(245, 245, 245)
        self.set_draw_color(200, 200, 200)
        self.set_font("Courier", "", 8.5)
        self.set_text_color(40, 40, 40)
        lines = code.strip().split("\n")
        bh = len(lines) * 4.5 + 6
        yb = self.get_y()
        if yb + bh > self.h - 25:
            self.add_page()
            yb = self.get_y()
        self.rect(M + 5, yb, FW - 10, bh, style="DF")
        self.set_xy(M + 9, yb + 3)
        for ln in lines:
            self.cell(0, 4.5, ln)
            self.ln(4.5)
            self.set_x(M + 9)
        self.set_y(yb + bh + 3)

    def tbl(self, rows, widths):
        for i, r in enumerate(rows):
            is_h = (i == 0)
            if is_h:
                self.set_fill_color(58, 166, 255)
                self.set_text_color(255, 255, 255)
            else:
                self.set_fill_color(248, 248, 248)
                self.set_text_color(50, 50, 50)
            self.set_font("Helvetica", "B" if is_h else "", 9)
            xs = self.get_x()
            for j, (cell, w) in enumerate(zip(r, widths)):
                self.set_x(xs + sum(widths[:j]))
                self.cell(w, 7, str(cell), fill=True, align="C" if j > 0 else "L")
            self.ln(7)

    def img(self, path, caption, w=170):
        self.ln(3)
        p = Path(path)
        if not p.exists():
            self.p(f"[Diagram not found: {path}]")
            return
        from PIL import Image
        im = Image.open(p)
        iw, ih = im.size
        h = w * ih / iw
        if self.get_y() + h > self.h - 30:
            self.add_page()
        x = (self.w - w) / 2
        self.image(str(p), x=x, w=w)
        self.ln(2)
        self.set_font("Helvetica", "I", 8.5)
        self.set_text_color(100, 100, 100)
        self.cell(0, 5, f"Figure: {caption}", align="C")
        self.ln(6)


# ================================================================

def build():
    pdf = PDF()
    pdf.set_margin(M)
    pdf.set_auto_page_break(True, 18)

    # --- Cover ---
    pdf.cover()

    # --- TOC ---
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 10, "Table of Contents")
    pdf.ln(12)
    for n, t in [
        ("1", "Introduction to Template Matching"),
        ("2", "System Architecture"),
        ("3", "Matching Algorithms & Methods"),
        ("4", "Multi-Scale Matching & NMS"),
        ("5", "System Workflow & Flowchart"),
        ("6", "User Interfaces (Web, GUI, CLI)"),
        ("7", "API Reference"),
        ("8", "Deployment on Render Cloud"),
        ("9", "Performance & Benchmarking"),
        ("10", "Conclusion & Future Work"),
    ]:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(50, 50, 50)
        pdf.cell(12, 8, n, align="R")
        pdf.cell(0, 8, t)
        pdf.ln(9)

    # ====== CHAPTER 1: Introduction ======
    pdf.add_page()
    pdf.ch("1", "Introduction to Template Matching")
    pdf.sec("1.1 What is Template Matching?")
    pdf.p(
        "Template Matching is a fundamental computer vision technique used to find a sub-image "
        "(the template) within a larger image (the source). It works by sliding the template "
        "pixel-by-pixel across the source image and computing a similarity metric at each position. "
        "Locations with high similarity scores indicate where the template appears in the source image."
    )
    pdf.p(
        "This technique is widely used in industrial inspection, object detection, medical imaging, "
        "video tracking, and document analysis. The system implements six different correlation methods "
        "provided by OpenCV's matchTemplate() function, along with multi-scale search and "
        "Non-Maximum Suppression (NMS) for robust, production-ready template matching."
    )
    pdf.sec("1.2 Applications")
    for a in [
        "Industrial Quality Control - detecting defects, verifying component placement",
        "Medical Imaging - locating anatomical structures in scans",
        "Object Tracking - following an object across video frames",
        "Document Analysis - finding logos, signatures, watermarks",
        "Augmented Reality - registering templates for AR overlays",
        "Robotics - visual servoing and object localization",
    ]:
        pdf.b(a)
    pdf.ln(4)
    pdf.sec("1.3 System Concept Map")
    pdf.p("The following mind map provides a comprehensive overview of the system components and their relationships:")
    pdf.img(str(DIAGRAMS_DIR / "01_mind_map.png"), "System Concept Map (Mind Map)", w=175)

    # ====== CHAPTER 2: Architecture ======
    pdf.add_page()
    pdf.ch("2", "System Architecture")
    pdf.sec("2.1 Overview")
    pdf.p(
        "The system follows a layered architecture with clear separation of concerns. "
        "It consists of four main layers: UI Layer, API Layer, Core Engine, and Infrastructure."
    )
    pdf.img(str(DIAGRAMS_DIR / "02_architecture.png"), "System Architecture - 4-Layer Design", w=175)

    pdf.sec("2.2 Layer Descriptions")
    pdf.p("UI Layer: Provides three interfaces for interacting with the system:")
    pdf.b("Web Interface - Flask-based with modern HTML/CSS/JS frontend, drag-and-drop upload")
    pdf.b("Desktop GUI - Tkinter application with built-in crop tool")
    pdf.b("CLI - Command-line interface for scripting and automation")
    pdf.ln(3)
    pdf.p("API Layer: RESTful endpoints handling image upload, matching, and result download:")
    pdf.b("/api/upload-source   - Upload the source (main) image")
    pdf.b("/api/upload-template - Upload the template image")
    pdf.b("/api/crop-template   - Extract template region from source")
    pdf.b("/api/match           - Execute template matching with configurable parameters")
    pdf.b("/api/compare         - Run all 6 methods for comparison")
    pdf.b("/api/download-result - Download the result image as PNG")
    pdf.ln(3)
    pdf.p("Core Engine: The computational heart of the system:")
    pdf.b("TemplateMatcher class    - Single-scale matching with any of 6 methods + NMS")
    pdf.b("MultiScaleMatcher class  - Multi-scale search across configurable scale range")
    pdf.b("Visualization module     - Draw bounding boxes, labels, and comparison grids")
    pdf.ln(3)
    pdf.p("Infrastructure:")
    pdf.b("OpenCV 4.8+ for cv2.matchTemplate()")
    pdf.b("Gunicorn as production WSGI server")
    pdf.b("Render Cloud for hosting (free tier)")

    # ====== CHAPTER 3: Methods ======
    pdf.add_page()
    pdf.ch("3", "Matching Algorithms & Methods")
    pdf.sec("3.1 Six Correlation Methods")
    pdf.p(
        "OpenCV's matchTemplate() provides six methods for computing the similarity between "
        "the template and each source image patch. Each method has different characteristics "
        "in terms of robustness to illumination changes, noise, and computational cost."
    )
    pdf.sec("3.2 Method Reference Table")
    pdf.tbl([
        ["Method", "Best For", "Score Range"],
        ["TM_CCOEFF", "General matching", "-inf to +inf"],
        ["TM_CCOEFF_NORMED", "Illumination changes (Recommended)", "-1.0 to 1.0"],
        ["TM_CCORR", "High contrast images", "0 to +inf"],
        ["TM_CCORR_NORMED", "Normalized cross-correlation", "0 to 1.0"],
        ["TM_SQDIFF", "Exact matches only", "0 to +inf (lower=better)"],
        ["TM_SQDIFF_NORMED", "Normalized squared difference", "0 to 1.0 (lower=better)"],
    ], [65, 65, 50])
    pdf.ln(4)
    pdf.p(
        "Recommendation: TM_CCOEFF_NORMED is the best general-purpose method. It is invariant "
        "to linear brightness changes and produces normalized scores in [-1, 1] where 1.0 "
        "indicates a perfect match. For the SQDIFF family, lower scores indicate better matches."
    )
    pdf.img(str(DIAGRAMS_DIR / "05_methods_comparison.png"), "Performance Comparison of 6 Matching Methods", w=170)
    pdf.sec("3.3 Mathematical Formulation")
    pdf.p("Correlation Coefficient (Normalized) formula:")
    pdf.p(
        "R(x,y) = Sum[(T - T_mean) * (I - I_mean)] / sqrt(Sum(T - T_mean)^2 * Sum(I - I_mean)^2)"
    )
    pdf.p(
        "Where T is the template, I is the source image region under the template, "
        "T_mean and I_mean are their respective mean pixel values. R(x,y) ranges from -1 to 1, "
        "where 1.0 signifies a perfect match."
    )

    # ====== CHAPTER 4: Multi-Scale ======
    pdf.add_page()
    pdf.ch("4", "Multi-Scale Matching & NMS")
    pdf.sec("4.1 Multi-Scale Approach")
    pdf.p(
        "In real-world scenarios, the template may appear at a different scale in the source "
        "image (due to camera distance, zoom, or object size variation). Single-scale matching "
        "would fail in such cases. The Multi-Scale Matcher solves this by resizing the template "
        "across a range of scales and performing matching at each scale."
    )
    pdf.p("Configuration parameters:")
    pdf.b("scale_range: (min_scale, max_scale) - default (0.3, 2.5) covers 30% to 250% of original size")
    pdf.b("scale_steps: Number of scale increments - default 25. More steps = finer search but slower execution")
    pdf.b("Template is resized at each scale using cv2.resize() with default interpolation")
    pdf.ln(3)
    pdf.sec("4.2 Non-Maximum Suppression (NMS)")
    pdf.p(
        "After matching, multiple overlapping detections may exist for the same object - "
        "especially in multi-scale mode where the same object may be detected at adjacent scales. "
        "NMS eliminates redundant detections by keeping only the highest-confidence box when "
        "two boxes overlap significantly."
    )
    pdf.p("IoU (Intersection over Union) calculation:")
    pdf.p("IoU(A,B) = |A intersect B| / |A union B|")
    pdf.p(
        "If IoU between two detections exceeds the NMS threshold (default 0.3), "
        "the detection with the lower confidence score is discarded. This typically reduces "
        "duplicate detections by 50-80% while preserving genuine matches."
    )
    pdf.img(str(DIAGRAMS_DIR / "04_multiscale_nms.png"), "Multi-Scale Matching at Different Scales and NMS Before/After", w=175)

    # ====== CHAPTER 5: Workflow ======
    pdf.add_page()
    pdf.ch("5", "System Workflow & Flowchart")
    pdf.sec("5.1 Step-by-Step Process")
    for title, desc in [
        ("1. Image Loading",
         "Source and template images are loaded from disk or uploaded via the web interface. "
         "Both color (BGR) and grayscale images are supported."),
        ("2. Grayscale Conversion",
         "Images are converted to grayscale using cv2.cvtColor() for computational efficiency. "
         "Color information is discarded as template matching operates on intensity values only."),
        ("3. Multi-Scale Check",
         "If multi-scale mode is enabled, the template is resized at N evenly-spaced scales "
         "within the configured range. Otherwise, single-scale matching proceeds directly."),
        ("4. Template Matching",
         "cv2.matchTemplate() is called with the selected method (e.g., TM_CCOEFF_NORMED). "
         "This produces a result map of size (W-w+1, H-h+1) where each pixel value is the "
         "similarity score at that offset position."),
        ("5. Threshold Filtering",
         "All locations where the score exceeds the threshold (or falls below it for SQDIFF "
         "methods) are extracted as candidate matches."),
        ("6. Non-Maximum Suppression",
         "Overlapping detections are filtered using NMS with a configurable IoU threshold "
         "to eliminate duplicates and keep only the best matches."),
        ("7. Visualization",
         "Bounding boxes are drawn on the source image at each match location. Confidence "
         "scores and match indices are overlaid. A summary bar displays method, threshold, "
         "match count, and execution time."),
    ]:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(50, 50, 50)
        pdf.set_x(M)
        pdf.cell(0, 6, title)
        pdf.ln(7)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_x(M + 6)
        pdf.multi_cell(FW - 6, 5.5, desc)
        pdf.ln(2)

    pdf.img(str(DIAGRAMS_DIR / "03_flowchart.png"), "Template Matching Algorithm Flowchart", w=130)

    # ====== CHAPTER 6: Interfaces ======
    pdf.add_page()
    pdf.ch("6", "User Interfaces (Web, GUI, CLI)")
    pdf.sec("6.1 Web Interface")
    pdf.p(
        "The web interface is a modern, bilingual (Arabic/English) single-page application "
        "built with Flask (backend) and vanilla HTML/CSS/JS (frontend)."
    )
    pdf.p("Key features:")
    for f in [
        "Drag-and-drop image upload with instant preview",
        "Interactive crop tool - select a region in the source to use as template",
        "Live parameter adjustment: method selector, threshold slider, multi-scale toggle",
        "Advanced settings panel: scale range, scale steps, NMS threshold",
        "Real-time result display with confidence scores",
        "Side-by-side comparison of all 6 matching methods",
        "One-click result download as PNG",
        "Keyboard shortcut: Ctrl+Enter to run matching",
        "Responsive dark-themed design - works on desktop and mobile",
    ]:
        pdf.b(f)
    pdf.ln(3)
    pdf.p("Launch command: python main.py web")
    pdf.p("Access at: http://127.0.0.1:5000")

    pdf.sec("6.2 Desktop GUI (Tkinter)")
    for f in [
        "Three-panel layout: Source Image | Template Image | Detection Result",
        "Built-in crop dialog for extracting template from source",
        "Real-time threshold adjustment with live label display",
        "Save result as PNG/JPEG file",
        "Threaded matching execution (non-blocking UI)",
    ]:
        pdf.b(f)
    pdf.ln(2)
    pdf.p("Launch command: python main.py gui")

    pdf.sec("6.3 Command Line Interface")
    pdf.p("The CLI supports scripting and batch processing:")
    pdf.code(
        "# Single match\n"
        "python main.py match source.jpg template.png --threshold 0.75\n\n"
        "# Multi-scale match\n"
        "python main.py match source.jpg template.png --multi-scale\n\n"
        "# Compare all 6 methods\n"
        "python main.py compare source.jpg template.png -o comparison.png\n\n"
        "# Generate test images\n"
        "python generate_test_images.py"
    )

    pdf.sec("6.4 Result Visualization Example")
    pdf.img(str(DIAGRAMS_DIR / "07_result_example.png"), "Sample Result - Detected Matches with Confidence Scores", w=165)

    # ====== CHAPTER 7: API Reference ======
    pdf.add_page()
    pdf.ch("7", "API Reference")
    pdf.sec("7.1 REST API Endpoints")
    pdf.tbl([
        ["Endpoint", "Description", "Parameters"],
        ["POST /api/upload-source", "Upload source image", "image: file (multipart)"],
        ["POST /api/upload-template", "Upload template image", "image: file"],
        ["POST /api/crop-template", "Crop template from source", "x, y, w, h (JSON)"],
        ["POST /api/match", "Execute template matching", "method, threshold (JSON)"],
        ["POST /api/compare", "Compare all 6 methods", "threshold (JSON)"],
        ["GET /api/download-result", "Download result PNG", "None"],
        ["GET /api/health", "Health check", "None"],
    ], [56, 72, 52])

    pdf.ln(4)
    pdf.sec("7.2 Example API Call")
    pdf.p("Match Request (POST /api/match):")
    pdf.code(
        '{\n'
        '  "method": "TM_CCOEFF_NORMED",\n'
        '  "threshold": 0.80,\n'
        '  "multi_scale": false,\n'
        '  "scale_min": 0.3,\n'
        '  "scale_max": 2.5,\n'
        '  "scale_steps": 25,\n'
        '  "nms_threshold": 0.3,\n'
        '  "max_matches": 50\n'
        '}'
    )
    pdf.ln(2)
    pdf.p("Match Response:")
    pdf.code(
        '{\n'
        '  "ok": true,\n'
        '  "match_count": 4,\n'
        '  "method": "Correlation Coefficient (Normalized)",\n'
        '  "elapsed_ms": 21.7,\n'
        '  "matches": [\n'
        '    {"id": 1, "x": 639, "y": 302, "confidence": 1.0000, "scale": 1.00},\n'
        '    {"id": 2, "x": 583, "y": 65,  "confidence": 0.7243, "scale": 1.00}\n'
        '  ],\n'
        '  "result_preview": "data:image/png;base64,...",\n'
        '  "source_shape": [800, 600],\n'
        '  "template_shape": [47, 73]\n'
        '}'
    )

    # ====== CHAPTER 8: Deployment ======
    pdf.add_page()
    pdf.ch("8", "Deployment on Render Cloud")
    pdf.sec("8.1 Deployment Pipeline")
    pdf.img(str(DIAGRAMS_DIR / "06_deployment.png"), "Render Cloud Deployment Pipeline", w=175)

    pdf.sec("8.2 Step-by-Step Deployment Guide")
    for title, desc in [
        ("Step 1: Push to GitHub",
         "Ensure all project files are committed and pushed to your GitHub repository: "
         "git add . && git commit -m \"Ready for Render\" && git push origin main"),
        ("Step 2: Log in to Render",
         "Go to dashboard.render.com and sign in with your GitHub account."),
        ("Step 3: Create Web Service",
         "Click 'New' then 'Web Service'. Connect your GitHub repository (Iraqi-ali/Template-Matching-)."),
        ("Step 4: Configure Service",
         "Render will auto-detect the render.yaml Blueprint file. If not, configure manually: "
         "Runtime=Python 3, Build=pip install -r requirements.txt, "
         "Start=gunicorn wsgi:app --bind 0.0.0.0:$PORT --timeout 120."),
        ("Step 5: Deploy",
         "Click 'Create Web Service'. Render builds and deploys automatically. "
         "Your app will be available at https://template-matching.onrender.com"),
    ]:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(50, 50, 50)
        pdf.set_x(M)
        pdf.cell(0, 6, title)
        pdf.ln(7)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_x(M + 6)
        pdf.multi_cell(FW - 6, 5, desc)
        pdf.ln(2)

    pdf.sec("8.3 render.yaml Configuration")
    pdf.code(
        "services:\n"
        "  - type: web\n"
        "    name: template-matching\n"
        "    runtime: python\n"
        "    plan: free\n"
        "    region: frankfurt\n"
        "    buildCommand: pip install -r requirements.txt\n"
        "    startCommand: gunicorn wsgi:app --bind 0.0.0.0:$PORT\n"
        "      --timeout 120 --workers 1 --threads 2\n"
        "    envVars:\n"
        "      - key: PYTHON_VERSION\n"
        "        value: \"3.11\""
    )

    pdf.sec("8.4 Important Notes")
    for n in [
        "Free tier: service sleeps after 15 minutes of inactivity (30-60 second cold start on wake)",
        "PORT is auto-assigned by Render; the application reads os.environ.get('PORT')",
        "Gunicorn with 1 worker + 2 threads is optimal for the 512 MB RAM free tier",
        "Maximum upload size: 16 MB (configurable in Flask app.config)",
        "OpenCV uses approximately 200 MB RAM in production; monitor usage via Render dashboard",
        "For better performance on free tier, keep source images under 1024x1024 pixels",
    ]:
        pdf.b(n)

    # ====== CHAPTER 9: Performance ======
    pdf.add_page()
    pdf.ch("9", "Performance & Benchmarking")
    pdf.sec("9.1 Benchmarks")
    pdf.p("Performance measured on an 800x600 source image with a 47x73 template on a standard CPU:")
    pdf.ln(2)
    pdf.tbl([
        ["Method", "Matches", "Time (ms)", "Speed", "Quality"],
        ["TM_CCOEFF", "45", "951", "Low", "Medium"],
        ["TM_CCOEFF_NORMED", "4", "19", "High", "Excellent"],
        ["TM_CCORR", "50", "1612", "Low", "Low"],
        ["TM_CCORR_NORMED", "49", "1619", "Low", "Medium"],
        ["TM_SQDIFF", "0", "10", "High", "Poor"],
        ["TM_SQDIFF_NORMED", "8", "111", "Medium", "Good"],
    ], [55, 25, 30, 30, 30])

    pdf.ln(4)
    pdf.sec("9.2 Performance Recommendations")
    for r in [
        "Always use NORMALIZED methods for reliable threshold-based filtering",
        "TM_CCOEFF_NORMED offers the best balance of speed, accuracy, and robustness",
        "Multi-scale mode adds approximately 25x overhead (25 scale steps = 25 matchTemplate calls)",
        "For production: set threshold to 0.75-0.85 to minimize false positives",
        "For large images (>1024px): resize to max 800px dimension before matching",
        "Unnormalized methods (CCOEFF, CCORR) are faster but produce less reliable scores",
    ]:
        pdf.b(r)

    # ====== CHAPTER 10: Conclusion ======
    pdf.add_page()
    pdf.ch("10", "Conclusion & Future Work")
    pdf.sec("10.1 Summary")
    pdf.p(
        "The Template Matching System provides a complete, production-ready solution for "
        "finding template images within source images. It implements six OpenCV correlation "
        "methods, supports multi-scale search, applies Non-Maximum Suppression to eliminate "
        "duplicate detections, and offers three user interfaces (Web, GUI, CLI) for maximum "
        "flexibility and ease of use."
    )
    pdf.p(
        "The system is fully deployable on Render Cloud with a one-click Blueprint configuration, "
        "making it accessible from anywhere with a web browser. The modular, layered architecture "
        "allows easy extension and integration into larger computer vision pipelines."
    )
    pdf.p(
        "Key achievements of this system include: (1) Comprehensive support for all 6 OpenCV "
        "matching methods, (2) Multi-scale matching for scale-invariant detection, "
        "(3) Production-grade NMS for clean results, (4) Bilingual web interface with "
        "intuitive UX, (5) Full REST API for programmatic access, and (6) One-click "
        "deployment to Render Cloud."
    )

    pdf.sec("10.2 Future Work")
    for item in [
        "GPU Acceleration - Use CUDA-enabled OpenCV for real-time matching on large images",
        "Rotation-Invariant Matching - Support rotated templates via angle-space search",
        "Feature-Based Matching - Combine with SIFT/ORB for more robust detection under transforms",
        "Batch Processing - Process multiple image/template pairs in parallel",
        "Docker Containerization - Add Dockerfile for consistent, portable deployment",
        "User Authentication - Add login system for multi-tenant deployments",
        "Video Stream Support - Real-time template matching on video frames",
        "Deep Learning Integration - Use YOLO/CNN for learned template features and semantic matching",
    ]:
        pdf.b(item)

    pdf.ln(12)
    pdf.set_draw_color(58, 166, 255)
    pdf.line(30, pdf.get_y(), 180, pdf.get_y())
    pdf.ln(8)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(58, 166, 255)
    pdf.cell(0, 8, "End of Document", align="C")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, "Template Matching System v1.0  |  June 2026", align="C")

    # --- SAVE ---
    pdf.output(str(OUTPUT))
    print(f"\n{'='*60}")
    print(f"  PDF successfully generated!")
    print(f"  File: {OUTPUT}")
    print(f"  Pages: {pdf.page_no()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    build()
