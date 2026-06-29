"""
Template Matching GUI Application
=================================
A Tkinter-based GUI for interactive template matching.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from pathlib import Path
from typing import Optional
import threading

from .template_matcher import (
    TemplateMatcher, MatchMethod, DetectionReport,
    draw_matches, draw_match_comparison, draw_match_with_differences,
)
from .multi_scale_matcher import MultiScaleMatcher
from .forgery_detector import (
    ForgeryDetector, draw_difference_boxes, draw_forgery_report,
    create_full_analysis_canvas,
)
from .utils import load_image, resize_to_max_dim


# ---------------------------------------------------------------------------
# Colour theme
# ---------------------------------------------------------------------------
BG_COLOR = "#1e1e2e"
FG_COLOR = "#cdd6f4"
ACCENT = "#89b4fa"
BUTTON_BG = "#313244"
BUTTON_FG = "#cdd6f4"
CANVAS_BG = "#11111b"
SUCCESS = "#a6e3a1"
WARN = "#f9e2af"


class TemplateMatchingApp:
    """Main GUI application window."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Template Matching")
        self.root.geometry("1280x780")
        self.root.configure(bg=BG_COLOR)
        self.root.minsize(1024, 680)

        # State
        self.source_img: Optional[np.ndarray] = None
        self.template_img: Optional[np.ndarray] = None
        self.result_img: Optional[np.ndarray] = None
        self.last_report: Optional[DetectionReport] = None

        # Method mapping
        self.method_names = [m.label for m in MatchMethod]
        self.method_map = {m.label: m for m in MatchMethod}

        self._build_ui()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        """Build all UI elements."""

        # ---- Top toolbar ----
        toolbar = tk.Frame(self.root, bg=BG_COLOR, padx=8, pady=6)
        toolbar.pack(fill=tk.X)

        btn_style = {
            "bg": BUTTON_BG, "fg": BUTTON_FG, "relief": tk.FLAT,
            "activebackground": ACCENT, "activeforeground": "#000",
            "padx": 12, "pady": 4, "font": ("Segoe UI", 9),
        }

        tk.Button(toolbar, text="📂 Load Source", command=self._load_source, **btn_style).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="📁 Load Template", command=self._load_template, **btn_style).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="✂️  Crop Template from Source", command=self._crop_template, **btn_style).pack(side=tk.LEFT, padx=2)

        tk.Label(toolbar, text="  Method:", bg=BG_COLOR, fg=FG_COLOR, font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(16, 2))
        self.method_var = tk.StringVar(value=self.method_names[1])  # TM_CCOEFF_NORMED
        method_menu = ttk.Combobox(
            toolbar, textvariable=self.method_var, values=self.method_names,
            state="readonly", width=28,
        )
        method_menu.pack(side=tk.LEFT, padx=2)

        tk.Label(toolbar, text="  Threshold:", bg=BG_COLOR, fg=FG_COLOR, font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(12, 2))
        self.threshold_var = tk.DoubleVar(value=0.80)
        thresh_scale = tk.Scale(
            toolbar, from_=0.10, to=0.99, resolution=0.01, orient=tk.HORIZONTAL,
            variable=self.threshold_var, length=140,
            bg=BG_COLOR, fg=FG_COLOR, troughcolor=BUTTON_BG,
            activebackground=ACCENT, highlightthickness=0,
        )
        thresh_scale.pack(side=tk.LEFT, padx=2)

        self.threshold_label = tk.Label(
            toolbar, text="0.80", bg=BG_COLOR, fg=ACCENT,
            font=("Segoe UI", 9, "bold"), width=4,
        )
        self.threshold_label.pack(side=tk.LEFT, padx=2)
        self.threshold_var.trace_add("write", lambda *_: self.threshold_label.configure(text=f"{self.threshold_var.get():.2f}"))

        self.multi_scale_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            toolbar, text="Multi-Scale", variable=self.multi_scale_var,
            bg=BG_COLOR, fg=FG_COLOR, selectcolor=BUTTON_BG,
            activebackground=BG_COLOR, activeforeground=ACCENT,
            font=("Segoe UI", 9),
        ).pack(side=tk.LEFT, padx=(12, 2))

        tk.Button(toolbar, text="🔍 RUN MATCHING", command=self._run_matching,
                  bg=SUCCESS, fg="#000", relief=tk.FLAT,
                  activebackground="#94e2d5", font=("Segoe UI", 9, "bold"),
                  padx=16, pady=4).pack(side=tk.RIGHT, padx=6)

        tk.Button(toolbar, text="💾 Save Result", command=self._save_result, **btn_style).pack(side=tk.RIGHT, padx=2)

        # ---- Main content area (3 panels) ----
        content = tk.Frame(self.root, bg=BG_COLOR)
        content.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # Panel proportions
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=1)
        content.columnconfigure(2, weight=1)
        content.rowconfigure(0, weight=1)

        # Source image panel
        self._create_image_panel(content, "Source Image", 0, 0)
        # Template image panel
        self._create_image_panel(content, "Template Image", 1, 0)
        # Result panel
        self._create_image_panel(content, "Detection Result", 2, 0)

        # ---- Status bar ----
        status_frame = tk.Frame(self.root, bg="#181825", height=26)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_label = tk.Label(
            status_frame, text=" Ready — Load source and template images to begin.",
            bg="#181825", fg="#a6adc8", anchor=tk.W, font=("Segoe UI", 8),
        )
        self.status_label.pack(fill=tk.X, padx=10, pady=3)

    def _create_image_panel(self, parent: tk.Frame, title: str, col: int, row: int):
        """Create a labelled image panel with a scrollable canvas."""
        frame = tk.Frame(parent, bg="#181825", bd=0, highlightthickness=0)
        frame.grid(row=row, column=col, sticky="nsew", padx=4, pady=2)
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)

        header = tk.Frame(frame, bg="#181825")
        header.grid(row=0, column=0, sticky="ew", pady=(4, 2))
        tk.Label(
            header, text=f"  {title}", bg="#181825", fg=FG_COLOR,
            font=("Segoe UI", 10, "bold"),
        ).pack(side=tk.LEFT)

        info_label = tk.Label(
            header, text="", bg="#181825", fg="#a6adc8",
            font=("Segoe UI", 8),
        )
        info_label.pack(side=tk.RIGHT, padx=8)

        canvas = tk.Canvas(frame, bg=CANVAS_BG, highlightthickness=0)
        canvas.grid(row=1, column=0, sticky="nsew")

        # Store references
        if "source" in title.lower():
            self.source_canvas = canvas
            self.source_info = info_label
        elif "template" in title.lower():
            self.template_canvas = canvas
            self.template_info = info_label
        else:
            self.result_canvas = canvas
            self.result_info = info_label

    # ------------------------------------------------------------------
    # Image loading & display
    # ------------------------------------------------------------------

    def _load_source(self):
        path = filedialog.askopenfilename(
            title="Select Source Image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("All", "*.*")],
        )
        if not path:
            return
        try:
            self.source_img = load_image(path)
            self.source_img = resize_to_max_dim(self.source_img, 600)
            self._show_image(self.source_img, self.source_canvas, self.source_info, "Source")
            self.status_label.configure(text=f" Source loaded: {Path(path).name}  |  {self.source_img.shape[1]}×{self.source_img.shape[0]}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _load_template(self):
        path = filedialog.askopenfilename(
            title="Select Template Image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("All", "*.*")],
        )
        if not path:
            return
        try:
            self.template_img = load_image(path)
            self._show_image(self.template_img, self.template_canvas, self.template_info, "Template")
            self.status_label.configure(text=f" Template loaded: {Path(path).name}  |  {self.template_img.shape[1]}×{self.template_img.shape[0]}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _crop_template(self):
        """Allow user to select a region in the source image as template."""
        if self.source_img is None:
            messagebox.showwarning("No Source", "Please load a source image first.")
            return

        # Simple approach: use a dialog
        dialog = CropDialog(self.root, self.source_img)
        self.root.wait_window(dialog.top)

        if dialog.result is not None:
            self.template_img = dialog.result
            self._show_image(self.template_img, self.template_canvas, self.template_info, "Template")
            self.status_label.configure(text=f" Template cropped from source  |  {self.template_img.shape[1]}×{self.template_img.shape[0]}")

    def _show_image(self, img: np.ndarray, canvas: tk.Canvas, info_label: tk.Label, label: str):
        """Display an OpenCV image on a tkinter Canvas."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        pil_img = Image.fromarray(img_rgb)
        imgtk = ImageTk.PhotoImage(pil_img)

        canvas.delete("all")
        canvas.configure(width=w, height=h, scrollregion=(0, 0, w, h))
        canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        canvas.image = imgtk  # keep reference

        info_label.configure(text=f"{label}  |  {w}×{h}")

    # ------------------------------------------------------------------
    # Matching execution
    # ------------------------------------------------------------------

    def _run_matching(self):
        if self.source_img is None or self.template_img is None:
            messagebox.showwarning("Missing Images", "Please load both source and template images.")
            return

        method_label = self.method_var.get()
        method = self.method_map[method_label]
        threshold = self.threshold_var.get()
        multi_scale = self.multi_scale_var.get()

        self.status_label.configure(text=" Running matching with validation...")

        def _worker():
            try:
                if multi_scale:
                    matcher = MultiScaleMatcher(
                        method=method, threshold=threshold,
                        scale_range=(0.3, 2.5), scale_steps=25,
                    )
                    report = matcher.match(self.source_img, self.template_img)
                else:
                    matcher = TemplateMatcher(
                        method=method, threshold=threshold,
                        validate_matches=True, strict_validation=True,
                    )
                    report = matcher.match(self.source_img, self.template_img)

                self.last_report = report

                # Generate result with difference boxes
                result_img = draw_match_with_differences(
                    self.source_img, self.template_img, report,
                )

                # Also run forgery analysis if matches found
                if report.match_count > 0:
                    detector = ForgeryDetector()
                    matched_regions = [(m.x, m.y, m.width, m.height) for m in report.matches]
                    forgery_report = detector.analyze(
                        self.source_img, self.template_img, matched_regions,
                    )
                    self.last_forgery = forgery_report
                else:
                    self.last_forgery = None

                self.root.after(0, lambda: self._show_result(result_img, report))
            except Exception as e:
                self.root.after(0, lambda: self._on_error(str(e)))

        threading.Thread(target=_worker, daemon=True).start()

    def _show_result(self, result_img: np.ndarray, report: DetectionReport):
        self.result_img = result_img
        self._show_image(result_img, self.result_canvas, self.result_info, "Result")

        # Build status with validation info
        validation_status = ""
        if report.validation is not None:
            if report.validation.is_valid:
                validation_status = " ✓ GENUINE MATCH"
            else:
                validation_status = " ⚠ CHECK VALIDATION"

        forgery_status = ""
        if hasattr(self, 'last_forgery') and self.last_forgery is not None:
            forgery_status = f" | Forgery Risk: {self.last_forgery.risk_level.label}"

        status = (
            f" {report.match_count} match(es){validation_status}  "
            f"|  Method: {report.method.label}  "
            f"|  Threshold: {report.threshold:.2f}  "
            f"|  {report.elapsed_ms:.1f} ms"
            f"{forgery_status}"
        )
        self.status_label.configure(text=status)

    def _on_error(self, msg: str):
        self.status_label.configure(text=f" ❌ Error: {msg}")
        messagebox.showerror("Matching Error", msg)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _save_result(self):
        if self.result_img is None:
            messagebox.showwarning("No Result", "Run matching first to generate a result.")
            return
        path = filedialog.asksaveasfilename(
            title="Save Result Image",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All", "*.*")],
        )
        if path:
            cv2.imwrite(path, self.result_img)
            self.status_label.configure(text=f" 💾 Saved to: {Path(path).name}")


# ---------------------------------------------------------------------------
# Crop dialog for extracting template from source
# ---------------------------------------------------------------------------

class CropDialog:
    """A simple dialog to let the user draw a rectangle on the source image."""

    def __init__(self, parent: tk.Tk, source_img: np.ndarray):
        self.result: Optional[np.ndarray] = None
        self.source_img = source_img
        self.start_x = self.start_y = 0
        self.rect_id = None

        self.top = tk.Toplevel(parent)
        self.top.title("Crop Template — Drag to select region")
        self.top.configure(bg=BG_COLOR)
        self.top.resizable(False, False)

        img_rgb = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        self.tk_img = ImageTk.PhotoImage(pil_img)
        h, w = source_img.shape[:2]

        self.canvas = tk.Canvas(self.top, width=w, height=h, bg=CANVAS_BG, highlightthickness=0)
        self.canvas.pack(padx=4, pady=4)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        btn_frame = tk.Frame(self.top, bg=BG_COLOR)
        btn_frame.pack(pady=4)
        tk.Button(btn_frame, text="Confirm Crop", command=self._confirm,
                  bg=SUCCESS, fg="#000", relief=tk.FLAT, padx=16, pady=4,
                  font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="Cancel", command=self.top.destroy,
                  bg=BUTTON_BG, fg=BUTTON_FG, relief=tk.FLAT, padx=16, pady=4).pack(side=tk.LEFT, padx=4)

    def _on_press(self, event):
        self.start_x, self.start_y = event.x, event.y
        if self.rect_id:
            self.canvas.delete(self.rect_id)

    def _on_drag(self, event):
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        x1, y1 = self.start_x, self.start_y
        self.rect_id = self.canvas.create_rectangle(
            x1, y1, event.x, event.y, outline=SUCCESS, width=2, dash=(5, 3),
        )

    def _on_release(self, event):
        pass

    def _confirm(self):
        if self.rect_id is None:
            return
        coords = self.canvas.coords(self.rect_id)
        if len(coords) != 4:
            return
        x1, y1, x2, y2 = map(int, coords)
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        w, h = x2 - x1, y2 - y1
        if w < 5 or h < 5:
            messagebox.showwarning("Too Small", "Selection is too small. Drag a larger rectangle.")
            return
        self.result = self.source_img[y1:y2, x1:x2].copy()
        self.top.destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_gui():
    """Launch the GUI application."""
    root = tk.Tk()
    app = TemplateMatchingApp(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()
