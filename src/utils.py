"""
Utility functions for image loading, resizing, and display.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union


# ------------------------------------------------------------------
# Image I/O
# ------------------------------------------------------------------

def load_image(path: Union[str, Path]) -> np.ndarray:
    """Load an image from disk. Raises FileNotFoundError if not found."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(str(p))
    if img is None:
        raise ValueError(f"Could not read image (possibly corrupted): {path}")
    return img


def save_image(img: np.ndarray, path: Union[str, Path]) -> None:
    """Save image to disk. Creates parent directories if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), img)


# ------------------------------------------------------------------
# Image utilities
# ------------------------------------------------------------------

def resize_to_max_dim(img: np.ndarray, max_dim: int = 800) -> np.ndarray:
    """Resize image so its largest dimension equals *max_dim*, preserving aspect ratio."""
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h))


def crop_roi(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Crop a rectangular region of interest from the image."""
    return img[y:y + h, x:x + w].copy()


def extract_template_from_source(
    source: np.ndarray,
    x: int, y: int, w: int, h: int,
) -> np.ndarray:
    """Extract a template from a source image at the given region."""
    return crop_roi(source, x, y, w, h)


# ------------------------------------------------------------------
# Visualisation helpers
# ------------------------------------------------------------------

def stack_images_horizontal(*images: np.ndarray) -> np.ndarray:
    """Horizontally stack images (they must have the same height)."""
    return np.hstack(images)


def stack_images_vertical(*images: np.ndarray) -> np.ndarray:
    """Vertically stack images (they must have the same width)."""
    return np.vstack(images)


def add_border(img: np.ndarray, thickness: int = 2, color=(0, 255, 0)) -> np.ndarray:
    """Add a coloured border around an image (in-place copy)."""
    return cv2.copyMakeBorder(
        img, thickness, thickness, thickness, thickness,
        cv2.BORDER_CONSTANT, value=color,
    )
