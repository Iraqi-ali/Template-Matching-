"""
Generate synthetic test images for Template Matching demos.
Creates a source image with random shapes and a template extracted from it.
"""

import cv2
import numpy as np
from pathlib import Path
import random


def create_test_images(
    output_dir: str = "examples",
    source_w: int = 800,
    source_h: int = 600,
    num_objects: int = 12,
) -> tuple:
    """
    Generate a synthetic source image with coloured shapes and a template
    that matches one of the shapes.

    Returns:
        (source_path, template_path)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Background with subtle gradient
    source = np.zeros((source_h, source_w, 3), dtype=np.uint8)
    for y in range(source_h):
        shade = int(40 + 20 * y / source_h)
        source[y, :] = (shade, shade, shade)

    template_info = None

    for i in range(num_objects):
        x = random.randint(30, source_w - 80)
        y = random.randint(30, source_h - 80)
        w = random.randint(40, 100)
        h = random.randint(40, 100)
        colour = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255),
        )

        shape_type = random.choice(["rect", "circle", "triangle"])

        if shape_type == "rect":
            cv2.rectangle(source, (x, y), (x + w, y + h), colour, -1)
        elif shape_type == "circle":
            cv2.circle(source, (x + w // 2, y + h // 2), min(w, h) // 2, colour, -1)
        else:
            pts = np.array([
                [x + w // 2, y],
                [x, y + h],
                [x + w, y + h],
            ], np.int32)
            cv2.fillPoly(source, [pts], colour)

        # Store the first object as the template
        if i == 0:
            template_info = (x, y, w, h)

    # Add some noise
    noise = np.random.randint(0, 30, source.shape, dtype=np.uint8)
    source = cv2.add(source, noise)

    source_path = str(out / "source.png")
    cv2.imwrite(source_path, source)
    print(f"✅ Source image saved: {source_path} ({source_w}×{source_h})")

    if template_info:
        x, y, w, h = template_info
        template = source[y:y + h, x:x + w].copy()
        template_path = str(out / "template.png")
        cv2.imwrite(template_path, template)
        print(f"✅ Template image saved: {template_path} ({w}×{h})")
        return source_path, template_path

    return source_path, None


if __name__ == "__main__":
    create_test_images()
    print("\nNow run: python main.py match examples/source.png examples/template.png")
