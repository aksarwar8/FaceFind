"""Draw local Unicode labels while preserving detector box coordinates."""

from functools import lru_cache
import os
from pathlib import Path
import unicodedata

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, features


@lru_cache(maxsize=12)
def label_fonts(size):
    configured = os.environ.get("FACEFIND_ANNOTATION_FONT")
    windows_fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    paths = (
        configured,
        windows_fonts / "Nirmala.ttc",
        windows_fonts / "Nirmala.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansBengali-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    fonts = []
    for path in filter(None, paths):
        try:
            fonts.append(ImageFont.truetype(str(path), size))
        except (OSError, ValueError):
            continue
    return (*fonts, ImageFont.load_default(size=size))


def supports_label(font, label):
    if not features.check_feature("raqm") and any(
        not char.isascii() and not unicodedata.name(char, "").startswith("LATIN ")
        for char in label
    ):
        return False
    missing = bytes(font.getmask("\U0010FFFF"))
    return all(
        bytes(font.getmask(char)) != missing
        for char in set(label)
        if not char.isspace() and unicodedata.category(char) != "Cf"
    )


def draw_label(draw, label, fallback, anchor, canvas_size, fonts, color):
    width, height = canvas_size
    padding = 3
    for text in (label, fallback):
        for font in fonts:
            if not supports_label(font, text):
                continue
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            box_width = right - left + 2 * padding
            box_height = bottom - top + 2 * padding
            if box_width > width or box_height > height:
                continue
            x = min(max(anchor[0], 0), width - box_width)
            y = anchor[1] - box_height - 2
            if y < 0:
                y = anchor[1] + 2
            y = min(max(y, 0), height - box_height)
            draw.rectangle((x, y, x + box_width - 1, y + box_height - 1), fill=(15, 23, 42), outline=color)
            draw.text((x + padding - left, y + padding - top), text, font=font, fill="white")
            return


def annotate_faces(image, results):
    annotated = image.copy()
    labels = []
    for number, result in enumerate(results, 1):
        x, y, width, height = map(int, result["bbox"])
        identity = result["identity"]
        status = result["status"]
        if status == "recognised":
            color, label = (40, 200, 100), identity
        elif status == "possible":
            color, label = (0, 165, 255), f"Possible: {identity}"
        elif status == "unknown":
            color, label = (60, 60, 230), "Unknown"
        else:
            color, label = (128, 128, 128), "Processing error"
        cv2.rectangle(annotated, (x, y), (x + width, y + height), color, 2)
        fallback = f"Face {result.get('face_number', number)}"
        clean_label = " ".join(str(label or fallback).split())
        # Oversized labels can use the face number without rasterizing unbounded text.
        labels.append((clean_label if len(clean_label) <= 160 else fallback, fallback, (x, y), color[::-1]))

    if not labels:
        return annotated
    canvas = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas)
    fonts = label_fonts(max(12, min(36, round(min(canvas.size) / 42))))
    for label, fallback, anchor, color in labels:
        draw_label(draw, label, fallback, anchor, canvas.size, fonts, color)
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
