import cv2
import numpy as np
import pytest
from PIL import ImageFont

from api import annotations


def render(name, *, number=1, status="recognised", bbox=(105, 0, 14, 20)):
    image = np.zeros((72, 120, 3), np.uint8)
    return annotations.annotate_faces(image, [
        {"bbox": bbox, "identity": name, "status": status, "face_number": number}
    ])


def test_ascii_and_unknown_labels_are_distinct_and_jpeg_encodes():
    alice = render("Alice")
    assert not np.array_equal(alice, render("Bob"))
    assert not np.array_equal(alice, render(None, status="unknown"))
    assert np.array_equal(render(None, status="unknown"), render("Unknown", status="unknown"))
    assert alice[20, 119].tolist() == [40, 200, 100]
    ok, jpeg = cv2.imencode(".jpg", alice)
    assert ok and cv2.imdecode(jpeg, cv2.IMREAD_COLOR).shape == alice.shape


@pytest.mark.parametrize("names", [
    ("Jos\u00e9", "Jo\u00e3o"),
    ("\u09b0\u09be\u09b9\u09c1\u09b2", "\u09b8\u09c1\u09ae\u09a8"),
    ("\u0930\u0935\u093f", "\u0905\u092e\u093f\u0924"),
])
def test_available_fonts_render_distinct_unicode_labels(names):
    if not all(any(annotations.supports_label(font, name) for font in annotations.label_fonts(12)) for name in names):
        pytest.skip("No locally available shaping font covers the tested names")
    assert not np.array_equal(render(names[0]), render(names[1]))


def test_missing_glyphs_and_long_names_use_stable_face_number(monkeypatch):
    monkeypatch.setattr(annotations, "label_fonts", lambda size: (ImageFont.load_default(size=size),))
    assert np.array_equal(render("\u738b", number=7), render("Face 7", number=7))
    assert not np.array_equal(render("\u738b", number=7), render("\u674e", number=8))
    assert np.array_equal(render("A" * 1000), render("Face 1"))
