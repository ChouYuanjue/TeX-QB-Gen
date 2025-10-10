import os
from contextlib import contextmanager
from typing import Iterator

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

try:
    import pytesseract  # type: ignore[import]
    # allow user to override tesseract binary path via env var
    tcmd = os.getenv('TESSERACT_CMD')
    if tcmd:
        pytesseract.pytesseract.tesseract_cmd = tcmd
except Exception:
    pytesseract = None

try:
    from paddleocr import PaddleOCR  # type: ignore[import]
    paddle_available = True
except ImportError:
    paddle_available = False


@contextmanager
def _open_image(image_path: str) -> Iterator[Image.Image]:
    with Image.open(image_path) as img:
        yield img.copy()


def _preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    """Enhance scanned pages to improve downstream OCR accuracy."""
    grayscale = ImageOps.grayscale(image)
    autocontrasted = ImageOps.autocontrast(grayscale, cutoff=2)
    denoised = autocontrasted.filter(ImageFilter.MedianFilter(size=3))
    sharpened = denoised.filter(ImageFilter.UnsharpMask(radius=2, percent=140, threshold=3))
    enhanced = ImageEnhance.Contrast(sharpened).enhance(1.6)
    threshold = enhanced.point(lambda px: 255 if px > 180 else 0)
    return threshold


def image_to_text(image_path: str, ocr_engine: str = 'tesseract') -> str:
    """Extract text from an image file using the specified OCR engine.

    Supports 'tesseract' and 'paddle' engines. Defaults to 'tesseract'.

    Raises a RuntimeError with a helpful message if the OCR engine isn't available.
    """
    if ocr_engine == 'paddle':
        return _paddle_image_to_text(image_path)
    else:
        return _tesseract_image_to_text(image_path)


def _tesseract_image_to_text(image_path: str) -> str:
    """Use pytesseract to extract text from an image file.

    Raises a RuntimeError with a helpful message if Tesseract isn't available.
    """
    if pytesseract is None:
        raise RuntimeError('pytesseract or Tesseract binary not available. Please install Tesseract and ensure it is on PATH, or set TESSERACT_CMD env var to its full path.')
    with _open_image(image_path) as img:
        prepared = _preprocess_image_for_ocr(img)
    try:
        text = pytesseract.image_to_string(prepared, lang='chi_sim+eng')  # Add Chinese support
    except pytesseract.pytesseract.TesseractNotFoundError as exc:  # type: ignore[attr-defined]
        raise RuntimeError('Tesseract OCR binary not found. Install https://tesseract-ocr.github.io/tessdoc/Installation.html and ensure it is on PATH, or set TESSERACT_CMD env var to its full path.') from exc
    except FileNotFoundError as exc:
        raise RuntimeError('Tesseract OCR executable not found on system PATH. Install it and retry, or set TESSERACT_CMD env var.') from exc
    return text


def _paddle_image_to_text(image_path: str) -> str:
    """Use PaddleOCR to extract text from an image file.

    Suitable for Chinese text. Raises a RuntimeError if PaddleOCR isn't available.
    """
    if not paddle_available:
        raise RuntimeError('PaddleOCR not available. Please install paddlepaddle and paddleocr via pip.')
    
    # Load image with PIL and convert to numpy array
    with _open_image(image_path) as img:
        prepared = _preprocess_image_for_ocr(img).convert('RGB')
    import numpy as np
    img_array = np.array(prepared)
    
    # Initialize PaddleOCR with Chinese support
    ocr = PaddleOCR(use_angle_cls=True, lang='ch')
    result = ocr.ocr(img_array)
    
    if result is None or len(result) == 0 or result[0] is None:
        return ""
    
    # Extract text from results - result[0] is a list of [bbox, (text, confidence)]
    texts = []
    for line in result[0]:
        if len(line) >= 2 and isinstance(line[1], (list, tuple)) and len(line[1]) >= 1:
            text = line[1][0]
            if text.strip():
                texts.append(text)
    return '\n'.join(texts)
