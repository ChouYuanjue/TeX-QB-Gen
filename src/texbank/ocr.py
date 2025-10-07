import os
import io
from PIL import Image

try:
    import pytesseract
    # allow user to override tesseract binary path via env var
    tcmd = os.getenv('TESSERACT_CMD')
    if tcmd:
        pytesseract.pytesseract.tesseract_cmd = tcmd
except Exception:
    pytesseract = None


def image_to_text(image_path: str) -> str:
    """Use pytesseract to extract text from an image file.

    Raises a RuntimeError with a helpful message if Tesseract isn't available.
    """
    if pytesseract is None:
        raise RuntimeError('pytesseract or Tesseract binary not available. Please install Tesseract and ensure it is on PATH, or set TESSERACT_CMD env var to its full path.')
    img = Image.open(image_path)
    try:
        text = pytesseract.image_to_string(img)
    except pytesseract.pytesseract.TesseractNotFoundError as exc:  # type: ignore[attr-defined]
        raise RuntimeError('Tesseract OCR binary not found. Install https://tesseract-ocr.github.io/tessdoc/Installation.html and ensure it is on PATH, or set TESSERACT_CMD env var to its full path.') from exc
    except FileNotFoundError as exc:
        raise RuntimeError('Tesseract OCR executable not found on system PATH. Install it and retry, or set TESSERACT_CMD env var.') from exc
    return text
