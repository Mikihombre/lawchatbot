# src/file_ocr.py

import fitz  # PyMuPDF
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from typing import List


# Jeśli Tesseract nie jest w PATH, ustaw ścieżkę:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def extract_text_from_pdf_native(uploaded_file) -> str:
    """
    Ekstrakcja tekstu z PDF z warstwą tekstową (typowe PDF-y z Worda, systemów sądowych).
    uploaded_file: obiekt UploadedFile ze Streamlit.
    """
    try:
        pdf_bytes = uploaded_file.read()
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            texts = []
            for page in doc:
                texts.append(page.get_text("text"))
        text = "\n".join(texts).strip()
        return text
    except Exception as e:
        return f"[Błąd odczytu PDF {uploaded_file.name}: {e}]"


def ocr_scanned_pdf(uploaded_file) -> str:
    """
    OCR dla zeskanowanych PDF-ów:
    - renderuje każdą stronę do obrazu (200 DPI),
    - puszcza tekst przez Tesseract (polski).
    Wolniejsze, ale działa na skanach bez warstwy tekstowej.
    """
    try:
        pdf_bytes = uploaded_file.read()
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            texts = []
            for page in doc:
                pix = page.get_pixmap(dpi=200)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                page_text = pytesseract.image_to_string(img, lang="pol")
                texts.append(page_text)
        return "\n".join(texts).strip()
    except Exception as e:
        return f"[Błąd OCR zeskanowanego PDF {uploaded_file.name}: {e}]"


def ocr_image_file(uploaded_file) -> str:
    """
    OCR dla pojedynczego obrazu (PNG/JPG/JPEG) z użyciem Tesseracta (język polski).
    """
    try:
        image = Image.open(uploaded_file).convert("RGB")
        text = pytesseract.image_to_string(image, lang="pol")
        return text.strip()
    except Exception as e:
        return f"[Błąd OCR obrazu {uploaded_file.name}: {e}]"


def extract_text_from_files(files: List) -> str:
    """
    Łączy tekst ze wszystkich załączonych plików:
    - PDF: najpierw próba odczytu tekstu natywnego (PyMuPDF),
           jeśli pusto lub błąd, fallback na OCR zeskanowanego PDF.
    - Obrazy: OCR Tesseractem.
    Zwraca jeden duży string, który możesz potraktować jako 'treść wniosku'.
    """
    if not files:
        return ""

    all_texts = []

    for f in files:
        if f.type == "application/pdf":
            # 1) spróbuj wyciągnąć tekst natywny
            text = extract_text_from_pdf_native(f)

            # 2) opcjonalny fallback na OCR skanu
            if (not text or text.startswith("[Błąd")):
                f.seek(0)
                text = ocr_scanned_pdf(f)

            all_texts.append(f"--- PDF: {f.name} ---\n{text}")
            f.seek(0)

        elif f.type.startswith("image/"):
            text = ocr_image_file(f)
            all_texts.append(f"--- OBRAZ: {f.name} ---\n{text}")

        else:
            all_texts.append(f"[Nieobsługiwany typ pliku: {f.name} ({f.type})]")

    return "\n\n".join(all_texts)
