import pdfplumber
import docx2txt
import pytesseract
from pdf2image import convert_from_bytes

def extract_text_from_pdf(file_bytes):
    text = ""
    try:
        with pdfplumber.open(file_bytes) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
    except:
        # Fallback OCR
        images = convert_from_bytes(file_bytes.read())
        for img in images:
            text += pytesseract.image_to_string(img)
    return text.strip()

def extract_text(file):
    if file.name.endswith(".pdf"):
        return extract_text_from_pdf(file)
    elif file.name.endswith(".docx"):
        return docx2txt.process(file)
    else:
        return ""
