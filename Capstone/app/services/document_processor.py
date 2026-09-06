import os
from pypdf import PdfReader
from docx import Document
from fastapi import UploadFile, HTTPException
from backend.app.config import settings

def validate_file(file: UploadFile):
    """
    Validates file extension and size.
    """
    # Extension validation
    filename = file.filename or ""
    ext = filename.split(".")[-1].lower() if "." in filename else ""
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: .{ext}. Allowed types: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )
        
    # File size validation (Requires reading some content or checking header)
    # We can read/seek to get length
    file.file.seek(0, os.SEEK_END)
    size = file.file.tell()
    file.file.seek(0)  # Reset pointer
    
    if size > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds maximum allowed size of {settings.MAX_FILE_SIZE // (1024 * 1024)}MB."
        )

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts text from PDF page-by-page.
    """
    try:
        reader = PdfReader(file_path)
        text_list = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_list.append(page_text)
        return "\n".join(text_list)
    except Exception as e:
        raise ValueError(f"Error parsing PDF: {str(e)}")

def extract_text_from_docx(file_path: str) -> str:
    """
    Extracts paragraphs from DOCX.
    """
    try:
        doc = Document(file_path)
        text_list = []
        for paragraph in doc.paragraphs:
            if paragraph.text:
                text_list.append(paragraph.text)
        return "\n".join(text_list)
    except Exception as e:
        raise ValueError(f"Error parsing DOCX: {str(e)}")

def extract_text_from_txt(file_path: str) -> str:
    """
    Reads plain text files. Attempts utf-8 and falls back to latin-1.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1") as f:
            return f.read()
    except Exception as e:
        raise ValueError(f"Error reading TXT: {str(e)}")

def extract_text(file_path: str) -> str:
    """
    Routes document based on extension and extracts text.
    """
    ext = file_path.split(".")[-1].lower() if "." in file_path else ""
    if ext == "pdf":
        return extract_text_from_pdf(file_path)
    elif ext == "docx":
        return extract_text_from_docx(file_path)
    elif ext in ("txt", "csv", "json"):  # txt/csv/json can be read as text
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file format: .{ext}")
