from pathlib import Path
import re
import uuid

import pymupdf
import pytesseract
from PIL import Image
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.document import Document

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

router = APIRouter(prefix="/documents", tags=["Documents"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
def extract_text_with_ocr(pdf):
    extracted_text = ""

    for page in pdf:
        pix = page.get_pixmap(dpi=200)

        image = Image.frombytes(
            "RGB",
            [pix.width, pix.height],
            pix.samples,
        )

        extracted_text += pytesseract.image_to_string(image)

    # Normalize OCR whitespace while preserving paragraph breaks.
    extracted_text = re.sub(r"(?<!\n)\n(?!\n)", " ", extracted_text)
    extracted_text = re.sub(r"[ \t]+", " ", extracted_text)

    return extracted_text.strip()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/")
async def create_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    # Validate file type
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    # Read file
    contents = await file.read()

    # Validate file size
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="File size exceeds the 10 MB limit.",
        )

    # Generate a safe unique filename
    safe_filename = f"{uuid.uuid4()}_{Path(file.filename).name}"
    file_path = UPLOAD_DIR / safe_filename

    # Save file
    file_path.write_bytes(contents)

    # Extract text
    extracted_text = ""

    try:
        pdf = pymupdf.open(file_path)
        for page in pdf:
            extracted_text += page.get_text()
        # If very little text was extracted, use OCR.
        if len(extracted_text.strip()) < 50:
            extracted_text = extract_text_with_ocr(pdf)

        pdf.close()

    except Exception as exc:
        file_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=400,
            detail=f"Could not process PDF: {str(exc)}",
        )

    # Store metadata + extracted text
    document = Document(
        filename=file.filename,
        content_type=file.content_type,
        storage_path=str(file_path),
        text_content=extracted_text or None,
    )

    db.add(document)
    db.commit()
    db.refresh(document)

    return document