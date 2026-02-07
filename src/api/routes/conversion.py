import logging
import time

from fastapi import APIRouter, HTTPException, UploadFile

from src.api.schemas import APIResponse, XLSXConversionResult
from src.converters import convert_bim_xlsx_from_bytes

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["conversion"])

# Maximum file size: 50MB
MAX_FILE_SIZE = 50 * 1024 * 1024
ALLOWED_EXTENSIONS = {".xlsx", ".xls"}


@router.post("/convert/xlsx-to-json", response_model=APIResponse[XLSXConversionResult])
async def convert_xlsx_to_json(
    file: UploadFile,
) -> APIResponse[XLSXConversionResult]:
    """Convert an uploaded Excel file containing BIM property tables to JSON."""
    # Validate file extension
    filename = file.filename or "uploaded_file"
    extension = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file extension. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Read file content
    start_time = time.time()
    try:
        file_content = await file.read()
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Failed to read uploaded file")

    # Validate file is not empty
    if len(file_content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # Validate file size
    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024 * 1024)}MB",
        )

    # Convert the file
    try:
        objects = convert_bim_xlsx_from_bytes(file_content, filename)
    except ValueError as e:
        logger.error(f"XLSX conversion validation error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"XLSX conversion error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during conversion")

    processing_time = time.time() - start_time

    return APIResponse(
        success=True,
        data=XLSXConversionResult(
            objects=objects,
            total_objects=len(objects),
            processing_time_seconds=round(processing_time, 3),
            source_filename=filename,
        ),
    )
