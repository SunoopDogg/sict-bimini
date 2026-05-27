import logging
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from api.bim.xlsx_parser import MissingColumnsError, parse_xlsx_to_raw
from api.routers.schemas import XLSXConversionResult

logger = logging.getLogger(__name__)

router = APIRouter(tags=["conversion"])

MAX_FILE_SIZE = 50 * 1024 * 1024
ALLOWED_EXTENSIONS = {".xlsx"}  # openpyxl does not support legacy .xls format


@router.post("/convert/xlsx-to-json", response_model=XLSXConversionResult)
async def convert_xlsx_to_json(file: UploadFile) -> XLSXConversionResult:
    filename = file.filename or "upload"
    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid extension. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",  # noqa: E501
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max {MAX_FILE_SIZE // (1024 ** 2)}MB",
        )

    start = time.time()
    with tempfile.NamedTemporaryFile(suffix=ext, delete=True) as tmp:
        tmp.write(content)
        tmp.flush()
        try:
            objects = parse_xlsx_to_raw(Path(tmp.name))
        except MissingColumnsError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        except Exception as e:
            logger.error("xlsx conversion error: %s", e)
            raise HTTPException(
                status_code=500, detail="Internal error during conversion"
            ) from None

    return XLSXConversionResult(
        objects=[obj.model_dump() for obj in objects],
        total_objects=len(objects),
        processing_time_seconds=round(time.time() - start, 3),
        source_filename=filename,
    )
