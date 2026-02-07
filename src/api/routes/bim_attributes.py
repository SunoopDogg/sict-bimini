import csv
import logging
import math
import os
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from src.api import dependencies as deps
from src.api.schemas import (
    APIResponse,
    BIMAttributeCreateRequest,
    BIMAttributeCreateResponse,
    BIMAttributeItem,
    BIMAttributeListResponse,
)
from src.utils import BIM_ATTRIBUTE_FIELDS, CSV_PATH, load_bim_attributes_from_csv

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["bim-attributes"])


@router.get("/bim-attributes", response_model=APIResponse[BIMAttributeListResponse])
async def list_bim_attributes(
    page: int = Query(default=1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(default=20, ge=1, le=100, description="Number of items per page"),
) -> APIResponse[BIMAttributeListResponse]:
    """List BIM attributes from the dataset with pagination."""
    if deps.bim_attributes_cache is None:
        raise HTTPException(status_code=503, detail="BIM attributes not loaded")

    total = len(deps.bim_attributes_cache)
    total_pages = math.ceil(total / page_size) if total > 0 else 0

    offset = (page - 1) * page_size
    items = [
        BIMAttributeItem(**attr.to_dict())
        for attr in deps.bim_attributes_cache[offset : offset + page_size]
    ]

    mtime = os.path.getmtime(CSV_PATH)
    last_modified = datetime.fromtimestamp(mtime, tz=timezone.utc)

    return APIResponse(
        success=True,
        data=BIMAttributeListResponse(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            last_modified=last_modified,
        ),
    )


@router.post("/bim-attributes", response_model=APIResponse[BIMAttributeCreateResponse])
async def append_bim_attributes(
    request: BIMAttributeCreateRequest,
) -> APIResponse[BIMAttributeCreateResponse]:
    """Append new BIM attribute rows to the dataset."""
    rows = [item.model_dump() for item in request.items]

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(BIM_ATTRIBUTE_FIELDS))
        writer.writerows(rows)

    deps.bim_attributes_cache = load_bim_attributes_from_csv(CSV_PATH)

    return APIResponse(
        success=True,
        data=BIMAttributeCreateResponse(
            added=len(rows),
            total=len(deps.bim_attributes_cache),
        ),
    )
