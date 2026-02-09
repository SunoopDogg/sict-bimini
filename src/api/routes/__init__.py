from .prediction import router as prediction_router
from .search import router as search_router
from .conversion import router as conversion_router
from .bim_attributes import router as bim_attributes_router

__all__ = [
    "prediction_router",
    "search_router",
    "conversion_router",
    "bim_attributes_router",
]
