"""Catalog validation hook — Protocol + no-op implementation.

Real catalog sources (KBIMS CSV, PPS master list) are deferred to a
separate spec. This file exists so the Predictor can be typed against
a stable interface and so adding validation later is a drop-in swap.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from api.bim.predict.schemas import PredictionCandidate


@runtime_checkable
class CatalogSource(Protocol):
    def validate(self, candidate: PredictionCandidate) -> PredictionCandidate:
        """Return the candidate (possibly augmented) or raise on hard-fail.

        NoOp implementations pass through. Real implementations may:
        - reject non-existent codes (raise)
        - enrich reasoning with catalog metadata
        - normalize code formatting
        """
        ...


class NoOpCatalog:
    """Pass-through catalog — used until real KBIMS/PPS sources land."""

    def validate(self, candidate: PredictionCandidate) -> PredictionCandidate:
        return candidate
