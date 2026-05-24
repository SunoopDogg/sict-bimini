"""Pool building and mode evaluation — pure functions, no external deps.

``build_pool`` dedupes neighbors by the target code field, keeping the
highest similarity score per code. ``evaluate_mode`` is WEAK only when the
retrieval signal is genuinely low (``top1_score < sim_threshold``) or when
nothing was retrieved. Pool *diversity* (``unique_count < n``) is not a
confidence signal — if 14/15 neighbors agree on the same code with top1
≈ 0.96, that's the strongest possible endorsement, and forcing WEAK there
just pushes the LLM into free-form generation where PPS-style open regexes
can truncate at ``max_tokens``.
"""
from __future__ import annotations

from api.bim.predict.schemas import CandidatePool, Neighbor, PredictionMode, TargetCode


def build_pool(neighbors: list[Neighbor], code_field: TargetCode) -> CandidatePool:
    code_to_max: dict[str, float] = {}
    for nb in neighbors:
        code = getattr(nb, code_field)
        if not code:
            continue
        if code not in code_to_max or nb.score > code_to_max[code]:
            code_to_max[code] = nb.score

    # top1 is the all-neighbor max; code_to_max drops empty-code neighbors.
    top1 = max((nb.score for nb in neighbors), default=0.0)
    return CandidatePool(
        code_to_max_score=code_to_max,
        top1_score=top1,
        unique_count=len(code_to_max),
    )


def evaluate_mode(
    pool: CandidatePool,
    n: int,
    *,
    sim_threshold: float,
) -> PredictionMode:
    if pool.unique_count == 0 or pool.top1_score < sim_threshold:
        return PredictionMode.WEAK
    return PredictionMode.STRONG
