"""Pool building and mode evaluation — pure functions, no external deps.

``build_pool`` dedupes neighbors by the target code field, keeping the
highest similarity score per code. ``evaluate_mode`` applies the 3-condition
switch from the spec: WEAK if pool < n, pool == 1, or top1 < threshold.
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

    # top1 is the all-neighbor max; code_to_max drops empty-code neighbors (spec §5.2)
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
    cond_a = pool.unique_count < n
    cond_b = pool.unique_count == 1
    cond_c = pool.top1_score < sim_threshold
    if cond_a or cond_b or cond_c:
        return PredictionMode.WEAK
    return PredictionMode.STRONG
