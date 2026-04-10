import pytest

from api.bim.predict.pool import build_pool, evaluate_mode
from api.bim.predict.schemas import CandidatePool, Neighbor, PredictionMode


def _n(score: float, kbims: str = "", pps: str = "") -> Neighbor:
    return Neighbor(
        stable_id=f"id-{score}-{kbims}-{pps}",
        score=score,
        kbims_code=kbims,
        pps_code=pps,
        ifc_type="IfcColumn",
        category="건축",
    )


class TestBuildPool:
    def test_empty_neighbors(self):
        pool = build_pool([], "kbims_code")
        assert pool.unique_count == 0
        assert pool.top1_score == 0.0
        assert pool.code_to_max_score == {}

    def test_dedupes_by_code_keeping_max_score(self):
        neighbors = [
            _n(0.6, kbims="KM001"),
            _n(0.9, kbims="KM001"),
            _n(0.7, kbims="KM002"),
        ]
        pool = build_pool(neighbors, "kbims_code")
        assert pool.code_to_max_score == {"KM001": 0.9, "KM002": 0.7}
        assert pool.top1_score == 0.9
        assert pool.unique_count == 2

    def test_skips_empty_code_field(self):
        neighbors = [
            _n(0.9, kbims="KM001"),
            _n(0.8, kbims=""),
        ]
        pool = build_pool(neighbors, "kbims_code")
        assert pool.unique_count == 1
        assert pool.top1_score == 0.9

    def test_uses_specified_code_field(self):
        neighbors = [
            _n(0.8, kbims="KM001", pps="A-1"),
            _n(0.7, kbims="", pps="A-2"),
        ]
        pool = build_pool(neighbors, "pps_code")
        assert set(pool.code_to_max_score) == {"A-1", "A-2"}


class TestEvaluateMode:
    @pytest.mark.parametrize(
        "unique_count, top1, n, sim_threshold, expected",
        [
            (5, 0.8, 5, 0.55, PredictionMode.STRONG),
            (3, 0.8, 5, 0.55, PredictionMode.WEAK),
            (1, 0.9, 1, 0.55, PredictionMode.WEAK),
            (5, 0.4, 5, 0.55, PredictionMode.WEAK),
            (5, 0.55, 5, 0.55, PredictionMode.STRONG),
            (0, 0.0, 5, 0.55, PredictionMode.WEAK),
        ],
    )
    def test_table(self, unique_count, top1, n, sim_threshold, expected):
        pool = CandidatePool(
            code_to_max_score={f"C{i}": 0.5 for i in range(unique_count)},
            top1_score=top1,
            unique_count=unique_count,
        )
        assert evaluate_mode(pool, n, sim_threshold=sim_threshold) == expected
