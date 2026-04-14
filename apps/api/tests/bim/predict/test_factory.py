from unittest.mock import MagicMock

from api.bim.predict.catalog import NoOpCatalog
from api.bim.predict.factory import (
    build_kbims_predictor,
    build_pps_predictor,
)
from api.bim.predict.predictor import Predictor
from api.core.config import BIMSettings


def test_build_kbims_predictor_uses_kbims_config():
    settings = BIMSettings()  # defaults
    embed = MagicMock()
    qdrant = MagicMock()
    vllm = MagicMock()

    predictor = build_kbims_predictor(
        settings=settings,
        embed_client=embed,
        qdrant_client=qdrant,
        vllm_client=vllm,
    )

    assert isinstance(predictor, Predictor)
    cfg = predictor._config  # type: ignore[attr-defined]
    assert cfg.target == "kbims_code"
    assert cfg.code_format_regex == settings.kbims_code_regex
    assert cfg.k_min == settings.retrieval_k_min
    assert cfg.sim_threshold == settings.mode_sim_threshold
    assert isinstance(cfg.catalog, NoOpCatalog)


def test_build_pps_predictor_uses_pps_config():
    settings = BIMSettings()
    predictor = build_pps_predictor(
        settings=settings,
        embed_client=MagicMock(),
        qdrant_client=MagicMock(),
        vllm_client=MagicMock(),
    )

    cfg = predictor._config  # type: ignore[attr-defined]
    assert cfg.target == "pps_code"
    assert cfg.code_format_regex == settings.pps_code_regex


def test_two_predictors_are_independent_instances():
    settings = BIMSettings()
    shared = dict(
        embed_client=MagicMock(),
        qdrant_client=MagicMock(),
        vllm_client=MagicMock(),
    )
    a = build_kbims_predictor(settings=settings, **shared)
    b = build_pps_predictor(settings=settings, **shared)

    assert a is not b
    assert a._config.target != b._config.target
