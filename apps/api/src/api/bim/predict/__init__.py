"""BIM code prediction module (RAG + vLLM guided_json).

Two predictors (kbims_code, pps_code) share one ``Predictor`` class and
differ only by injected ``PredictorConfig``. HTTP/FastAPI wiring lives
outside this module.
"""
