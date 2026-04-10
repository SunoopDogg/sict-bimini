"""Prompt templates — loaded via importlib.resources at runtime.

Keeping this as a package (not loose files) guarantees the .txt files
ship in the wheel and are resolvable from ``api.bim.predict.prompts``.
"""
