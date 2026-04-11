"""Prompt template loading + placeholder injection.

Templates live in ``api.bim.predict.prompts.{target}_{mode}.txt``
(4 files total). Loaded via ``importlib.resources.files`` so wheel
installs, editable installs, and dev mode all work.

Placeholder contract (all required):
    {attribute_block}     BIMAttribute rendered as embed_text()
    {candidates_block}    "- {code} (similarity {score:.2f})" lines
    {n}                   requested candidate count
"""
from __future__ import annotations

from importlib.resources import files

from api.bim.predict.schemas import CandidatePool, PredictionMode, TargetCode
from api.bim.schemas import BIMAttribute

_REQUIRED_PLACEHOLDERS = ("{attribute_block}", "{candidates_block}", "{n}")


class PromptBuilder:
    def build(
        self,
        *,
        target: TargetCode,
        mode: PredictionMode,
        attribute: BIMAttribute,
        pool: CandidatePool,
        n: int,
    ) -> str:
        template = self._load_template(target, mode)
        self._assert_placeholders(template)
        return template.format(
            attribute_block=self._render_attribute(attribute),
            candidates_block=self._render_candidates(pool),
            n=n,
        )

    def _load_template(
        self,
        target: str,
        mode: PredictionMode,
    ) -> str:
        target_prefix = target.removesuffix("_code")
        filename = f"{target_prefix}_{mode.value}.txt"
        return (
            files("api.bim.predict.prompts")
            .joinpath(filename)
            .read_text(encoding="utf-8")
        )

    @staticmethod
    def _assert_placeholders(template: str) -> None:
        for ph in _REQUIRED_PLACEHOLDERS:
            if ph not in template:
                raise KeyError(
                    f"Prompt template missing required placeholder {ph}"
                )

    @staticmethod
    def _render_attribute(attr: BIMAttribute) -> str:
        existing = []
        if attr.kbims_code:
            existing.append(f"(existing kbims_code={attr.kbims_code})")
        if attr.pps_code:
            existing.append(f"(existing pps_code={attr.pps_code})")
        suffix = " " + " ".join(existing) if existing else ""
        return attr.embed_text() + suffix

    @staticmethod
    def _render_candidates(pool: CandidatePool) -> str:
        if not pool.code_to_max_score:
            return "(이웃 후보 없음)"
        items = sorted(
            pool.code_to_max_score.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )
        return "\n".join(
            f"- {code} (similarity {score:.2f})" for code, score in items
        )
