"""End-to-end processing of one enquiry. Shared by the Streamlit app and run_tests.py."""
from __future__ import annotations

from dataclasses import dataclass, field

from . import rules
from .generate import Draft, generate
from .retrieval import RetrievalResult, Retriever

# Pre-check codes that stop drafting altogether (nothing safe to draft from).
BLOCKING = {"prompt_injection", "product_not_covered", "low_confidence"}


@dataclass
class Result:
    enquiry: str
    retrieval: RetrievalResult
    pre_hits: list[rules.RuleHit]
    draft: Draft
    post_hits: list[rules.RuleHit]
    decision: str
    reasons: list[str]
    pv_flag: bool
    pv_reason: str = ""
    hit_codes: list[str] = field(default_factory=list)
    quality_flag: bool = False
    quality_reason: str = ""


def process(enquiry: str, retriever: Retriever, api_key: str | None = None) -> Result:
    retrieval = retriever.search(enquiry)
    pre = rules.pre_checks(enquiry, retrieval, retriever)

    blocking = [h for h in pre if h.code in BLOCKING]
    draft = generate(enquiry, retrieval, retriever, api_key,
                     block_reason=blocking[0].reason if blocking else None)

    post = rules.post_checks(draft.text, draft.coverage, draft.citation_check)
    decision, reasons = rules.triage(pre + post)

    safety = rules.detect_safety(enquiry, pre)
    pv_reason = ""
    if safety.pv_flag:
        pv_reason = (" ".join(safety.pv_reasons) + " Report to pharmacovigilance within 24 hours "
                     "regardless of the triage outcome.")
    return Result(enquiry, retrieval, pre, draft, post, decision, reasons, safety.pv_flag, pv_reason,
                  hit_codes=sorted({h.code for h in pre + post}),
                  quality_flag=safety.quality_flag, quality_reason=" ".join(safety.quality_reasons))
