"""End-to-end processing of one enquiry. Shared by the Streamlit app and run_tests.py.

Layering:
  1. Deterministic rules (regex/lexicon) -> the floor for triage, PV and quality flags.
  2. Optional LLM safety classifier (api / claude_cli backends) -> can only ADD flags and
     escalation reasons, tagged [llm]. It never removes or downgrades a rule result.
  3. Drafting (extractive or LLM) + citation/grounding post-checks.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from . import rules
from .generate import Draft, generate
from .llm import select_backend
from .retrieval import RetrievalResult, Retriever
from .safety_llm import LLMSafety, classify

# Pre-check codes that stop drafting altogether (nothing safe to draft from).
BLOCKING = {"prompt_injection", "product_not_covered", "low_confidence"}

# LLM classifier field -> rule code it can add.
LLM_TO_CODE = {"paediatric_or_elderly": "paediatric_elderly", "off_label": "off_label",
               "patient_specific": "patient_specific"}
# LLM special situation -> words that show the regex already raised the same thing.
SS_REGEX_MARKER = {"pregnancy_exposure": "pregnancy", "overdose": "overdose",
                   "medication_error": "medication error", "lack_of_efficacy": "lack of efficacy",
                   "off_label_use": "off-label", "paediatric_use": "paediatric"}
SS_LABEL = {"pregnancy_exposure": "exposure during pregnancy or breastfeeding", "overdose": "overdose",
            "medication_error": "possible medication error", "lack_of_efficacy": "reported lack of efficacy",
            "off_label_use": "reported off-label use", "paediatric_use": "reported paediatric use"}


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
    backend: str = "extractive"
    llm_hits: list[rules.RuleHit] = field(default_factory=list)
    llm_safety: LLMSafety | None = None
    # Three decisions, for honest layer attribution:
    decision_rules_pre: str = ""           # regex/lexicon pre-checks on the enquiry only
    decision_rules_plus_draft: str = ""    # pre-checks + post-checks on whatever draft was produced
    escalated_by: list[str] = field(default_factory=list)  # subset of rules_pre / draft_checks / llm_classifier
    pv_regex: bool = False
    quality_regex: bool = False
    layers: dict = field(default_factory=dict)   # flag -> "regex" | "llm" | "both"


def _layer(regex: bool, llm: bool) -> str:
    return "both" if regex and llm else "regex" if regex else "llm" if llm else ""


def process(enquiry: str, retriever: Retriever, api_key: str | None = None,
            backend: str | None = None) -> Result:
    backend = select_backend(backend, api_key)
    with ThreadPoolExecutor(max_workers=1) as pool:
        # The classifier sees ONLY the enquiry, so it can run while we retrieve and draft.
        safety_future = pool.submit(classify, enquiry, backend, api_key)

        retrieval = retriever.search(enquiry)
        pre = rules.pre_checks(enquiry, retrieval, retriever)
        blocking = [h for h in pre if h.code in BLOCKING]
        draft = generate(enquiry, retrieval, retriever, backend, api_key,
                         block_reason=blocking[0].reason if blocking else None)
        post = rules.post_checks(draft.text, draft.coverage, draft.citation_check)
        llm_safety: LLMSafety = safety_future.result()

    # Note: `post` checks the DRAFT, which may be LLM-written, so pre+post is not a pure regex floor.
    decision_rules_pre, _ = rules.triage(pre)
    decision_rules_plus_draft, _ = rules.triage(pre + post)
    safety = rules.detect_safety(enquiry, pre)
    pv_reasons = list(safety.pv_reasons)
    quality_reasons = list(safety.quality_reasons)
    regex_codes = {h.code for h in pre + post}
    layers: dict[str, str] = {}

    # ---- Merge: LLM can only ADD ----
    llm_hits: list[rules.RuleHit] = []
    f = llm_safety.flags if llm_safety.ok else {}
    ev = f.get("evidence", "")
    ev_txt = f" (evidence: “{ev}”)" if ev else ""

    for key, code in LLM_TO_CODE.items():
        llm_says = bool(f.get(key))
        layers[code] = _layer(code in regex_codes, llm_says)
        if llm_says and code not in regex_codes:
            llm_hits.append(rules.RuleHit(code, f"LLM classifier flagged {key.replace('_', ' ')}{ev_txt}.", "llm"))
    llm_serious = bool(f.get("adverse_event") and f.get("serious"))
    layers["serious_adverse_event"] = _layer("serious_adverse_event" in regex_codes, llm_serious)
    if llm_serious and "serious_adverse_event" not in regex_codes:
        llm_hits.append(rules.RuleHit("serious_adverse_event",
                                      f"LLM classifier flagged a possibly serious adverse event{ev_txt}.", "llm"))

    regex_ae = any(r.startswith("Possible adverse event") for r in safety.pv_reasons)
    layers["adverse_event"] = _layer(regex_ae, bool(f.get("adverse_event")))
    if f.get("adverse_event") and not regex_ae:
        pv_reasons.append(f"[llm] Possible adverse event{ev_txt}.")
    for ss in f.get("special_situations", []):
        regex_has = any(SS_REGEX_MARKER[ss] in r.lower() for r in safety.pv_reasons)
        layers[f"ss:{ss}"] = _layer(regex_has, True)
        if not regex_has:
            pv_reasons.append(f"[llm] Special situation: {SS_LABEL[ss]}{ev_txt}.")
    pv_llm = bool(f.get("adverse_event") or f.get("special_situations"))
    layers["pv"] = _layer(safety.pv_flag, pv_llm)

    layers["quality"] = _layer(safety.quality_flag, bool(f.get("quality_complaint")))
    if f.get("quality_complaint") and not safety.quality_flag:
        quality_reasons.append(f"[llm] Possible product quality complaint{ev_txt}. Forward to Quality Assurance.")

    decision, reasons = rules.triage(pre + post + llm_hits)
    pv_flag = bool(pv_reasons)
    pv_reason = (" ".join(pv_reasons) + " Report to pharmacovigilance within 24 hours "
                 "regardless of the triage outcome.") if pv_flag else ""
    return Result(enquiry, retrieval, pre, draft, post, decision, reasons, pv_flag, pv_reason,
                  hit_codes=sorted({h.code for h in pre + post + llm_hits}),
                  quality_flag=bool(quality_reasons), quality_reason=" ".join(quality_reasons),
                  backend=backend, llm_hits=llm_hits, llm_safety=llm_safety,
                  decision_rules_pre=decision_rules_pre, decision_rules_plus_draft=decision_rules_plus_draft,
                  escalated_by=[name for name, hits in (("rules_pre", pre), ("draft_checks", post),
                                                         ("llm_classifier", llm_hits)) if hits],
                  pv_regex=safety.pv_flag, quality_regex=safety.quality_flag,
                  layers={k: v for k, v in layers.items() if v})
