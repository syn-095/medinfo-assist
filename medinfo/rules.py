"""Deterministic escalation rules and adverse-event (PV) detection.

Rules run twice:
  * pre_checks  - on the enquiry and retrieval result, BEFORE any LLM call
  * post_checks - on the draft and its citation check, AFTER generation
Any hit -> ESCALATE TO MEDICAL EXPERT. The LLM can never lower a decision
made here; it can only add reasons to escalate.

These are transparent keyword/regex rules on purpose: they are auditable and
predictable, and they err towards escalation. Their recall limits are logged
in NOTES.md.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .retrieval import RetrievalResult, Retriever

DRAFT_OK = "DRAFT OK FOR MI OFFICER REVIEW"
ESCALATE = "ESCALATE TO MEDICAL EXPERT"


@dataclass
class RuleHit:
    code: str
    reason: str
    stage: str = "pre"   # "pre" or "post"


# ---- Product label facts used by the rules (would come from a config/PIM in production) ----
PRODUCT_RULES = {
    "Veloxitab": {
        "indication_terms": ["hypertension", "high blood pressure", "blood pressure"],
        "elderly_max_covered_age": 79,   # label: no recommendation for >= 80
    },
    "Corvanex": {
        "indication_terms": ["type 2 diabetes", "diabetes", "hba1c", "blood sugar", "glucose", "glycaemic"],
        "elderly_max_covered_age": None,  # label: no adjustment in elderly, so covered
    },
}

# Conditions that might be asked about; any not in the product's indication terms -> off-label.
CONDITIONS = [
    "type 1 diabetes", "type 2 diabetes", "diabetes", "hypertension", "high blood pressure", "migraine",
    "weight loss", "weight management", "lose weight", "obesity", "anxiety", "depression", "heart failure",
    "pcos", "polycystic", "acne", "insomnia", "sleep", "erectile", "hair loss", "prediabetes", "pre-diabetes",
    "kidney stones", "gout", "asthma", "cancer", "covid", "dementia", "alzheimer",
]

# Drug lexicon for interaction questions. A drug named in the enquiry that does not
# appear anywhere in the product's approved documents -> interaction not covered.
DRUGS = [
    "lithium", "levothyroxine", "warfarin", "apixaban", "rivaroxaban", "clopidogrel", "aspirin", "ibuprofen",
    "naproxen", "diclofenac", "paracetamol", "acetaminophen", "metformin", "insulin", "gliclazide",
    "sulfonylurea", "simvastatin", "atorvastatin", "rosuvastatin", "amlodipine", "ramipril", "lisinopril",
    "losartan", "bisoprolol", "atenolol", "digoxin", "furosemide", "spironolactone", "bendroflumethiazide",
    "omeprazole", "lansoprazole", "sertraline", "fluoxetine", "citalopram", "amitriptyline", "tramadol",
    "codeine", "morphine", "prednisolone", "methotrexate", "ciclosporin", "tacrolimus", "ketoconazole",
    "itraconazole", "fluconazole", "clarithromycin", "erythromycin", "amoxicillin", "ciprofloxacin",
    "rifampicin", "carbamazepine", "phenytoin", "sodium valproate", "allopurinol", "sildenafil",
    "st john's wort", "oral contraceptive", "contraceptive pill", "aliskiren", "grapefruit", "alcohol",
]

PATTERNS = {
    "off_label_explicit": r"\boff[- ]?label\b|\bunlicen[cs]ed\b|\boutside (the |its )?licen[cs]e\b|\bnot (licen[cs]ed|indicated)\b",
    "pregnancy": r"\bpregnan\w*|\bbreast[- ]?feed\w*|\bbreast ?milk\b|\blactat\w*|\bnursing (mother|baby)\b|\btrying (to|for a) (conceive|baby)\b|\bconception\b|\bexpecting (a baby|twins)\b|\bmother[- ]to[- ]be\b",
    "paediatric": r"\bchild(ren)?\b|\bpa?ediatric\w*|\binfant\w*|(?<!expecting a )\bbab(y|ies)\b|\btoddler\w*|\badolescen\w*|\bteen(ager)?s?\b|\bunder 18\b|\bkids?\b|\bneonat\w*",
    "age": r"\b(\d{1,3})[- ]?(?:years?|yrs?)[- ]?old\b|\baged (\d{1,3})\b|\b(\d{1,3}) ?y/?o\b",
    "elderly_words": r"\bvery elderly\b|\bfrail\b|\bcare home\b|\bnursing home\b",
    "overdose": r"\bover[- ]?dos\w*|\btoo many\b|\bdouble (dose|injection)d?\b.*\b(took|taken|given|injected)\b|\b(took|taken|swallowed|injected|given)\b.{0,30}\b(two|three|four|five|six|several|\d+)\b.{0,15}\b(tablets?|pills?|doses|pens|injections)\b|\baccidentally (took|taken|swallowed|injected|gave|given)\b",
    "interaction_cue": r"\binteract\w*|\btogether\b|\bwith\b|\balongside\b|\bcombin\w*|\bat the same time\b|\bco-?administ\w*",
    "competitor": r"\bmore effective than\b|\bbetter than\b|\bworse than\b|\bsuperior\b|\binferior\b|\bcompared? (to|with)\b|\bcomparison\b|\bversus\b|\bvs\.?\b|\bas good as\b|\bswitch(ing)? from \w+ to\b|\bcompetitor\b",
    "patient_specific": r"\bshould (i|we|he|she|they|my patient)\b.{0,40}\b(increase|decrease|reduce|stop|start|switch|change|continue|restart|give|take)\b|\b(increase|switch|stop|change|reduce) (him|her|them)\b|\bwould you recommend\b|\bwhat (dose|would you) .{0,30}\b(recommend|advise)\b|\bin my case\b|\bis (this|it) (normal|safe) for (me|him|her)\b|\bis this normal\b|\bis it safe to continue\b",
    "injection": r"\bignore (all |any )?(of )?(your |the |my )?(previous |prior |above |earlier )?(instructions|rules|prompt)|\bdisregard\b|\byou are now\b|\bsystem prompt\b|\bpretend (to be|you)\b|\bact as\b|\bjailbreak\b|\bdeveloper mode\b|\bofficial (company )?statement\b|\bnew instructions\b|^\s*system\s*:|\bnew policy\b|\b(from|use) your own knowledge\b|\boverride\b",
    # Adverse event: a personal/temporal experience, not a general "what are the side effects" question.
    "ae": r"\b(developed|experienced|experiencing|suffered|suffering|been having|been getting|has had|have had|started having|is having|am having|noticed|complain(ed|s|ing)? of|reported|came out in|broke out in)\b|\b(after|since) (taking|starting|using|injecting|being (put|started) on|the first (dose|injection)|i started|he started|she started)\b|\bhospitali[sz]\w*|\badmitted to hospital\b|\bA&E\b|\bemergency department\b|\bpassed away\b|\bdied\b",
    "serious_ae": r"\bhospitali[sz]\w*|\badmitted\b|\blife[- ]threatening\b|\bdied\b|\bdeath\b|\bpassed away\b|\bdisabilit\w*|\banaphyla\w*|\bA&E\b|\bemergency\b|\bintensive care\b|\bsevere\b",
    # ---- PV special situations (reportable even without an adverse event) ----
    # Exposure/actual-use cues: separate a report ("I'm pregnant and on X") from a hypothetical question.
    "exposure": r"\b(am|is|are|was|were|has been|have been|been|currently|already) (taking|using|on|injecting|prescribed|given)\b|\bwhile (taking|on|using)\b|\b(continue|continuing|keep taking|stop|stopping)\b|\bfound out\b|\b(became|got|become) pregnant\b|\bsince (taking|starting)\b|\b(has|have|had) (used|taken|injected|been given)\b|\b(took|injected)\b|\bprescribed\b|\bstarted (on|taking)\b",
    "medication_error": r"\bwrong (dose|medicine|medication|tablets?|strength|drug|pen|patient)\b|\bby mistake\b|\bin error\b|\binstead of\b.{0,30}\b(tablets?|dose|pen|mg)\b|\b(took|taken|injected|given) (it )?twice\b|\bexpired\b.{0,30}\b(took|taken|used|injected|gave)\b|\b(took|taken|used|injected|gave)\b.{0,30}\bexpired\b|\baccidentally\b|\bdispens\w+ error\b",
    "lack_of_efficacy": r"\b(not|isn'?t|aren'?t|doesn'?t|don'?t|didn'?t|hasn'?t|stopped|no longer) (seem to |seems to )?(work|working|worked|help|helping|helped|lowering|controlling)\b|\bno (effect|improvement|difference)\b|\bineffective\b|\black of (efficacy|effect)\b|\bstill (high|raised|elevated|uncontrolled|not controlled)\b",
    # ---- Product quality complaint (routes to QA, separate from PV) ----
    "quality": r"\bcloudy\b|\bbent (needle|pen)\b|\bbroken\b|\bdiscolou?r\w*|\btamper\w*|\bdamaged\b|\bcrack(ed|s)?\b|\bleak\w*|\bparticles?\b|\bforeign (body|object|matter)\b|\bwrong colou?r\b|\bcrumbl\w*|\bfaulty\b|\bjammed\b|\bmissing (tablets?|pills?)\b|\b(strange|odd|funny) (smell|taste)\b|\bseal\b.{0,20}\b(broken|open)|\bmislabel\w*|\bwon'?t (click|inject|dispense)\b|\bcounterfeit\b",
    # Claims a draft must never make (checked post-generation).
    "prohibited_claims": r"\bcures?\b|\bguarantee\w*|\bcompletely safe\b|\b100% safe\b|\bno side effects\b|\bbetter than\b|\bsuperior to\b|\bmiracle\b",
}


def _has(pattern_key: str, text: str) -> bool:
    return re.search(PATTERNS[pattern_key], text, re.I) is not None


def _ages(text: str) -> list[int]:
    out = []
    for m in re.finditer(PATTERNS["age"], text, re.I):
        for g in m.groups():
            if g:
                out.append(int(g))
    return out


def find_conditions(text: str) -> list[str]:
    """All CONDITIONS mentioned in text. Longest terms are matched first and their
    span is masked, so "type 1 diabetes" or "pre-diabetes" is not also read as "diabetes".
    Plurals are accepted ("migraines")."""
    text = text.lower()
    found = []
    for cond in sorted(CONDITIONS, key=len, reverse=True):
        pat = rf"(?<![\w-]){re.escape(cond)}s?(?![\w-])"
        if re.search(pat, text):
            found.append(cond)
            text = re.sub(pat, lambda m: "#" * len(m.group(0)), text)
    return found


def detect_adverse_event(enquiry: str) -> tuple[bool, bool, str]:
    """Return (is_ae, is_serious, matched_text)."""
    m = re.search(PATTERNS["ae"], enquiry, re.I)
    if not m:
        return False, False, ""
    serious = _has("serious_ae", enquiry)
    return True, serious, m.group(0)


@dataclass
class SafetyFlags:
    pv_flag: bool
    pv_reasons: list[str]
    quality_flag: bool
    quality_reasons: list[str]


def detect_safety(enquiry: str, pre_hits: list[RuleHit]) -> SafetyFlags:
    """PV flag = adverse event OR special situation. Quality flag = product quality complaint.
    Both are independent of triage: they are raised whatever the drafting outcome."""
    pv: list[str] = []
    is_ae, serious, matched = detect_adverse_event(enquiry)
    if is_ae:
        pv.append(f"Possible adverse event (matched '{matched}'{', possibly serious' if serious else ''}).")
    exposure = _has("exposure", enquiry)
    codes = {h.code for h in pre_hits}
    if "pregnancy" in codes and exposure:
        pv.append("Special situation: exposure during pregnancy or breastfeeding.")
    if _has("overdose", enquiry):
        pv.append("Special situation: overdose.")
    if _has("medication_error", enquiry):
        pv.append("Special situation: possible medication error.")
    if _has("lack_of_efficacy", enquiry):
        pv.append("Special situation: reported lack of efficacy.")
    if "off_label" in codes and exposure:
        pv.append("Special situation: reported off-label use.")
    q = []
    m = re.search(PATTERNS["quality"], enquiry, re.I)
    if m:
        q.append(f"Possible product quality complaint (matched '{m.group(0)}'). Forward to Quality Assurance; "
                 "ask the enquirer to keep the product and packaging (batch number).")
    return SafetyFlags(bool(pv), pv, bool(q), q)


def pre_checks(enquiry: str, retrieval: RetrievalResult, retriever: Retriever) -> list[RuleHit]:
    text = enquiry.lower()
    hits: list[RuleHit] = []
    add = lambda code, reason: hits.append(RuleHit(code, reason, "pre"))

    if _has("injection", enquiry):
        add("prompt_injection", "Enquiry contains instruction-like text aimed at the system (possible prompt injection). Not drafted automatically.")

    if not retrieval.products:
        add("product_not_covered", "No product from the approved document set is named in the enquiry, so there is no approved source to answer from.")

    # Off-label: explicit wording, or a condition that is not the product's indication.
    if _has("off_label_explicit", enquiry):
        add("off_label", "Enquiry refers to off-label / unlicensed use.")
    else:
        found = find_conditions(text)
        for product in retrieval.products:
            ind = set(PRODUCT_RULES.get(product, {}).get("indication_terms", []))
            # Every condition mentioned must be an approved indication (exact term match).
            # Errs to the safe side: a comorbidity mentioned in passing also escalates.
            off = [c for c in found if c not in ind]
            if off:
                add("off_label", f"Possible off-label use: {', '.join(repr(c) for c in off)} "
                                 f"not an approved indication for {product}.")
        if re.search(r"\b(don'?t|do not|doesn'?t|does not) have diabetes\b|\bnot diabetic\b|\bnon-?diabetic\b", text):
            if not any(h.code == "off_label" for h in hits):
                add("off_label", "Use in a non-diabetic person is outside the approved indication.")

    if _has("pregnancy", enquiry):
        add("pregnancy", "Pregnancy / breastfeeding enquiry: approved data are insufficient; requires medical expert review.")

    ages = _ages(enquiry)
    if _has("paediatric", enquiry) or any(a < 18 for a in ages):
        add("paediatric_elderly", "Paediatric use: safety and efficacy under 18 years not established in the label.")
    for product in retrieval.products:
        max_age = PRODUCT_RULES.get(product, {}).get("elderly_max_covered_age")
        if max_age is not None and (any(a > max_age for a in ages) or _has("elderly_words", enquiry)):
            add("paediatric_elderly", f"Elderly dosing outside the label: {product} label gives no recommendation above {max_age} years.")
            break

    if _has("overdose", enquiry):
        add("overdose", "Possible overdose: requires urgent medical expert review (advise emergency care if symptomatic).")

    # Interactions: any lexicon drug named in the enquiry but absent from all approved docs for the product.
    if retrieval.products:
        named = [d for d in DRUGS if re.search(rf"\b{re.escape(d)}\b", text)]
        for product in retrieval.products:
            ptext = retriever.product_text(product)
            missing = [d for d in named if d not in ptext]
            if missing:
                add("interaction_not_covered", f"Interaction with {', '.join(missing)} is not covered in the approved {product} documents.")

    if _has("competitor", enquiry):
        add("competitor_comparison", "Comparative claim versus another product requested; not supported by approved material.")

    if _has("patient_specific", enquiry):
        add("patient_specific", "Requests patient-specific treatment advice; MI can share label information only, a medical expert must review.")

    is_ae, serious, _ = detect_adverse_event(enquiry)
    if is_ae and serious:
        add("serious_adverse_event", "Possible serious adverse event (e.g. hospitalisation or severe reaction); escalate in addition to PV reporting.")

    if retrieval.products and not retrieval.confident:
        add("low_confidence", f"Low retrieval confidence (top BM25 score {retrieval.top_score:.2f}): approved documents may not cover this question.")

    return hits


def post_checks(draft_text: str, coverage: str, citation_check) -> list[RuleHit]:
    hits: list[RuleHit] = []
    add = lambda code, reason: hits.append(RuleHit(code, reason, "post"))
    if coverage == "NONE":
        add("no_supporting_source", "The draft states that the approved documents do not answer the question.")
    if citation_check is not None:
        if citation_check.invalid:
            add("citation_check_failed", f"Draft cites sources that were not retrieved: {', '.join(citation_check.invalid)}.")
        if coverage != "NONE" and not citation_check.cited:
            add("citation_check_failed", "Draft makes statements but contains no citations.")
        if citation_check.uncited_sentences:
            add("uncited_statements", f"{len(citation_check.uncited_sentences)} sentence(s) in the draft have no citation.")
        if citation_check.ungrounded:
            add("ungrounded_values", "Figures in the draft do not appear in the section they cite: "
                                     + "; ".join(citation_check.ungrounded))
    if re.search(PATTERNS["prohibited_claims"], draft_text or "", re.I):
        add("prohibited_claim", "Draft contains promotional or absolute claim language (e.g. 'cure', 'guaranteed', 'better than').")
    return hits


def triage(hits: list[RuleHit]) -> tuple[str, list[str]]:
    if hits:
        return ESCALATE, [f"[{h.stage}] {h.code}: {h.reason}" for h in hits]
    return DRAFT_OK, ["All deterministic checks passed; draft is grounded in retrieved approved sections."]
