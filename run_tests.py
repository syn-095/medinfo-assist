"""Run a test set headless and print expected vs actual.

Usage:
  python run_tests.py                                 # offline extractive (regex floor only)
  python run_tests.py --backend claude_cli --workers 4  # Claude Code subscription (dev eval only)
  python run_tests.py --backend api                   # Anthropic API (needs ANTHROPIC_API_KEY)
  python run_tests.py --data data/heldout.json        # any test set
Writes test_results.csv (one row per case) alongside the printed report.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from medinfo.generate import check_citations
from medinfo.llm import BACKENDS, select_backend
from medinfo.pipeline import process
from medinfo.retrieval import Retriever
from medinfo.rules import DRAFT_OK

ROOT = Path(__file__).parent
sys.stdout.reconfigure(encoding="utf-8")  # § and ✓ on Windows consoles


def citation_check_selftest(retriever: Retriever) -> bool:
    """The check must catch fake, unretrieved and missing citations, and wrong figures."""
    texts = {"VLX-SMPC-001 §4.2": retriever.by_id["VLX-SMPC-001 §4.2"].text}
    retrieved = set(texts)
    good = check_citations("The starting dose is 10 mg once daily. [VLX-SMPC-001 §4.2]", retrieved, texts)
    fake = check_citations("The starting dose is 10 mg once daily. [VLX-SMPC-001 §4.6]", retrieved, texts)
    made_up = check_citations("Veloxitab is safe in pregnancy. [VLX-SMPC-001 §9.9]", retrieved, texts)
    uncited = check_citations("The starting dose is 10 mg once daily.", retrieved, texts)
    wrong_dose = check_citations("The maximum dose is 40 mg once daily. [VLX-SMPC-001 §4.2]", retrieved, texts)
    return (good.passed and not fake.passed and not made_up.passed and not uncited.passed
            and not wrong_dose.passed and bool(wrong_dose.ungrounded))


def run_case(e: dict, retriever: Retriever, backend: str, key: str | None) -> tuple[dict, object]:
    t0 = time.monotonic()
    r = process(e["enquiry"], retriever, key, backend=backend)
    secs = time.monotonic() - t0
    actual = "answer" if r.decision == DRAFT_OK else "escalate"
    actual_regex = "answer" if r.decision_regex == DRAFT_OK else "escalate"
    exp_src = e.get("expected_source")
    src_hit = (exp_src in r.retrieval.ids) if exp_src else None
    src_cited = (exp_src in r.draft.citation_check.cited) if (exp_src and r.draft.mode != "not_covered") else None
    cc = r.draft.citation_check
    ls = r.llm_safety
    row = {
        "id": e["id"],
        "expected_triage": e["expected_triage"], "actual_triage": actual, "regex_only_triage": actual_regex,
        "triage_ok": actual == e["expected_triage"],
        "expected_pv": e["expected_pv"], "actual_pv": r.pv_flag, "regex_only_pv": r.pv_regex,
        "pv_ok": r.pv_flag == e["expected_pv"],
        "expected_source": exp_src or "", "source_retrieved": "" if src_hit is None else src_hit,
        "source_cited": "" if src_cited is None else src_cited,
        "backend": r.backend, "draft_backend": r.draft.backend, "model": r.draft.model or (ls.model if ls else ""),
        "mode": r.draft.mode,
        "citation_check": "n/a" if r.draft.mode == "not_covered" else ("PASS" if cc.passed else "FAIL"),
        "cited": "; ".join(cc.cited),
        "rules_fired": ", ".join(r.hit_codes),
        "llm_added": ", ".join(h.code for h in r.llm_hits),
        "layers": "; ".join(f"{k}={v}" for k, v in sorted(r.layers.items())),
        "llm_status": ls.status if ls else "",
        "llm_evidence": (ls.flags.get("evidence", "") if ls and ls.ok else ""),
        "seconds": round(secs, 1),
        "expected_rule": e.get("expected_rule") or "",
        "enquiry": e["enquiry"],
    }
    return row, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=BACKENDS, default=None,
                    help="default: auto (api if ANTHROPIC_API_KEY, else MEDINFO_BACKEND, else extractive)")
    ap.add_argument("--llm", action="store_true", help="legacy alias for --backend api")
    ap.add_argument("--workers", type=int, default=1, help="cases processed in parallel (LLM backends)")
    ap.add_argument("--data", default=str(ROOT / "data" / "test_enquiries.json"), help="test set JSON")
    ap.add_argument("--out", default=str(ROOT / "test_results.csv"))
    args = ap.parse_args()

    key = os.environ.get("ANTHROPIC_API_KEY")
    backend = select_backend("api" if args.llm else args.backend, key)
    if backend == "api" and not key:
        sys.exit("--backend api needs ANTHROPIC_API_KEY")
    retriever = Retriever(ROOT / "data" / "approved")
    data_path = Path(args.data)
    if not data_path.is_file():
        found = ", ".join(p.name for p in (ROOT / "data").glob("*.json")) or "none"
        sys.exit(f"Test set not found: {data_path}\nJSON files in data/: {found}")
    cases = json.loads(data_path.read_text(encoding="utf-8"))

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = list(pool.map(lambda e: run_case(e, retriever, backend, key), cases))
    wall = time.monotonic() - t0
    rows = [row for row, _ in results]
    full = {row["id"]: r for row, r in results}

    # ---- Table ----
    hdr = (f"{'ID':<4} {'Expected':<9} {'Actual':<9} {'OK':<3} {'PV exp/act':<11} {'Ret':<4} {'Cit':<4} "
           f"{'Check':<5} {'Layers (flag=regex/llm/both)'}")
    print(hdr)
    print("-" * (len(hdr) + 30))
    yn = lambda b: "Y" if b else "N"
    for x in rows:
        src = "-" if x["source_retrieved"] == "" else yn(x["source_retrieved"])
        cit = "-" if x["source_cited"] == "" else yn(x["source_cited"])
        ok = "✓" if (x["triage_ok"] and x["pv_ok"]) else "✗"
        llm_err = "  [LLM classifier ignored: " + x["llm_status"][:60] + "]" if x["llm_status"].startswith("error") else ""
        print(f"{x['id']:<4} {x['expected_triage']:<9} {x['actual_triage']:<9} {ok:<3} "
              f"{yn(x['expected_pv'])}/{yn(x['actual_pv']):<9} {src:<4} {cit:<4} {x['citation_check']:<5} "
              f"{x['layers'] or '-'}{llm_err}")

    # ---- Summary ----
    n = len(rows)
    t_ok = sum(x["triage_ok"] for x in rows)
    t_ok_regex = sum(x["regex_only_triage"] == x["expected_triage"] for x in rows)
    pv_ok = sum(x["pv_ok"] for x in rows)
    pv_ok_regex = sum(x["regex_only_pv"] == x["expected_pv"] for x in rows)
    drafted = [x for x in rows if x["citation_check"] != "n/a"]
    cc_ok = sum(x["citation_check"] == "PASS" for x in drafted)
    src_rows = [x for x in rows if x["source_retrieved"] != ""]
    src_ok = sum(bool(x["source_retrieved"]) for x in src_rows)
    cit_rows = [x for x in rows if x["source_cited"] != ""]
    cit_ok = sum(bool(x["source_cited"]) for x in cit_rows)
    cit_miss = [x["id"] for x in cit_rows if not x["source_cited"]]
    missed_esc = [x["id"] for x in rows if x["expected_triage"] == "escalate" and x["actual_triage"] == "answer"]
    over_esc = [x["id"] for x in rows if x["expected_triage"] == "answer" and x["actual_triage"] == "escalate"]
    # Layer attribution (only meaningful with an LLM backend)
    llm_triage_catch = [x["id"] for x in rows if x["expected_triage"] == "escalate"
                        and x["regex_only_triage"] == "answer" and x["actual_triage"] == "escalate"]
    llm_pv_catch = [x["id"] for x in rows if x["expected_pv"] and not x["regex_only_pv"] and x["actual_pv"]]
    llm_over_esc = [x["id"] for x in rows if x["expected_triage"] == "answer"
                    and x["regex_only_triage"] == "answer" and x["actual_triage"] == "escalate"]
    llm_pv_fp = [x["id"] for x in rows if not x["expected_pv"] and not x["regex_only_pv"] and x["actual_pv"]]
    llm_errors = [x["id"] for x in rows if x["llm_status"].startswith("error")]
    llm_fallback = [x["id"] for x in rows if "->extractive" in x["draft_backend"]]

    print()
    model = next((x["model"] for x in rows if x["model"]), "-")
    print(f"Test set: {Path(args.data).name} ({n} cases) · backend: {backend} · model: {model} · "
          f"wall {wall:.0f}s, workers {args.workers}")
    print(f"Triage correct:          {t_ok}/{n}   (regex floor alone: {t_ok_regex}/{n})")
    print(f"  Missed escalations:    {len(missed_esc)} {missed_esc}   <- the safety-critical error")
    print(f"  Over-escalations:      {len(over_esc)} {over_esc}")
    print(f"PV flag correct:         {pv_ok}/{n}   (regex floor alone: {pv_ok_regex}/{n})")
    print(f"Expected source in top-k:{src_ok}/{len(src_rows)}")
    print(f"Expected source cited:   {cit_ok}/{len(cit_rows)} drafted {cit_miss}")
    print(f"Citation check passed:   {cc_ok}/{len(drafted)} drafted responses")
    print(f"Citation check self-test (catches fake/unretrieved/missing citations and wrong figures): "
          f"{'PASS' if citation_check_selftest(retriever) else 'FAIL'}")
    if backend != "extractive":
        print(f"LLM-only catches:        triage {llm_triage_catch}  PV {llm_pv_catch}")
        print(f"LLM-added over-escalations: {llm_over_esc}   LLM-added PV false positives: {llm_pv_fp}")
        print(f"LLM classifier errors (ignored, regex kept): {llm_errors}   Drafting fallbacks to extractive: {llm_fallback}")

    # ---- Detail: LLM-only flags (any flag the regex did not raise) ----
    llm_only = [(x["id"], x["layers"], x["llm_evidence"]) for x in rows if "=llm" in x["layers"]]
    if llm_only:
        print("\nFlags raised by the LLM only:")
        for cid, layers, ev in llm_only:
            flags = [kv.split("=")[0] for kv in layers.split("; ") if kv.endswith("=llm")]
            print(f"  {cid}: {', '.join(flags)}   evidence: “{ev}”")

    # ---- Detail: every citation / grounding failure ----
    fails = [x for x in drafted if x["citation_check"] == "FAIL"]
    if fails:
        print("\nCitation / grounding failures:")
        for x in fails:
            cc = full[x["id"]].draft.citation_check
            for c in cc.invalid:
                print(f"  {x['id']}: cites unretrieved section {c}")
            for s in cc.uncited_sentences:
                print(f"  {x['id']}: uncited sentence: “{s}”")
            for g in cc.ungrounded:
                print(f"  {x['id']}: ungrounded value: {g}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
