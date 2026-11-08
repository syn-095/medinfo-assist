"""Run all synthetic test enquiries headless and print expected vs actual.

Usage:
  python run_tests.py            # offline extractive mode (no API calls)
  python run_tests.py --llm      # use Claude if ANTHROPIC_API_KEY is set
Writes test_results.csv alongside the printed table.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from medinfo.generate import check_citations
from medinfo.pipeline import process
from medinfo.retrieval import Retriever
from medinfo.rules import DRAFT_OK

ROOT = Path(__file__).parent
sys.stdout.reconfigure(encoding="utf-8")  # § and ✓ on Windows consoles


def citation_check_selftest(retriever: Retriever) -> bool:
    """The check must catch a citation to a section that was not retrieved."""
    texts = {"VLX-SMPC-001 §4.2": retriever.by_id["VLX-SMPC-001 §4.2"].text}
    retrieved = set(texts)
    good = check_citations("The starting dose is 10 mg once daily. [VLX-SMPC-001 §4.2]", retrieved, texts)
    fake = check_citations("The starting dose is 10 mg once daily. [VLX-SMPC-001 §4.6]", retrieved, texts)
    made_up = check_citations("Veloxitab is safe in pregnancy. [VLX-SMPC-001 §9.9]", retrieved, texts)
    uncited = check_citations("The starting dose is 10 mg once daily.", retrieved, texts)
    # Real, retrieved citation but a wrong figure: must fail the grounding check.
    wrong_dose = check_citations("The maximum dose is 40 mg once daily. [VLX-SMPC-001 §4.2]", retrieved, texts)
    return (good.passed and not fake.passed and not made_up.passed and not uncited.passed
            and not wrong_dose.passed and bool(wrong_dose.ungrounded))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="use Claude (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--data", default=str(ROOT / "data" / "test_enquiries.json"),
                    help="test set JSON (e.g. a held-out set)")
    ap.add_argument("--out", default=str(ROOT / "test_results.csv"))
    args = ap.parse_args()

    key = os.environ.get("ANTHROPIC_API_KEY") if args.llm else None
    if args.llm and not key:
        print("--llm given but ANTHROPIC_API_KEY is not set; running offline.")
    retriever = Retriever(ROOT / "data" / "approved")
    cases = json.loads(Path(args.data).read_text(encoding="utf-8"))

    rows = []
    for e in cases:
        r = process(e["enquiry"], retriever, key)
        actual = "answer" if r.decision == DRAFT_OK else "escalate"
        exp_src = e.get("expected_source")
        src_hit = (exp_src in r.retrieval.ids) if exp_src else None
        # Cited is stricter than retrieved: only meaningful when a draft was produced.
        src_cited = (exp_src in r.draft.citation_check.cited) if (exp_src and r.draft.mode != "not_covered") else None
        cc = r.draft.citation_check
        rows.append({
            "id": e["id"],
            "expected_triage": e["expected_triage"], "actual_triage": actual,
            "triage_ok": actual == e["expected_triage"],
            "expected_pv": e["expected_pv"], "actual_pv": r.pv_flag, "pv_ok": r.pv_flag == e["expected_pv"],
            "expected_source": exp_src or "", "source_retrieved": "" if src_hit is None else src_hit,
            "source_cited": "" if src_cited is None else src_cited,
            "mode": r.draft.mode,
            "citation_check": "n/a" if r.draft.mode == "not_covered" else ("PASS" if cc.passed else "FAIL"),
            "cited": "; ".join(cc.cited),
            "rules_fired": ", ".join(r.hit_codes),
            "expected_rule": e.get("expected_rule") or "",
            "enquiry": e["enquiry"],
        })

    # ---- Table ----
    hdr = f"{'ID':<4} {'Expected':<9} {'Actual':<9} {'OK':<3} {'PV exp/act':<11} {'Ret':<4} {'Cit':<4} {'Check':<5} Rules fired"
    print(hdr)
    print("-" * len(hdr) + "-" * 30)
    yn = lambda b: "Y" if b else "N"
    for x in rows:
        src = "-" if x["source_retrieved"] == "" else yn(x["source_retrieved"])
        cit = "-" if x["source_cited"] == "" else yn(x["source_cited"])
        ok = "✓" if (x["triage_ok"] and x["pv_ok"]) else "✗"
        print(f"{x['id']:<4} {x['expected_triage']:<9} {x['actual_triage']:<9} {ok:<3} "
              f"{yn(x['expected_pv'])}/{yn(x['actual_pv']):<9} {src:<4} {cit:<4} {x['citation_check']:<5} {x['rules_fired']}")

    # ---- Summary ----
    n = len(rows)
    t_ok = sum(x["triage_ok"] for x in rows)
    pv_ok = sum(x["pv_ok"] for x in rows)
    drafted = [x for x in rows if x["citation_check"] != "n/a"]
    cc_ok = sum(x["citation_check"] == "PASS" for x in drafted)
    src_rows = [x for x in rows if x["source_retrieved"] != ""]
    src_ok = sum(bool(x["source_retrieved"]) for x in src_rows)
    cit_rows = [x for x in rows if x["source_cited"] != ""]
    cit_ok = sum(bool(x["source_cited"]) for x in cit_rows)
    cit_miss = [x["id"] for x in cit_rows if not x["source_cited"]]
    missed_esc = [x["id"] for x in rows if x["expected_triage"] == "escalate" and x["actual_triage"] == "answer"]
    over_esc = [x["id"] for x in rows if x["expected_triage"] == "answer" and x["actual_triage"] == "escalate"]
    print()
    print(f"Test set: {Path(args.data).name} ({n} cases) · Mode: {'Claude' if key else 'offline extractive'}")
    print(f"Triage correct:          {t_ok}/{n}")
    print(f"  Missed escalations:    {len(missed_esc)} {missed_esc}   <- the safety-critical error")
    print(f"  Over-escalations:      {len(over_esc)} {over_esc}")
    print(f"PV flag correct:         {pv_ok}/{n}")
    print(f"Expected source in top-k:{src_ok}/{len(src_rows)}")
    print(f"Expected source cited:   {cit_ok}/{len(cit_rows)} drafted {cit_miss}")
    print(f"Citation check passed:   {cc_ok}/{len(drafted)} drafted responses")
    print(f"Citation check self-test (catches fake/unretrieved/missing citations and wrong figures): "
          f"{'PASS' if citation_check_selftest(retriever) else 'FAIL'}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
