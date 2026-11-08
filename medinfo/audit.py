"""Append-only CSV audit log. One row per event (system draft, reviewer action)."""
from __future__ import annotations

import csv
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

COLUMNS = [
    "timestamp_utc", "case_id", "event", "reviewer", "enquiry", "retrieved_sources", "generation_mode",
    "draft", "citation_check", "triage_decision", "triage_reasons", "pv_flag", "pv_reason", "quality_flag", "reviewer_action",
    "final_response", "reviewer_comment",
]


def new_case_id() -> str:
    return "MI-" + datetime.now(timezone.utc).strftime("%Y%m%d") + "-" + uuid.uuid4().hex[:6].upper()


def log_event(path: str | Path, case_id: str, event: str, result, reviewer: str = "system",
              reviewer_action: str = "", final_response: str = "", reviewer_comment: str = "") -> dict:
    path = Path(path)
    cc = result.draft.citation_check
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "case_id": case_id,
        "event": event,
        "reviewer": reviewer,
        "enquiry": result.enquiry,
        "retrieved_sources": json.dumps([
            {"id": h.chunk.chunk_id, "version": h.chunk.version, "score": round(h.score, 2)}
            for h in result.retrieval.hits], ensure_ascii=False),
        "generation_mode": result.draft.mode,
        "draft": result.draft.text,
        "citation_check": json.dumps({"passed": cc.passed, "cited": cc.cited, "invalid": cc.invalid,
                                      "uncited": len(cc.uncited_sentences),
                                      "ungrounded": cc.ungrounded}, ensure_ascii=False),
        "triage_decision": result.decision,
        "triage_reasons": " | ".join(result.reasons),
        "pv_flag": "YES" if result.pv_flag else "NO",
        "pv_reason": result.pv_reason,
        "quality_flag": "YES" if result.quality_flag else "NO",
        "reviewer_action": reviewer_action,
        "final_response": final_response,
        "reviewer_comment": reviewer_comment,
    }
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), [])
        if header != COLUMNS:
            # Schema changed: archive the old log rather than appending misaligned rows.
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            path.rename(path.with_name(f"{path.stem}.archived-{stamp}{path.suffix}"))
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)
    return row


def read_log(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
