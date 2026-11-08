"""MedInfo Assist - Streamlit prototype (synthetic data, fictional products only).

Run:  streamlit run app.py
Set ANTHROPIC_API_KEY (env var or .streamlit/secrets.toml) for Claude drafting;
without it the app runs in offline extractive mode.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from medinfo import audit, rules
from medinfo.generate import MODEL, check_citations
from medinfo.pipeline import process
from medinfo.retrieval import Retriever

ROOT = Path(__file__).parent
DOC_DIR = ROOT / "data" / "approved"
AUDIT_PATH = ROOT / "audit_log.csv"

st.set_page_config(page_title="MedInfo Assist", page_icon="💊", layout="wide")


@st.cache_resource
def get_retriever() -> Retriever:
    return Retriever(DOC_DIR)


def get_api_key() -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        return None


@st.cache_data
def examples() -> list[dict]:
    return json.loads((ROOT / "data" / "test_enquiries.json").read_text(encoding="utf-8"))


retriever = get_retriever()
api_key = get_api_key()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("💊 MedInfo Assist")
    st.caption("Prototype · synthetic data · fictional products")
    page = st.radio("View", ["Enquiry workbench", "Audit log", "Approved documents"])
    st.divider()
    reviewer = st.text_input("Reviewer (MI officer)", value="MI Officer 1")
    if api_key:
        st.success(f"Drafting mode: Claude ({MODEL})")
        use_llm = st.toggle("Use Claude for drafting", value=True)
    else:
        st.info("Drafting mode: **offline extractive** (no API key set). Rules, citation check and audit work as normal.")
        use_llm = False
    st.caption("Nothing is ever sent to an enquirer automatically. Every response needs human sign-off.")


# ---------------- Enquiry workbench ----------------
def workbench():
    st.header("Enquiry workbench")
    ex = examples()
    labels = ["(type your own)"] + [f"{e['id']} · {e['enquiry'][:80]}" for e in ex]
    pick = st.selectbox("Load a synthetic test enquiry (optional)", labels)
    default = "" if pick == labels[0] else ex[labels.index(pick) - 1]["enquiry"]
    enquiry = st.text_area("Enquiry text", value=default, height=110, key=f"enq_{pick}")

    if st.button("Process enquiry", type="primary", disabled=not enquiry.strip()):
        with st.spinner("Retrieving approved sources, checking rules, drafting..."):
            res = process(enquiry.strip(), retriever, api_key if use_llm else None)
        case_id = audit.new_case_id()
        audit.log_event(AUDIT_PATH, case_id, "SYSTEM_DRAFT", res, reviewer="system")
        st.session_state.update(result=res, case_id=case_id, actioned=False)

    res = st.session_state.get("result")
    if not res:
        return
    case_id = st.session_state["case_id"]
    st.divider()
    st.subheader(f"Case {case_id}")

    # Safety flag first: it applies whatever else happens.
    if res.pv_flag:
        st.error(f"⚠️ **PHARMACOVIGILANCE FLAG**: {res.pv_reason}")

    escalate = res.decision == rules.ESCALATE
    (st.warning if escalate else st.success)(f"**Triage: {res.decision}**")
    with st.expander("Triage reasons", expanded=escalate):
        for r in res.reasons:
            st.markdown(f"- {r}")

    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"**Draft response** · mode: `{res.draft.mode}` · coverage: `{res.draft.coverage}`")
        if res.draft.note:
            st.caption(res.draft.note)
        edited = st.text_area("Review and edit the draft", value=res.draft.text, height=260, key=f"draft_{case_id}")
        cc = check_citations(edited, res.retrieval.ids, res.retrieval.texts) if edited != res.draft.text else res.draft.citation_check
        if res.draft.mode == "not_covered":
            st.info("No draft generated: the approved documents do not support an answer.")
        elif cc.passed:
            st.success(f"Citation check passed: {len(cc.cited)} cited section(s), all among the retrieved sources.")
        else:
            msg = []
            if cc.invalid:
                msg.append(f"cites sections that were not retrieved: {', '.join(cc.invalid)}")
            if not cc.cited:
                msg.append("contains no citations")
            if cc.uncited_sentences:
                msg.append(f"{len(cc.uncited_sentences)} uncited sentence(s)")
            if cc.ungrounded:
                msg.append(f"{len(cc.ungrounded)} sentence(s) with figures not found in the cited section")
            st.error("Citation check FAILED: " + "; ".join(msg))
            for s in cc.uncited_sentences:
                st.caption(f"Uncited: “{s}”")
            for s in cc.ungrounded:
                st.caption(f"Not in cited source: {s}")

    with right:
        st.markdown("**Retrieved approved sources**")
        if not res.retrieval.hits:
            st.caption("No sections retrieved.")
        for h in res.retrieval.hits:
            c = h.chunk
            cited = "✅ cited" if c.chunk_id in res.draft.citation_check.cited else ""
            with st.expander(f"{c.chunk_id} · {c.heading} · score {h.score:.2f} {cited}"):
                st.caption(f"{c.title} · v{c.version} · approved {c.approval_date}")
                st.write(c.text)
        st.caption(f"Products detected: {', '.join(res.retrieval.products) or 'none'} · "
                   f"top score {res.retrieval.top_score:.2f}")

    # ---------------- Human review ----------------
    st.divider()
    st.markdown("**Reviewer decision**")
    if st.session_state.get("actioned"):
        st.success("Action recorded in the audit log. Process a new enquiry to continue.")
        return
    pv_ok = True
    if res.pv_flag:
        pv_ok = st.checkbox("I have forwarded this enquiry to pharmacovigilance (required before any action).")
    comment = st.text_input("Reviewer comment (required to escalate or to approve edited text)")
    edited_changed = edited.strip() != res.draft.text.strip()

    if res.pv_flag and pv_ok:
        comment_logged = "[PV referral confirmed by reviewer] " + comment
    else:
        comment_logged = comment

    c1, c2, c3 = st.columns(3)
    can_approve = pv_ok and not escalate and res.draft.mode != "not_covered"
    with c1:
        if st.button("✅ Approve draft as is", disabled=not can_approve or edited_changed):
            audit.log_event(AUDIT_PATH, case_id, "REVIEWER_ACTION", res, reviewer, "APPROVED", res.draft.text, comment_logged)
            st.session_state["actioned"] = True
            st.rerun()
    with c2:
        if st.button("✏️ Approve edited draft", disabled=not can_approve or not edited_changed or not comment.strip()):
            note = comment_logged + ("" if cc.passed else " [edited text failed citation check at approval]")
            audit.log_event(AUDIT_PATH, case_id, "REVIEWER_ACTION", res, reviewer, "EDITED_AND_APPROVED", edited, note)
            st.session_state["actioned"] = True
            st.rerun()
    with c3:
        if st.button("⬆️ Escalate to medical expert", disabled=not pv_ok or not comment.strip()):
            audit.log_event(AUDIT_PATH, case_id, "REVIEWER_ACTION", res, reviewer, "ESCALATED", edited, comment_logged)
            st.session_state["actioned"] = True
            st.rerun()
    if escalate:
        st.caption("Approval is disabled: the rules engine requires medical expert review for this case.")
    elif not pv_ok:
        st.caption("Confirm the pharmacovigilance referral first.")


# ---------------- Audit log ----------------
def audit_page():
    st.header("Audit log")
    rows = audit.read_log(AUDIT_PATH)
    if not rows:
        st.info("No events logged yet.")
        return
    df = pd.DataFrame(rows).iloc[::-1]
    st.caption(f"{len(df)} events · append-only CSV at `{AUDIT_PATH.name}`")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("Download audit_log.csv", AUDIT_PATH.read_bytes(), "audit_log.csv", "text/csv")


# ---------------- Approved documents ----------------
def docs_page():
    st.header("Approved document library")
    st.caption("Only documents with status 'Approved' are indexed. Each section is one retrievable, citable chunk.")
    by_doc: dict[str, list] = {}
    for c in retriever.chunks:
        by_doc.setdefault(c.doc_id, []).append(c)
    for doc_id, chunks in by_doc.items():
        c0 = chunks[0]
        with st.expander(f"{doc_id} · {c0.title} · v{c0.version} · approved {c0.approval_date}"):
            for c in chunks:
                st.markdown(f"**§{c.section} {c.heading}**")
                st.write(c.text)


{"Enquiry workbench": workbench, "Audit log": audit_page, "Approved documents": docs_page}[page]()
