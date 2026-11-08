I'm doing a take-home build task for Ankura's AI by Design summer internship (Digital AI Advisory). It's due Wed 11 Nov 17:30 GMT. Today (Sun 8 Nov) I have 12:30–16:30 to get a WORKING prototype. Tests, the deck, the README and hosting happen Mon–Tue, so don't do those today beyond what I list below. Be efficient: make sensible decisions yourself, ask me only about things that block you, and keep explanations short.

## The brief (Scenario 3: Pharma, medical information)
A fictional pharma company receives product information enquiries. The information team searches approved documents, drafts responses and asks experts to review questions that need specialist judgement. Responses must stay traceable to approved source material, with a clear approach to expert review, escalation and staff training. Fictional products and enquiries only; synthetic data only. Assessors also care about business value, risks/controls/human oversight, workflow and role changes, assumptions and trade-offs, and an assessment of the AI tools used.

## What to build: "MedInfo Assist"
A small app where a medical information officer pastes in an enquiry and gets:
1. **A draft response grounded only in approved documents**, with inline citations (doc ID + section) for every factual sentence. If the approved docs don't cover the question, it says so and doesn't answer from general knowledge.
2. **A triage decision**: `DRAFT OK FOR MI OFFICER REVIEW` or `ESCALATE TO MEDICAL EXPERT`, with the reason(s) given.
3. **A safety flag**: if the enquiry mentions a possible adverse event, flag it for pharmacovigilance reporting, whatever else happens.
4. **A human review step**: the officer can approve, edit or escalate, and every action is written to an audit log (timestamp, enquiry, retrieved sources, draft, decision, reviewer action).

Nothing is ever sent to a customer automatically. A human always signs off.

## Escalation rules (deterministic, checked before and after the LLM)
Escalate if any of these apply: off-label use; pregnancy/breastfeeding; paediatric or elderly dosing not in the label; overdose; drug interactions not covered in the docs; comparative claims against competitor products; patient-specific treatment advice; low retrieval confidence or no supporting source; or the LLM's draft cites something that isn't in the retrieved chunks (citation check fails). Adverse events → PV flag, as in point 3.

## Synthetic data (generate this first)
- 2 fictional products (e.g. "Veloxitab" tablets, "Corvanex" injection). For each: an approved SmPC-style document (indications, dosing, contraindications, warnings, interactions, adverse effects, storage, pregnancy section saying "insufficient data, consult HCP"), plus 1–2 approved standard response letters / FAQs. Markdown files in `data/approved/`, each with a doc ID, version and approval date at the top.
- `data/test_enquiries.json`: 30 enquiries with the expected outcome (`answer` / `escalate` / `pv_flag`), the expected source doc and a short rationale. Mix: about 12 answerable, about 12 should escalate (spread across the rules), about 3 adverse-event mentions, about 3 traps (out of scope, prompt-injection-style "ignore your instructions", a question about a product not in the docs).

## Tech (keep it simple)
- Python + Streamlit (one `app.py`), so it can go on Streamlit Community Cloud on Tuesday.
- Retrieval: chunk by document section; BM25 (`rank_bm25`) is fine. No vector DB needed. Show the retrieved chunks and scores in the UI.
- Generation: Anthropic API (`claude-sonnet-5-5`), key from env/`st.secrets`. The system prompt requires answers only from the provided chunks, citations in `[DOC-ID §section]` form, and refusal when the docs don't cover the question.
- **Offline fallback mode**: if no API key is set, give an extractive answer (the top retrieved sections, quoted with citations) so assessors can still test the hosted app without a key. The rules engine works in both modes.
- Audit log: append to `audit_log.csv`, viewable in a sidebar tab.
- Structure: `app.py`, `medinfo/retrieval.py`, `medinfo/rules.py`, `medinfo/generate.py`, `medinfo/audit.py`, `data/`, `requirements.txt`. Use git from the start, with small commits.

## Done by 16:30 means
- [ ] Synthetic docs + 30 test enquiries written
- [ ] `streamlit run app.py` works locally: enquiry in → citations, triage, PV flag, approve/edit/escalate → audit log row
- [ ] A citation check that verifies every cited section was actually retrieved
- [ ] `run_tests.py` that runs all 30 enquiries headless and prints a basic table (expected vs actual triage, citation check pass/fail). Getting it to run is enough for today; I'll do the analysis on Monday.
- [ ] A short `NOTES.md` that logs, as you go, key design decisions, assumptions, and where AI tools helped or failed. I'll use it for the deck.

Order: data → retrieval → rules → generation + citation check → UI → audit → test runner. If time gets tight, drop UI polish first, never the rules, citation check or audit log. Start now: give me a 5-line plan, then build.
