# MedInfo Assist: build notes

Running log of design decisions, assumptions, and where AI tools helped or failed. Raw material for the deck.

## Design decisions

- **Synthetic data first.** 2 fictional products: Veloxitab (velorastat tablets, hypertension) and Corvanex (corvatide weekly pen, type 2 diabetes). Each has an SmPC-style doc plus an approved FAQ / standard-response doc. Each doc has YAML-style front matter (doc_id, version, approval_date, status) so every citation can be traced to a specific approved version.
- **Section = retrieval chunk = citation unit.** SmPC sections use real SmPC numbering (4.2 posology, 4.5 interactions, 4.6 pregnancy...), FAQs use Q1..Qn. Citation format `[DOC-ID §section]`, e.g. `[VLX-SMPC-001 §4.2]`. This makes the citation check a simple set comparison against retrieved chunk IDs.
- **Test set designed around the rules**, not around the happy path: 12 answerable, 12 escalations (2 per main rule: off-label, pregnancy/lactation, paediatric/elderly, overdose, uncovered interaction; 1 each competitor, patient-specific, no-source), 3 adverse events, 3 traps (prompt injection, unknown product, out of scope). Deliberate contrast pairs: clarithromycin (covered interaction, answer) vs lithium (not covered, escalate); general "what are the side effects" (no PV) vs "a patient developed a rash" (PV).
- **PV flag is independent of triage.** An AE always raises the PV flag; the triage decision is made separately (a factual "is rash a known side effect" can still be drafted). Serious AEs (hospitalisation, life-threatening, death) also force escalation. This is a design choice; a stricter company SOP might escalate every AE.

## Assumptions

- Enquiries are in English, free text, one enquiry per submission.
- Overdose with no harm reported is treated as escalation, not as an AE (many PV SOPs would also log it as a special situation; noted as an open question).
- Pregnancy exposure is escalated; in real PV, pregnancy exposure is usually also a reportable "special situation". Kept separate for clarity.

## Where AI tools helped / failed

- Claude Code (Opus) generated the synthetic SmPCs, FAQs and the 30 labelled enquiries in one pass from the brief. Helped: fast, consistent section numbering, realistic SmPC tone. Needs human check: the expected labels are my judgement and are the ground truth for testing, so they were reviewed rather than trusted.
