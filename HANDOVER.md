# Handover (overnight, Tue 10 → Wed 11 Nov)

Due **Wed 11 Nov, 17:30 GMT**. All work is on branch `worktree-overnight-docs` (a worktree of this repo). Nothing has been pushed, and no GitHub repo has been created.

## What is done

- **Supporting document**: `docs/deck.html`, exported to `docs/MedInfo_Assist_William_Wilson.pdf`. **5 pages, A4.** Each page was checked in print layout: no overflow, nothing cut off, the footer is clear on every page. Architecture diagram drawn as inline SVG. UI screenshot in `docs/img/ui_draft_pv.png`. It didn't fit in the PDF without cutting content, so it is used only in the repo.
- **Held-out results reported** alongside the training set on page 3 (and the trade-off on page 5), leading with missed escalations, with provenance taken from each case's `rationale`, and n = 15 called indicative.
- **README.md**: what it is, quick start, backends, tests, 8 sample enquiries (each checked offline: answer ×2, escalation ×2, PV flag, PV + serious escalation, QA flag, prompt injection), folder layout, limitations.
- **requirements.txt** checked: every third-party import (`streamlit`, `rank_bm25`, `anthropic`, `pandas`) is listed. Tested in a clean venv (resolved to streamlit 1.67.0, rank-bm25 0.2.2, anthropic 2.8.0, pandas 3.1.1, numpy 2.5.4): the held-out offline run matched the committed CSV exactly, and a Streamlit AppTest with no key loaded all three pages and processed an enquiry with no exceptions. Upper bounds were added at the tested majors, and streamlit is capped below 1.68, because the app uses `use_container_width`, which Streamlit has deprecated.
- **App starts with no API key**: launched headless with `ANTHROPIC_API_KEY` and `MEDINFO_BACKEND` unset. It showed "Backend: offline extractive" and processed an enquiry end to end (PV flag, draft, citation check).
- **Offline results reproduce exactly**: `run_tests.py` on both sets matches the committed CSVs, field for field (excluding timings).
- **.gitignore** rewritten. It still had a UTF-8 BOM, which silently broke its first rule. It now covers `.env`, `.streamlit/secrets.toml`, `audit_log*.csv`, venvs, `__pycache__` and `.claude/worktrees/`. `test_results*.csv` are now tracked, because the document cites them.
- **Hosting check**: every path is built from `Path(__file__).parent`, with no absolute paths. The app runs from the repo root. A scan of the tracked files found no emails, keys or absolute user paths.
- **NOTES.md**: a new "Known issues" section lists the held-out failure modes. None were fixed (build frozen).
- **Frozen**: no changes to the pipeline, rules, prompts or test labels.

Commits (in order): `.gitignore` → held-out data + 4 result CSVs → `docs/` → NOTES known issues → requirements bounds → README → this file.

## Every number in the document, with its source

Abbreviations: **TR** = `test_results.csv` (training, offline), **TRC** = `test_results_claude_cli.csv` (training, claude_cli), **HO** = `test_results_heldout.csv`, **HOC** = `test_results_heldout_cli.csv`. "Missed" = `expected_triage=escalate` and `actual_triage=answer`; "over" = the reverse.

**Page 1**
- 35 sections, 4 approved docs: count of `##` sections in `data/approved/*.md` (13 + 12 + 5 + 5).

**Page 2**
- `claude-sonnet-5-5` pinned for the API: `medinfo/llm.py` `API_MODEL`.
- `claude-sonnet-5`, 55/55 calls: TRC `draft_model` (25 drafted rows) + `classifier_model` (30 rows). Also NOTES.md line 93.
- `claude-sonnet-5-5`, 27/27 calls: HOC `draft_model` (12) + `classifier_model` (15).
- Latency roughly 8–20 s per call: NOTES.md, Backends, `claude_cli` bullet.
- 30 labelled enquiries, 30/30: `data/test_enquiries.json`; TR `triage_ok`.
- First CLI run lost 19/30 classifier and 14 drafting replies, 0 missed escalations: NOTES.md, "claude_cli runs on test_enquiries.json", first paragraph.
- §4.8 boilerplate citation: NOTES.md, "Observations from the first live claude_cli call" (E26).

**Page 3**
- Training mix 30 = 12 answerable, 12 escalations, 3 AEs, 3 traps: NOTES.md, Design decisions. 15 held-out cases: `data/heldout.json`. Provenance: each case's `rationale`.
- Table, training offline: missed 0/17, over 0/13, triage 30/30, PV 29/30 (0 missed; the one error is E23, a false positive), source cited 22/23, citation 25/25: TR.
- Table, training claude_cli: 0/17, 3/13 (E02, E03, E10), 27/30, PV 26/30 (0 missed; false positives E10, E14, E17, E23), 23/23, 22/25 (fails E17, E19, E25), LLM-only catches 0 and 0: TRC (`escalated_by`, `pv_by`, `regex_pv`).
- Table, held-out offline: 3/7 (H01, H10, H13), 2/8 (H05, H15), 10/15, PV 11/15 (3 missed: H01, H02, H03; false positive H06), 9/11 (H02, H12 not cited), 12/12: HO.
- Table, held-out claude_cli: 0/7, 5/8 (H02, H05, H06, H11, H15), 10/15, PV 13/15 (0 missed; false positives H06, H14), 9/11, 9/12 (fails H01, H06, H13). "3 triage": `rules_pre_triage` missed H01, H10, H13, and the final result escalated all three, with `llm_added` containing the needed code. "3 PV": `pv_by=llm` with `regex_pv=False` and `expected_pv=True` on H01, H02, H03 (H14 is also `pv_by=llm` but is a false positive).
- 17 and 13 denominators: count of `expected_triage` in `data/test_enquiries.json`. 7 and 8: same in `data/heldout.json`. 6 PV: `expected_pv=true` in `heldout.json`. 11: held-out rows with a non-empty `expected_source`.
- Per-case causes (H05 regex patient_specific, H15 low_confidence + `quality=llm`, H02 `no_supporting_source`, H06 `uncited_statements, ungrounded_values`, H11 `llm_added=patient_specific`, H12/H14 `off_label`): HO/HOC columns `rules_fired`, `llm_added`, `layers`, `escalated_by`.
- Ages "16"/"86": the enquiry text. "The regex knows hospitalised but not in hospital": the `serious_ae` pattern in `medinfo/rules.py`.
- E25, E27 / E17, E18 (coverage statement / age false-fails): NOTES.md (current run: E17, E25 in TRC; E18, E27 from the earlier run).
- **Not in any CSV:** H01, H13 (age) and H06 (tablet strength) as the specific values behind `ungrounded_values`. That came from your overnight prompt (from the run's console output); the CSV records only that the check failed. Please confirm.

**Page 5**
- 1,000 enquiries/month, 60% answerable, 20 min, 8 min, 12 min saved: **assumptions**, labelled as such in the document.
- 6/8 offline, 3/8 claude_cli: HO/HOC, answerable cases with `actual_triage=answer` (offline: H02, H04, H06, H07, H08, H11; claude_cli: H04, H07, H08).
- 13/13, 10/13: TR / TRC, answerable cases drafted.
- 90 h = 1,000 × 0.6 × 6/8 × 12 / 60; 45 h = the same × 3/8.
- 3 → 0 of 7; 3 → 0 of 6; 2 → 5 of 8; PV false positives 1 → 2: HO vs HOC.
- Training: +3 over-escalations, +3 PV false positives: TRC (E02, E03, E10; `pv_by=llm` on E10, E14, E17).
- 4 documents, 2 products, held-out 15: `data/`.

## Open placeholders

- **None in the document** (no TODO or bracketed placeholders).
- No live app URL is in the PDF or README yet. Add it after deploying (see below), if you want it there.
- The **API path has never been run**, and the document says so. If you get a key tomorrow, do not re-tune anything; at most run it and mention it in one line.
- `TODAY_PROMPT.md` is tracked in the repo. `DECK_PROMPT.md` and `OVERNIGHT_PROMPT.md` sit untracked in the main checkout and were not committed. Decide whether you want your build prompts in a public repo (they contain no personal data, but they do name Ankura and the deadline).
- NOTES.md mentions that the CLI injected "the subscriber's email address". The address itself appears nowhere in the repo.

## Your steps for tomorrow

1. **Held-out set: already filled in.** The page 3 table and page 5 trade-off use the 10 Nov run. Read pages 3 and 5 against the number list above, especially the one item marked "Not in any CSV".
2. **Bring the branch into main** (from the main checkout, `...\Internship stuff\Ankura`):
   ```powershell
   git merge --ff-only worktree-overnight-docs
   git worktree remove .claude/worktrees/overnight-docs
   ```
3. **Re-export the PDF** (only if you edit `docs/deck.html`), from the repo root in PowerShell:
   ```powershell
   & "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --headless --disable-gpu --no-pdf-header-footer --print-to-pdf="$PWD\docs\MedInfo_Assist_William_Wilson.pdf" "$PWD\docs\deck.html"
   ```
   Then check it is still 5 pages. Page 4 has the least spare room (about 5 mm). If a page spills over, the PDF will have 6 pages.
4. **Create the repo**: on GitHub, make a new empty repo (public, so that Streamlit Community Cloud can read it and assessors can see it), then:
   ```powershell
   git remote add origin https://github.com/<you>/medinfo-assist.git
   git push -u origin main
   ```
5. **Deploy**: at share.streamlit.io, choose New app → your repo, branch `main`, main file `app.py`. In Advanced settings pick Python 3.12, and **leave Secrets empty** so it runs offline. Once it is up, try README enquiries #1, #5, #7 and #8, and check that the sidebar says "offline extractive".
6. Optionally add the live URL to README.md (and to the page 1 subtitle, then re-export), commit and push. Submit the PDF and the repo link before 17:30.
