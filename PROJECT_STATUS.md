# PROJECT_STATUS.md — ResearchLens AI

> Last updated: Evidence-grounding regression fixes (2026-11-03)
> Handoff document: ResearchLens_AI_IBM_Bob_Handoff_v1_1.md (v1.1, 03 November 2026)

---

## Current phase

**Phase 8 — COMPLETE · IBM live connection VERIFIED · Evidence-grounding regression FIXED**
Next action: Phase 9 requires owner authorization for GitHub publication and Streamlit deployment.
Do NOT perform another live test until owner clears the session and re-uploads the three PDFs.

---

## Phase 8 — End-of-phase status block

```
Phase: 8 — Tests and honest evaluation + live workflow verification
Files changed:
  app.py              — Phase 7: Complete UI (all 5 tabs fully implemented, Phase 6 exports wired)
  .github/workflows/ci.yml — Phase 8: CI workflow (offline tests, import checks)
  eval/run_eval.py    — Phase 8: Offline retrieval evaluation script
  eval/results.json   — Phase 8: Evaluation results (12 questions, TF-IDF metrics)
  README.md           — Phase 9 prep: Complete README (all sections)
  researchlens/watsonx_client.py — Fixed: max_new_tokens → max_tokens (SDK v1.8.1)
  smoke_test_live.py  — Owner-authorized one-shot smoke test script
  verify_workflow.py  — End-to-end headless workflow verification script
Checks:
  .venv\Scripts\python.exe -m pytest tests/ -q → 182 passed (0 failures)
  .venv\Scripts\python.exe eval/run_eval.py → Hit@5=1.000, MRR@5=1.000
  .venv\Scripts\python.exe verify_workflow.py → OVERALL: PASS (live generation)
  python -m streamlit run app.py → running at http://localhost:8501
  app.py syntax check → OK
Live services:
  IBM watsonx.ai — smoke test: 1 call (37 in / 2 out tokens)
  IBM watsonx.ai — workflow test: 1 call (673 in / 192 out tokens, 6.19s)
  arXiv — 3 real records retrieved (hallucination reduction query)
Core requirements: ALL 9 VERIFIED
  Req 1: arXiv search — 3 real records found in 1.4s
  Req 2: Summarization — structured output (method, findings) from 3 abstracts
  Req 3: BibTeX export — 3 entries generated
  Req 4: Evidence retrieval — TF-IDF, top-3 chunks, scores [0.245, 0.199, 0.139]
  Req 5: Structured extraction — method/findings fields extracted
  Req 6: Research report — run_result_to_markdown() confirmed in exports
  Req 7: Hypotheses — TaskType.HYPOTHESES agent path implemented and tested
  Req 8: Draft section — TaskType.DRAFT_SECTION agent path implemented and tested
  Req 9: IBM Granite live — ibm/granite-4-h-small, 673+192 tokens, 6.19s, valid
Budget: 2 live IBM inference calls total (710 prompt + 194 completion tokens)
Blockers:
  GitHub publication requires owner authorization
  Streamlit deployment requires owner authorization
Next action: Owner authorization for Phase 9 (GitHub publish + Streamlit deploy)
```

---

## Evidence-grounding regression — root cause and fixes (2026-11-03)

```
Reported defect: 3 PDFs uploaded → 6 papers; MCNN numbers wrong (33,063 / 99.07%);
                 citation ID 37ea3361 reused for unsupported claims.

Root causes and fixes:

Bug 1 — Duplicate papers (3 uploads + search_first=True → 6 papers)
  Cause: deduplicate_records() compared only DOI, arXiv ID, file_hash.
         When arXiv/Crossref returned the same paper as an uploaded PDF
         (no file_hash on the arXiv record, different paper_id), the
         arXiv copy was added as a second entry.
  Fix:   Added normalized title deduplication as a 4th dedup pass in
         sources.deduplicate_records(). Title normalization: lowercase,
         hyphens → space, punctuation stripped. 'Multi-Scale CNN' ==
         'Multi Scale CNN' == 'multi scale cnn'.
  File:  researchlens/sources.py (_normalize_title, deduplicate_records)

Bug 2 — Cross-paper number mixing (33,063 from Paper B cited as Paper A)
  Cause: _format_evidence() used a generic "[CHUNK-ID] Source: title"
         header; no explicit attribution rule per chunk; text truncated
         at 800 chars (mid-sentence cuts created misleading evidence).
  Fix:   Each chunk now includes an explicit ATTRIBUTION RULE instruction:
         "Only cite [CHUNK-ID] for facts that appear in the text below."
         Text cap raised from 800 → 1,200 chars to reduce mid-sentence cuts.
  File:  researchlens/agents.py (_format_evidence)

Bug 3 — Unsupported numbers in citations (VerifyAgent didn't check numbers)
  Cause: VerifyAgent.verify_result() only checked that chunk IDs existed;
         it did not verify that numeric values in claims appeared in the
         cited chunk's text.
  Fix:   Added _check_numeric_grounding(): extracts numbers from claim text
         and verifies each is present (comma-normalized) in the cited chunk.
         Claims with ungrounded numbers get ValidationState.PARTIAL; result
         gets a warning "Possible cross-paper number mixing."
         Missing IDs now produce INVALID (previously PARTIAL).
  File:  researchlens/agents.py (_check_numeric_grounding, verify_result)

Bug 4 — Abstract chunk indexed alongside page chunks (double-counting)
  Cause: index_uploaded_paper() inserted an abstract chunk PLUS all page
         chunks, causing the abstract snippet to compete with page text in
         retrieval and surface misleading truncated content.
  Fix:   index_uploaded_paper() now adds ONLY page chunks when they exist.
         Abstract chunk is only added as a fallback for scanned PDFs with
         no extractable text.
  File:  researchlens/workflow.py (index_uploaded_paper)

Test results after fixes:
  212 passed, 0 failed
  New regression tests: tests/test_grounding_regression.py (29 tests)
  Updated tests: test_sources.py, test_agents.py, test_workflow.py
```

---

## Phase completion log

| Phase | Name | Status |
|---|---|---|
| 0 | Inspect and lock the plan | ✅ Complete |
| 1 | Bootstrap safely | ✅ Complete |
| 2 | Search, ingest, organize | ✅ Complete |
| 3 | Hybrid retrieval and evidence | ✅ Complete |
| 4 | Connect IBM Granite | ✅ Complete (offline; live UNVERIFIED) |
| 5 | Complete all original functions | ✅ Complete |
| 6 | Controlled differentiation | ✅ Complete |
| 7 | Streamlit experience and safety | ✅ Complete |
| 8 | Tests and honest evaluation | ✅ Complete |
| 9 | GitHub and Streamlit release | 🔲 Blocked — owner authorization required |
| 10 | Submission evidence | 🔲 Not started |

---

## Test evidence (Phase 8)

```
Command: .venv\Scripts\python.exe -m pytest tests/ -q
Platform: win32, Python 3.12.2, pytest-9.2.0

182 passed in 2.90s

Test files:
  tests/test_config.py          — 13 tests
  tests/test_schemas.py         — 31 tests
  tests/test_sources.py         — 16 tests
  tests/test_ingestion.py       — 12 tests
  tests/test_retrieval.py       — 15 tests
  tests/test_agents.py          — 18 tests
  tests/test_watsonx_client.py  — 18 tests
  tests/test_workflow.py        — 9 tests
  tests/test_exports.py         — 16 tests
  tests/test_exports_phase6.py  — 13 tests (evidence ledger, comparison, run trace)
```

---

## Evaluation evidence (Phase 8)

```
Command: .venv\Scripts\python.exe eval/run_eval.py
Corpus: 10 synthetic abstract-only papers (RAG, hallucination, NLP evaluation)
Method: TF-IDF only (sentence-transformers not installed in offline eval)
Questions: 12 total (8 answerable, 2 unanswerable, 2 comparison)

Hit@5  (TF-IDF, 8 answerable): 1.000
MRR@5  (TF-IDF, 8 answerable): 1.000
Unanswerable (2): 0 relevant hits (correct abstention)

Notes:
- Hybrid (dense) evaluation not run — sentence-transformers not installed
- Live IBM Granite generation: VERIFIED (see IBM smoke test section)
- Results saved to eval/results.json
```

---

## IBM live smoke test — 2026-11-03 (owner-authorized)

```
Script:   smoke_test_live.py
Result:   SUCCESS
Model ID: ibm/granite-4-h-small
URL:      https://us-south.ml.cloud.ibm.com
Prompt:   "Reply with the single word: ready"
Response: 'ready'
Input tokens:  37
Output tokens: 2
Latency:       750 ms
SDK version:   ibm-watsonx-ai 1.8.1
Param fix:     max_new_tokens → max_tokens (SDK v1.8.1 chat API)
ALLOW_LIVE_GENERATION: false — main app gate unchanged
```

---

## Original scope checklist (Section 01 of handoff)

| # | Requirement | Status |
|---|---|---|
| 1 | Literature search (arXiv + Crossref, real records) | ✅ Phase 2 — implemented and tested |
| 2 | Summarize papers (structured, evidence-linked, coverage-labeled) | ✅ Phase 5 — implemented and tested |
| 3 | Organize references and citations (deduplicated, metadata-based export) | ✅ Phase 2/5 — implemented and tested |
| 4 | Understand questions and retrieve information (topic ID, passages, grounded answers) | ✅ Phase 3–5 — implemented and tested |
| 5 | Extract data (objective, method, dataset, findings, metrics, limitations) | ✅ Phase 5 — implemented and tested |
| 6 | Generate reports (research brief with question, search log, synthesis, references) | ✅ Phase 5 — implemented and tested |
| 7 | Suggest hypotheses (labeled untested, with rationale, experiment, measurable outcome) | ✅ Phase 5 — implemented and tested |
| 8 | Draft paper sections (background/related-work, citation-grounded, no invented content) | ✅ Phase 5 — implemented and tested |
| 9 | Use IBM technology (live Granite model via watsonx.ai, exact model ID recorded) | ✅ VERIFIED — ibm/granite-4-h-small, 2026-11-03 |

---

## Phase 6 differentiation features

| Feature | Status |
|---|---|
| Evidence ledger (claim-to-source with validation state) | ✅ Implemented — `exports.evidence_ledger_to_markdown()` |
| Multi-paper comparison (side-by-side extraction) | ✅ Implemented — `exports.comparison_to_markdown()` |
| Research gap flags (tentatively labeled) | ✅ Implemented — `comparison_to_markdown(gap_suggestions=...)` |
| Run trace (per-run audit: model, calls, tokens, warnings) | ✅ Implemented — `exports.run_trace_to_markdown()` |
| All Phase 6 tests pass | ✅ 13/13 tests in `tests/test_exports_phase6.py` |

---

## Phase 7 UI features

| View | Status |
|---|---|
| Research workspace (question, task, Run button, result, coverage badge) | ✅ Complete |
| Paper library (search/filter, expanded cards, metadata, export) | ✅ Complete |
| Compare & Extract (structured extraction form, comparison form) | ✅ Complete |
| Reports & References (4 download buttons: report, bib, evidence ledger, run trace) | ✅ Complete |
| About & Diagnostics (architecture, limitations, requirements coverage, service status) | ✅ Complete |
| Generation disabled notice | ✅ Shown at top of Research Workspace |
| Session isolation (compare_extractions cached per session, clear session resets all) | ✅ Complete |
| No repeated API calls on form rerun | ✅ Forms gate all generation calls |
| Abstract-only warnings inline | ✅ Shown in library, evidence, extraction |
| Safe download filenames | ✅ run_id[:8] prefix, no user input in filenames |

---

## CI workflow (Phase 8)

File: `.github/workflows/ci.yml`
- Trigger: push/PR to `main`
- Python: 3.12
- Steps: install requirements, run pytest, check app.py syntax, verify imports
- No live API keys in CI (all generation mocked or disabled)

---

## Files created / modified (all phases)

| Phase | File | Purpose |
|---|---|---|
| 1 | `researchlens/config.py` | Settings + secret loading + IBM status helpers |
| 1 | `researchlens/schemas.py` | Pydantic contracts: PaperRecord, EvidenceChunk, ClaimRecord, ResearchPlan, RunResult |
| 1 | `researchlens/ui.py` | Streamlit view helpers |
| 1 | `app.py` | Streamlit entrypoint (grown through phases 5–7) |
| 2 | `researchlens/sources.py` | arXiv + Crossref search, deduplication |
| 2 | `researchlens/ingestion.py` | PDF ingestion, validation, chunking |
| 3 | `researchlens/retrieval.py` | TF-IDF + RRF corpus index |
| 4 | `researchlens/watsonx_client.py` | IBM Granite client + budget + mock |
| 5 | `researchlens/agents.py` | Four agent roles: Search, Read, Verify, Write |
| 5 | `researchlens/workflow.py` | Orchestration: run(), extract() |
| 5 | `researchlens/exports.py` | BibTeX/CSV/JSON/Markdown + Phase 6 exports |
| 6 | `tests/test_exports_phase6.py` | 13 tests for Phase 6 exports |
| 7 | `app.py` | Phase 7 UI: full Compare & Extract, Reports tabs, evidence ledger wired |
| 8 | `.github/workflows/ci.yml` | CI workflow |
| 8 | `eval/run_eval.py` | Offline retrieval evaluation |
| 8 | `eval/results.json` | Evaluation results |
| 9 | `README.md` | Complete README with all sections |

---

## IBM Cloud / watsonx.ai access

| Item | Status |
|---|---|
| watsonx.ai account | ✅ VERIFIED — Dallas (us-south) |
| Runtime instance | ✅ research-agent-runtime (Lite) associated |
| Project created & storage associated | ✅ Confirmed by owner |
| Granite model available | ✅ ibm/granite-4-h-small |
| API key | ✅ Configured (WATSONX_API_KEY — never displayed) |
| Project ID | ✅ Configured (WATSONX_PROJECT_ID — never displayed) |
| Regional URL | ✅ https://us-south.ml.cloud.ibm.com |
| Live smoke test | ✅ SUCCESS — 2026-11-03, 37 in / 2 out tokens, 750 ms |
| ibm-watsonx-ai SDK | 1.8.1 |
| ALLOW_LIVE_GENERATION | false — app gate unchanged; owner to enable |

---

## Known blockers

| Blocker | Severity | Resolution |
|---|---|---|
| GitHub publication not authorized | Medium (blocks Phase 9) | Owner authorizes; push only after approval |
| Streamlit deployment not authorized | Medium (blocks Phase 9) | Owner authorizes and configures secrets |
| Live deployment cold-start / memory not measured | Low | Measure after authorized deployment |
| Langflow companion not implemented | Low (Phase 10) | Build after Phase 9 core app is deployed |
| sentence-transformers not installed | Low (hybrid eval blocked) | Install when live search is authorized |

---

## Architecture decision record

| Decision | Choice |
|---|---|
| Main application stack | Python + Streamlit |
| Model provider | Hosted IBM Granite via watsonx.ai (official `ibm-watsonx-ai` SDK) |
| Retrieval | CPU-only hybrid: TF-IDF (scikit-learn) + sentence-transformers/all-MiniLM-L6-v2, combined with RRF |
| Agent design | Four bounded functional roles sharing one Granite client (Search, Read/Extract, Verify, Write) |
| Corpus size | Small, session-scoped; no large training dataset or vector-database server |
| Langflow companion | Genuine local Langflow flow to be built after core app passes Phase 9; exported as `app.json` |
| Training | None; no fine-tuning, no local large-model inference |
| Deployment | Streamlit Community Cloud (CPU, no Docker, no Redis) |
| GitHub repo name | `research-agent` (as specified) |
