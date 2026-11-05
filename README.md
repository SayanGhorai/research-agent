# ResearchLens AI

**Evidence-Grounded Research Agent** — IBM Granite-powered literature search, paper summarization, multi-paper comparison, citation management, and hypothesis generation.

---

## Status

| Phase | Name | Status |
|---|---|---|
| 0 | Inspect and lock the plan | ✅ Complete |
| 1 | Bootstrap safely | ✅ Complete |
| 2 | Search, ingest, organize | ✅ Complete |
| 3 | Hybrid retrieval and evidence | ✅ Complete |
| 4 | Connect IBM Granite | ✅ Complete (offline SDK; **live smoke test UNVERIFIED** — IBM credentials required) |
| 5 | Complete all original functions | ✅ Complete |
| 6 | Controlled differentiation | ✅ Complete (evidence ledger, comparison, run trace) |
| 7 | Streamlit experience and safety | ✅ Complete |
| 8 | Tests and honest evaluation | ✅ Complete (182 tests pass; offline retrieval evaluated) |
| 9 | GitHub and Streamlit release | 🔲 Pending owner authorization |
| 10 | Submission evidence | 🔲 Pending |

See [`PROJECT_STATUS.md`](PROJECT_STATUS.md) for the full phase log.

---

## Original scope — all nine requirements

| # | Requirement | Status |
|---|---|---|
| 1 | Literature search (arXiv + Crossref, real records) | ✅ Implemented |
| 2 | Summarize papers (structured, evidence-linked, coverage-labeled) | ✅ Implemented |
| 3 | Organize references and citations (deduplicated, metadata-based export) | ✅ Implemented |
| 4 | Understand questions and retrieve information | ✅ Implemented |
| 5 | Extract data (objective, method, dataset, findings, metrics, limitations) | ✅ Implemented |
| 6 | Generate reports (research brief, search log, synthesis, references) | ✅ Implemented |
| 7 | Suggest hypotheses (labeled untested, rationale, experiment, outcome) | ✅ Implemented |
| 8 | Draft paper sections (background/related-work, citation-grounded) | ✅ Implemented |
| 9 | Use IBM technology (live Granite via watsonx.ai) | ⚠️ **UNVERIFIED** — IBM credentials required |

---

## Features

- Natural-language research question input with explicit **Run** button
- arXiv + Crossref literature search (live, deduplicated)
- PDF upload with page-indexed text extraction and size/safety validation
- TF-IDF + sentence-transformer hybrid retrieval (CPU-only, lazy-loaded)
- Reciprocal-rank fusion (RRF, k=60) combining lexical and semantic signals
- Evidence-grounded answers, summaries, structured extraction
- Multi-paper side-by-side comparison with tentative gap/disagreement flags
- **Evidence ledger** — per-claim source linkage with validation state
- **Run trace** — per-run audit trail (model, calls, tokens, warnings)
- Research report, hypothesis, and paper-section generation via IBM Granite
- Session library with BibTeX / CSV / JSON export
- Four agent roles: Search · Read/Extract · Verify · Write
- Session-isolated corpus (no persistent disk; clear session removes all state)
- Live generation disabled by default — explicit gate required

---

## Quick start (local)

```bash
# 1. Clone and enter the project
git clone https://github.com/<owner>/research-agent.git
cd research-agent

# 2. Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure credentials (optional — app runs without them in preview mode)
cp .env.example .env
# Edit .env — add your watsonx.ai API key, project ID, URL, and model ID

# 5. Run the app
python -m streamlit run app.py
```

**Test command (verified):**
```bash
python -m pytest -q
```

**Launch command (verified):**
```bash
python -m streamlit run app.py
```

---

## Environment variables

See [`.env.example`](.env.example) for the full list. Key variables:

| Variable | Purpose |
|---|---|
| `IBM_WATSONX_API_KEY` | IBM watsonx.ai API key (owner-supplied; never commit) |
| `IBM_WATSONX_PROJECT_ID` | watsonx.ai project ID |
| `IBM_WATSONX_URL` | Regional URL (e.g. `https://us-south.ml.cloud.ibm.com`) |
| `IBM_WATSONX_MODEL_ID` | Exact model ID from the accessible catalog |
| `ALLOW_LIVE_GENERATION` | Set `true` only after a successful smoke test |

Never commit a populated `.env` or `secrets.toml`.

---

## Architecture

| Component | Choice |
|---|---|
| Main app | Python 3.12 + Streamlit |
| Model | Hosted IBM Granite via watsonx.ai (`ibm-watsonx-ai` SDK) |
| Keyword retrieval | TF-IDF (scikit-learn) |
| Semantic retrieval | sentence-transformers/all-MiniLM-L6-v2 (CPU, lazy-loaded) |
| Fusion | Reciprocal-rank fusion (RRF, k=60) |
| Agent roles | Search · Read/Extract · Verify · Write |
| Session corpus | Up to 1,500 chunks; 3 PDFs, 40 pages each |
| LLM budget | 3 calls per action; 5 per session |
| Langflow companion | Planned — see Phase 10 |

---

## Tests and evaluation

- **182 tests, 0 failures** (offline, no live API keys in CI)
- Test coverage: config/secrets, ingestion, retrieval, grounding, workflow, exports (including Phase 6)
- Offline retrieval evaluation: Hit@5 = 1.000, MRR@5 = 1.000 (10-paper synthetic corpus, TF-IDF only)
- Evaluation results: [`eval/results.json`](eval/results.json)
- Live IBM Granite generation: **UNVERIFIED** — IBM credentials not supplied

See [`eval/run_eval.py`](eval/run_eval.py) to re-run the offline evaluation.

---

## CI

GitHub Actions workflow: [`.github/workflows/ci.yml`](.github/workflows/ci.yml)
Runs on push/PR to `main`. Uses Python 3.12. No live API keys in CI. Mocks all providers.

---

## Deployment (Streamlit Community Cloud)

1. Push to GitHub (after owner approves public publication).
2. Create a new app pointing to `app.py`, Python 3.12, branch `main`.
3. Add credentials in **App settings → Secrets** using valid TOML (see [`.streamlit/secrets.example.toml`](.streamlit/secrets.example.toml)).
4. Set `ALLOW_LIVE_GENERATION = "true"` in secrets only after a live smoke test.

> Note: Session state is in-memory only. Uploaded documents are not persisted to disk between restarts.

---

## Limitations

- Abstract-only sources are labelled; detailed synthesis requires uploaded full text.
- Hypotheses and research gap suggestions are explicitly untested.
- Embeddings evaluated on a small session corpus only — not a scientific benchmark.
- IBM Granite access requires the owner's watsonx.ai entitlement (see above).
- Session state is not persisted across restarts — export results before closing.
- Comparison disagreements and gaps are tentative — not confirmed literature findings.
- Live deployment: community Cloud resource limits may require reduced chunk caps.
- Langflow companion: planned, not yet implemented.

---

## Data and model notices

- arXiv search uses the public Atom feed API (no account required; rate-limited).
- Crossref search uses the public REST API (no account required; rate-limited).
- sentence-transformers model `all-MiniLM-L6-v2`: Apache 2.0 license; downloaded on first semantic search.
- IBM Granite model: accessed via watsonx.ai SDK; no local weights; owner's account required.
- No user data is stored persistently; session content is cleared on restart or explicit Clear Session.

---

## License

*To be selected by the owner.*

---

## Third-party notices

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for dependency attributions.
