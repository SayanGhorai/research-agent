ResearchLens AI
An evidence-grounded research assistant powered by IBM Granite and watsonx.ai.

ResearchLens AI helps students and researchers find literature, read uploaded papers, organize references, and prepare research drafts. It combines document-scoped retrieval with IBM Granite generation and displays the supporting passages, PDF page references, and validation warnings alongside the output.

Developed by Sayan Ghorai, VIT Bhopal University, for the IBM SkillsBuild / AICTE internship through Edunet Foundation.

Selected problem statement: No. 1 — Research Agent
Domain: Education, academic research, and scientific literature assistance

GitHub Repository · Hosted Application · [Project Presentation](ResearchLens_AI_Sayan_Ghorai_Project_Presentation.pptx) · [Problem Statement](yourproblemstatement.pdf)
Research prototype: Generated answers and references may contain mistakes. Citation checks do not establish scientific truth. Review the original sources before using an output in academic work. Local Granite generation has been demonstrated; hosted AI generation requires separate private credential configuration and was not confirmed in the recorded deployment checks.

Project overview
Research work involves repeated literature searches, reading, note-taking, comparison, and reference management. Important details can be missed, and a result from a cited study can be mistaken for the current paper's own finding.

ResearchLens AI provides one workspace for these activities. Users can upload research PDFs or discover papers through arXiv and Crossref, select the sources relevant to their question, and request an answer or research draft. The application retrieves selected passages rather than asking the language model to answer without source material.

The project supports different research topics. Plant-disease research papers were used as demonstration documents; the application itself is not a plant-disease classifier.
Verified project status
Submission snapshot: 5 November 2026. These statuses reflect the recorded local tests and deployment evidence, not a guarantee of current external-service availability.
Area Recorded status
Python and Streamlit application Implemented and run locally
PDF ingestion, selected-paper retrieval, and reference exports Tested with real research PDFs
IBM Granite through watsonx.ai Live local generation demonstrated
Questions, comparisons, extraction, summaries, and reports Reviewed live; outputs included partial validation and known errors
Hypotheses and section drafting Implemented and tested with mocked responses; live quality review not completed
Existing regression suite 224 tests passed in the recorded local run
Additional UI helper checks 24 separate offline checks passed
GitHub source and submission artifacts Pushed to main
Streamlit Community Cloud Hosted interface deployed; cloud Granite generation not independently confirmed

The initial source commit was b48c64d. Submission files were added in commit 47cf8cb.
Research capabilities
Capability What the application provides
Literature discovery arXiv and Crossref search, metadata retrieval, and duplicate handling
Research-question answering Answers based on passages from the selected sources
Paper summarization Organized summaries of objectives, methods, samples, findings, metrics, and limitations
Structured extraction Six research fields with separate draft and review-status handling
Paper comparison Side-by-side comparison of two or three selected papers
Research reports Draft briefs with supporting evidence and references
Hypothesis generation Suggestions intended to be labelled untested, with rationale and proposed evaluation
Research-section drafting Draft text grounded in available evidence, with unsupported details left for review
Reference management Editable bibliographic metadata and JSON, CSV, and BibTeX exports

These are implemented routes, not claims that every generated answer is complete or correct. In particular, live tests found citation mismatches, omitted information, unsupported interpretations, and occasional incorrect reference details.
Architecture and agent roles
Sources → PDF extraction / metadata ingestion → page-aware passages → scoped retrieval → IBM Granite draft → deterministic checks → review and export

The application uses four bounded Python roles rather than four separate language models:
Role Responsibility
Search Agent Builds a literature-search query and calls the scholarly-source adapters
Read / Extract Agent Assembles relevant evidence from eligible papers and prepares extraction information
Write Agent Sends the bounded evidence and task instructions to IBM Granite
Verify Agent Checks source identifiers, citation scope, numerical consistency, and selected output requirements

The workflow is user-triggered. It does not run an open-ended agent conversation or repeatedly regenerate answers until they appear correct. The demonstrated configuration permits one model request per action and does not add an automatic answer-repair call.
Retrieval configuration
The tested demonstration uses TF-IDF lexical retrieval. Evidence selection is restricted to the chosen paper IDs, and retrieved passages retain their document identity and PDF page location.

Optional MiniLM semantic retrieval and reciprocal-rank fusion are supported in the code, but semantic retrieval is disabled in the demonstrated configuration. sentence-transformers is not a default runtime dependency. Installing it, loading the embedding model, and validating semantic retrieval would be a separate step.

ENABLE_SEMANTIC_SEARCH=false does not disable Granite. It controls evidence retrieval; ALLOW_LIVE_GENERATION separately controls model generation.
Technology stack
Component Implementation
Application Python 3.12 and Streamlit
Language model ibm/granite-4-h-small through IBM watsonx.ai
IBM integration ibm-watsonx-ai Python SDK
PDF extraction pypdf with fonttools support
Lexical retrieval scikit-learn and NumPy
Typed records Pydantic
Scholarly-source adapters Requests and feedparser
Private configuration Environment variables, Streamlit Secrets, and python-dotenv
Development and delivery IBM Bob assistance, local testing, Git, GitHub, and Streamlit Community Cloud

IBM Bob assisted development and troubleshooting. IBM Granite is the application's runtime LLM. Subsequent implementation checks and testing were performed on the Python project.
Interface and exports
The interface has five tabs: Research Workspace, Paper Library, Compare & Extract, Reports & References, and About & Diagnostics.

Users can choose a task, restrict evidence to selected papers, inspect source passages, edit bibliography details, and download research artifacts. Available exports include library JSON/CSV/BibTeX, draft Markdown, an evidence ledger, a run trace, and structured-extraction CSV outputs.

The updated interface displays generated content before detailed validation warnings. Blocked attempts are distinguished from previous saved results. In structured extraction, generated values that do not pass the checks remain available as Needs review rather than being silently treated as information absent from the paper. A review worksheet preserves draft values separately from accepted fields.
Run locally
Use Python 3.12. On Windows, the following commands create a project-specific environment and install the runtime dependencies:
git clone https://github.com/SayanGhorai/research-agent.git
cd research-agent
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy .env.example to a private .env file and configure it using the exact variable names below. For preview use, leave live generation disabled. For authorized generation, supply the IBM settings from your own working watsonx.ai project and enable the generation flag.

Start the application:
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
On macOS or Linux, use the corresponding virtual-environment Python executable. No local Granite model weights or GPU are required for the demonstrated TF-IDF and hosted-generation setup.
Configuration
The application resolves explicit environment variables first, followed by Streamlit Secrets, local .env values, and defaults. Restart the application after configuration changes so cached settings and workflow objects are recreated.
Variable Purpose / demonstrated setting
WATSONX_API_KEY Private IBM API key
WATSONX_PROJECT_ID Project associated with the IBM service
WATSONX_URL Regional endpoint; the local demo used https://us-south.ml.cloud.ibm.com
WATSONX_MODEL_ID ibm/granite-4-h-small for the recorded demonstration
ALLOW_LIVE_GENERATION false for preview; true for explicitly enabled generation
ENABLE_SEMANTIC_SEARCH false for the tested lightweight configuration
APP_ENV local on the developer's computer; production for the hosted application
APP_ACCESS_PASSWORD Private application access password; required by the production live-generation guard
MAX_LLM_CALLS_PER_ACTION Set to 1 for the controlled demonstration
MAX_OUTPUT_TOKENS Set to 1000 for the final local demonstration
CROSSREF_MAILTO Optional contact for Crossref requests

Use WATSONX*API_KEY, not IBM_WATSONX_API_KEY; the same naming distinction applies to the other WATSONX* settings.

A configured status means required values are present. It does not prove successful IBM authentication, model availability, or remaining inference allowance.
Usage and access safeguards
The demonstrated upload limits are 3 PDFs per session, 10 MB per PDF, and 40 pages per PDF. Source scope and page references remain attached to the retrieved text.

The UI records model attempts and applies a five-call browser-session guard. Clearing the library does not reset this counter. It is not an account-wide token budget, billing limit, or persistent global rate limit; another browser session can have a separate counter.

MAX_OUTPUT_TOKENS limits response length, not input tokens or overall account consumption. PDF parsing and TF-IDF retrieval do not require Granite inference. Online literature discovery uses the relevant providers, while generated answers, comparisons, and reports use IBM inference when enabled.

Keep .env, .streamlit/secrets.toml, access passwords, and uploaded private documents out of Git. Session data is not a durable research archive: export results before closing or restarting. Clearing local session content does not erase provider-side records or account usage.
Testing and evaluation
The recorded local verification completed 224 regression tests. A separate set of 24 offline UI helper checks covered warning placement, stale-result handling, extraction review states, upload-limit configuration, and preservation of the session call guard.

The regression suite covers configuration, schemas, PDF ingestion, duplicate uploads, retrieval, document scope, workflow behaviour, the IBM client, evidence checks, and exports. Additional offline scenarios replayed observed answer-quality problems and exercised the seven generation routes using mocked model responses.

To install development dependencies and run the existing suite:
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
The recorded offline runs used controlled test settings and no live inference. A previously inherited one-call setting affected a test expecting the default three-call budget; use isolated test settings when reproducing results rather than changing application code to satisfy that mismatch.

Passing software tests does not establish research-answer accuracy. The separate UI checks are not automatically included in the 224-test count. A GitHub Actions workflow is provided in [`.github/workflows/ci.yml`](.github/workflows/ci.yml); inspect the repository's Actions results for hosted CI status.
Retrieval evaluation
[`eval/run_eval.py`](eval/run_eval.py) provides an offline evaluation path using an illustrative corpus. Manual relevance labels and keyword-proxy scores are treated separately. Historical values in [`eval/results.json`](eval/results.json) should not be presented as a general research benchmark or proof of model accuracy.

The recorded evaluation does not establish semantic-retrieval performance, reliable model abstention, or scientific correctness across arbitrary PDFs.
Measured token demonstration
A local paper-summary demonstration was accompanied by screenshots of the IBM project's cumulative resource usage:
Measurement Tokens
Before 46,074
After 48,550
Observed increase 2,476

These are project-dashboard readings, not remaining quota. The difference is the observed increase during the demonstration interval; reporting delays or other activity can affect attribution. The screenshots are included in the presentation.
Deployment
The hosted interface was deployed from this repository's main branch with app.py as the entry point and Python 3.12. Streamlit Cloud configuration is separate from the developer's local .env file.

Enter private credentials through the hosting platform's Secrets settings, not the public repository. For live generation, the application requires complete IBM configuration, ALLOW_LIVE_GENERATION=true, and satisfaction of its access controls. Production live generation is blocked without a configured application password.

The last recorded cloud check established that the interface loads. The successful live Granite demonstrations documented here ran locally. Do not assume that the hosted link has working generation solely because the homepage is available. Localhost-only launch flags are for local testing, not cloud configuration.
Repository and submission files
File / directory Purpose
app.py Streamlit interface, session management, and user actions
researchlens/ Configuration, source adapters, ingestion, retrieval, agents, workflows, schemas, and exports
tests/ Automated regression tests
eval/ Offline retrieval evaluation code and stored results
requirements.txt Runtime dependencies
requirements-dev.txt Development and test dependencies
.github/workflows/ci.yml Automated test workflow
yourproblemstatement.pdf Project-specific Research Agent problem statement
ResearchLens_AI_Sayan_Ghorai_Project_Presentation.pptx Submission presentation and demonstration evidence
app.json Descriptive project-information manifest
THIRD_PARTY_NOTICES.md Dependency and third-party acknowledgements

app.json is documentation, not executable application code or an imported visual-workflow export. The application runs from app.py and does not depend on the manifest for execution.

[`PROJECT_STATUS.md`](PROJECT_STATUS.md) contains the earlier development log. Some phase entries and test counts may predate the submission snapshot in this README.
Known limitations

- Grounding and completeness: A generated draft can omit requested facts, attach the wrong passage citation, mix a paper's own findings with related work, or claim that available information is missing.
- Numerical interpretation: Decimal-to-percentage conversions, derived values, and training/testing distinctions can produce warnings or require manual review. A warning does not always mean the reported number is false.
- Document coverage: Scanned pages, tables, images, and inconsistent source text may not be fully represented by text extraction. A full-text label does not mean every detail reached the model.
- References: Uploaded filenames may initially serve as titles. Authors, year, DOI, and generated bibliography entries require checking; metadata is not guaranteed to be extracted correctly.
- Comparisons and deployment: Different datasets and evaluation protocols do not justify direct performance rankings. Per-session safeguards are not enterprise access controls or account-wide spending limits.
  The validation states valid, partial, invalid, and unchecked describe the implemented checks. Even valid does not mean scientifically verified. Hypotheses remain proposals, and generated sections remain drafts.
  Future scope
  Future work includes better per-claim citation alignment, table and bibliography extraction, broader multi-domain evaluation, optional semantic retrieval assessment, and persistent usage controls. Further live evaluation of hypotheses and section drafting is also needed.

These are improvement opportunities, not features claimed as complete in the current submission.
Author and acknowledgements
Sayan Ghorai — VIT Bhopal University

Developed as part of the IBM SkillsBuild / AICTE internship through Edunet Foundation, with IBM Bob development assistance and IBM Granite generation through watsonx.ai.

Demonstration research papers belong to their respective authors and publishers. Their reported model accuracies are findings from those papers, not accuracy scores for ResearchLens AI.
License and third-party notices
A project-specific license has not yet been selected. Do not assume an open-source license from the repository's public visibility alone. Third-party components and research documents remain subject to their respective terms.

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for acknowledgements.
