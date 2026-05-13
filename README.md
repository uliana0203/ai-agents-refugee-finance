# AI Agents - Pension Advice for Ukrainian Refugees in Poland

A multi-agent pipeline that generates synthetic financial profiles of Ukrainian refugees in Poland and produces retirement savings advice grounded in Polish pension law.

Based on **NBP 2025 survey data** (n = 3,965 respondents, April-June 2025).

Dataset: [Uliana333/ukrainian-refugees-financial-advisory](https://huggingface.co/datasets/Uliana333/ukrainian-refugees-financial-advisory)

---

## Architecture

```
RefugeeAgent  ->  ConsultantAgent (draft)  ->  RefugeeAgent (clarify)  ->  ConsultantAgent (final)  ->  EvaluatorAgent
   (profile)          (draft + Qs)                 (answers)                 (grounded advice)         (quality score)
```

Each iteration is orchestrated by **LangGraph** and the result is appended to `runs/runs.jsonl`.

### Agents

**RefugeeAgent** - `agents/refugee.py`
Generates a synthetic refugee profile. Demographic anchors (gender, employment, income band, language level, months in Poland) are drawn from NBP 2025 statistical distributions. Budget figures are computed algorithmically in PLN with hard expense floors. An LLM generates qualitative persona details (age, education, retirement goals). Clarifying answers follow a rule-based path for common questions; an LLM fallback handles the rest.

**ConsultantAgent** - `agents/consultant.py`
Two-pass RAG advisor. `draft()` retrieves sources and proposes up to 3 clarifying questions. `final()` uses the Q&A together with a fresh retrieval pass to write a structured recommendation. The final answer is enforced via a Pydantic schema (`_FinalAnswerStruct`) using function calling; a JSON-mode LLM is used as fallback. If required sections are still missing after generation, `_ensure_required_final_sections()` fills them deterministically and records each repair in `repair_issues`. Sources come from three channels: a local FAISS vectorstore (PDF/HTML/XLSX documents), live web pages (gov.pl, zus.pl, euraxess.pl, fetched via MCP), and optional Tavily search. Web sources are numbered from S100, Tavily sources from S200. Private pension vehicles (PPK/IKE/IKZE/OFE) are automatically stripped unless they appear in the retrieved sources.

**EvaluatorAgent** - `agents/evaluator.py`
Two-stage quality check described in detail in the [Evaluation](#evaluation) section.

---

## Models

| Agent | Model | Purpose |
|---|---|---|
| RefugeeAgent | `gpt-4o-mini` | Persona, constraints, clarifying answers |
| ConsultantAgent | `gpt-4.1-mini` | Draft, clarifying questions, final structured advice |
| EvaluatorAgent | `deepseek-chat` | Rubric scoring (temperature 0.1) |

**Approximate cost per 1,000 cases:**

| Model | ~Input tokens | ~Output tokens | Cost per 1k cases |
|---|---|---|---|
| gpt-4o-mini | 1,200 | 400 | ~$0.30 |
| gpt-4.1-mini | 3,500 | 900 | ~$2.20 |
| deepseek-chat | 4,000 | 600 | ~$0.70 |
| **Total** | | | **~$3.20** |

---

## Evaluation

Answer quality is assessed in two sequential stages. The result of both stages is stored under the `evaluator` key of each run record.

### Stage 1 - Deterministic checks

These checks run without an LLM call and produce hard flags:

| Check | What it detects | Effect on final verdict |
|---|---|---|
| **Arithmetic consistency** | Compares income, expenses, surplus stated in the answer against `profile_json` ground truth (tolerance +-PLN 30) | Mismatch -> `math_error` issue (severity: high) |
| **Citation presence** | Counts `[S#]` / `[W#]` tags; flags if Poland-specific regulatory claims appear without any citation | -> `missing_citation` issue (severity: medium) |
| **Private-program hallucination** | Checks whether PPK / IKE / IKZE appear in the answer but not in the retrieved sources | -> `unsupported_claim` issue (severity: high) |
| **Unknown citations** | Detects citation IDs not present in `final_sources` | -> `unsupported_claim` issue (severity: high) |

### Stage 2 - LLM rubric (DeepSeek)

Five dimensions are scored 0-10. Each score must be backed by a direct quote from the answer text.

| Dimension | What is evaluated | Deductions |
|---|---|---|
| **Groundedness** | Are Poland-specific claims (ZUS, pension age, benefit amounts) supported by `[S#]` citations? | -2 per unsupported factual claim |
| **Arithmetic consistency** | Does the answer correctly restate income, expenses and surplus from the profile? | -4 if wrong numbers; -2 if omitted |
| **Actionability** | Are concrete next steps given that match this specific profile? For deficit profiles: quality of benefit referrals (Rodzina 800+, MOPS), not saving amount. | Assessed contextually |
| **Clarity** | Is the answer accessible to a person with basic financial literacy? | -2 per unexplained jargon |
| **Safety / ethics** | Are legal entitlements stated with appropriate caveats? | -3 if specific pension amounts are presented as guaranteed facts |

**Exemptions** - the following are never penalised as unsupported claims:
- Rodzina 800+ benefit (PLN 800/child/month) when `dependents > 0`
- MOPS/GOPS social assistance when profile shows unemployment + low income
- ZUS contribution gap when profile shows unemployed/economically_inactive
- Any statement explicitly labelled "(general rule of thumb, not Poland-specific)"

### Score aggregation and penalties

```
base_score  = mean(groundedness, arithmetic, actionability, clarity, safety)
final_score = base_score - penalties_from_stage_1
```

Penalty weights (configurable in `EvaluatorConfig`):

| Issue type | Severity | Penalty |
|---|---|---|
| `math_error` | high | -4 |
| `private_program_hallucination` | high | -4 |
| `unsupported_claim` | high | -2 |
| `unsupported_claim` | medium | -1 |
| `missing_citation` | medium | -1 |

Final score is clamped to `[0, 10]`.

### Case routing

Each case is routed to one of three outcomes and labelled in `case_status`:

| Label | Condition |
|---|---|
| `accepted` | `allow_to_show = True` AND `grounded_only_pass = True` AND `overall_score >= 9` |
| `needs_review` | `allow_to_show = True` but score 7-8, or grounding check failed |
| `flagged` | `allow_to_show = False` OR `overall_score < 7` |

`grounded_only_pass` is `False` if any `unsupported_claim` or `math_error` with severity `high` was detected.

---

## Setup

### 1. Obtain API credentials

| Key | Where to get it |
|---|---|
| `OPENAI_API_KEY` | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| `DEEPSEEK_API_KEY` | [platform.deepseek.com](https://platform.deepseek.com) -> API Keys |
| `TAVILY_API_KEY` | [app.tavily.com](https://app.tavily.com) (optional; only needed if `CONSULTANT_ENABLE_TAVILY=true`) |

### 2. Clone and configure environment

```bash
cp .env.example .env
# Fill in OPENAI_API_KEY and DEEPSEEK_API_KEY (and optionally TAVILY_API_KEY)
```

### 3. Install dependencies

```bash
pip install poetry
poetry install
```

### 4. Add source documents

Place files in:
- `books_refuge/` - NBP reports, surveys, `.xlsx` tables about Ukrainian refugees in Poland
- `books_consultant/` - Polish pension law, ZUS guides, MISSOC documents

Supported formats: `.pdf`, `.html`, `.xlsx`

### 5. Run locally

```bash
# Default: 5 cases
poetry run python run_langgraph.py

# Custom batch size
N_CASES=50 poetry run python run_langgraph.py
```

Results are saved to `runs/runs.jsonl`.

---

## Docker

### Prerequisites

```bash
cp .env.example .env   # fill in API keys
```

### Build

```bash
docker compose build
```

### Run pipeline

```bash
# Generate 5 cases (default)
docker compose run --rm pipeline

# Custom batch size
N_CASES=50 docker compose run --rm pipeline

# Large batch in background
N_CASES=500 docker compose run -d pipeline
```

### View logs of a running container

```bash
docker compose logs -f pipeline
```

### Stop

```bash
docker compose down
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | yes | | Used by RefugeeAgent and ConsultantAgent |
| `DEEPSEEK_API_KEY` | yes | | Used by EvaluatorAgent |
| `DEEPSEEK_BASE_URL` | no | `https://api.deepseek.com/v1` | DeepSeek endpoint |
| `N_CASES` | no | `5` | Cases to generate per run |
| `CONSULTANT_ENABLE_WEB` | no | `true` | Live web retrieval on/off |
| `CONSULTANT_ALLOWED_DOMAINS` | no | `gov.pl,zus.pl,...` | Comma-separated domain allowlist |
| `CONSULTANT_WEB_TIMEOUT_MS` | no | `20000` | Web fetch timeout (ms) |
| `CONSULTANT_WEB_MAX_SOURCES` | no | `4` | Max live web sources per query |
| `CONSULTANT_WEB_CACHE_DIR` | no | `books_consultant/live_web_cache` | Cache directory for fetched pages |
| `CONSULTANT_CURATED_URLS` | no | | Extra URLs always included (comma-separated) |
| `CONSULTANT_ENABLE_TAVILY` | no | `false` | Tavily search on/off |
| `TAVILY_API_KEY` | no | | Required if Tavily is enabled |

---

## Profile Variables (NBP 2025)

| Variable | Distribution | Source |
|---|---|---|
| `gender` | female 66%, male 34% | NBP 2025 |
| `employment_status` | permanent_job 54%, other_work 17%, unemployed 14%, economically_inactive 11%, self_employed 4% | NBP 2025 |
| `polish_language_level` | none 5%, basic 39%, intermediate 42%, advanced 14% | NBP 2025 Fig. 11 |
| `income` | PLN 800-8,500 by gender x employment band | NBP 2025 Fig. 22 |
| `months_in_poland` | 6-48 months | |
| `remittances_flag` | 36% send remittances | NBP 2025 |

---

## Analysis

`eda_runs.ipynb` contains the full exploratory analysis of the generated dataset. It produces all figures and tables reported in the paper: profile distributions (gender, employment, language, dependents), budget distributions, evaluator score statistics, issue frequency, score breakdown by employment status and budget position, and Pearson correlations between profile variables and rubric scores.

---

## Output Format

Each line of `runs/runs.jsonl` is a JSON object:

```
run_id                  UUID
ts_unix                 UNIX timestamp
case_status             accepted | needs_review | flagged
refugee
  anchors               gender, employment_status, income_min/max, months, dependents, ...
  profile_json          income, savings, housing, food, transport, ... (PLN)
  persona_json          age, education, financial_literacy, retirement_goals, ...
  profile_text          human-readable text sent to the consultant
  user_query            generated retirement question
consultant_draft
  draft_answer          first-pass answer
  clarifying_questions  up to 3 questions
refugee_clarifying_qa   list of {question, answer}
consultant_final
  final_answer          numbered-section text
  final_answer_struct   structured JSON (summary, quick_budget_check, next_steps, ...)
  final_sources         sources actually cited in the answer
  repair_issues         list of sections filled deterministically when LLM output was incomplete
evaluator
  deterministic_checks  citation_count, arithmetic mismatches, private_programs_flag, ...
  llm_rubric            scores per dimension + issues list
  flagged_spans         age claims and private program mentions
  final                 overall_score (0-10), allow_to_show, grounded_only_pass
```
