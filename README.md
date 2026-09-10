# AI Agents for Retirement-Oriented Financial Guidance for Displaced Ukrainians in Poland

A multi-agent research pipeline for generating synthetic financial profiles modelled on displaced Ukrainians in Poland and producing retirement-oriented financial guidance grounded in Polish institutional sources.

The synthetic profile generator is informed primarily by the **2025 survey of Ukrainian migrants in Poland conducted by Narodowy Bank Polski and published in 2026**. The survey sample used as the principal calibration anchor contains **3,965 respondents**. Not all profile variables are survey-derived. Several variables and conditional sampling rules are researcher-defined modelling assumptions, as described below.

Dataset: [Uliana333/ukrainian-refugees-financial-advisory](https://huggingface.co/datasets/Uliana333/ukrainian-refugees-financial-advisory)

---

## Architecture

```text
RefugeeAgent  ->  ConsultantAgent (draft)  ->  RefugeeAgent (clarify)  ->  ConsultantAgent (final)  ->  EvaluatorAgent
   (profile)          (draft + Qs)                 (answers)                 (grounded advice)         (quality control)
```

Each advisory cycle is orchestrated with **LangGraph**. The main pipeline stores run records in `runs/runs.jsonl`.

The architecture separates three functions:

1. synthetic case construction
2. retrieval-grounded recommendation generation
3. post-generation evaluation and routing

The three-agent design is a functional decomposition used for traceability and controllability. It is not presented as an experimentally established optimal architecture.

---

## Agents

### RefugeeAgent

File: `agents/refugee.py`

The RefugeeAgent generates a structured synthetic client case.

Quantitative financial variables are generated algorithmically from sampled anchors and predefined budget rules. The LLM is used only for qualitative persona attributes, contextual constraints, retirement-oriented questions, and fallback clarification answers.

The agent uses:

- survey-derived demographic and labour-market anchors
- survey-anchored income bands combined with researcher-defined within-band sampling
- researcher-defined rules for dependents, duration of residence, savings, expenditure shares, and conditional Polish-language proficiency
- deterministic budget generation in PLN
- top-k retrieval from the refugee-context knowledge base for qualitative constraints only
- rule-based clarification templates with an LLM fallback

The RefugeeAgent does not generate financial advice.

### ConsultantAgent

File: `agents/consultant.py`

The ConsultantAgent is a two-stage retrieval-augmented advisory component.

`draft()` retrieves evidence and generates a preliminary response together with up to three clarifying questions.

`final()` incorporates the clarification answers, performs a fresh retrieval pass, and produces a structured recommendation.

The final response follows a fixed schema with:

- summary
- quick budget check
- suggested monthly retirement-saving amount
- retirement-related options in Poland
- next steps
- sources used

The ConsultantAgent combines three evidence channels:

- local FAISS vector retrieval
- curated live retrieval from trusted institutional domains
- optional Tavily search restricted to the same domain allow-list

Retrieved source objects use explicit identifiers so that cited evidence can be checked downstream.

The ConsultantAgent also applies deterministic safeguards before the recommendation is passed to the EvaluatorAgent. Budget values in the final quick-budget block are overwritten with values recomputed from the structured profile. Unsupported mentions of private pension vehicles such as PPK, IKE, IKZE, and OFE are removed when those terms are not supported by the retrieved evidence.

### EvaluatorAgent

File: `agents/evaluator.py`

The EvaluatorAgent is architecturally separate from the ConsultantAgent. It does not perform additional retrieval.

It receives:

- the final recommendation
- the structured client profile
- the source objects retrieved during the ConsultantAgent stage
- repair metadata produced during structured-output validation

Evaluation combines deterministic checks with an LLM-based rubric. The resulting scores are **model-based evaluation outcomes**. They have not been validated against financial professionals or target users.

---

## Models

| Agent | Model | Purpose |
|---|---|---|
| RefugeeAgent | `gpt-4o-mini` | Persona generation, qualitative constraints, user questions, clarification answers |
| ConsultantAgent | `gpt-4.1-mini` | Draft generation, clarifying questions, final structured guidance |
| EvaluatorAgent | `deepseek-chat` | Rubric-based post-generation assessment |

Default ConsultantAgent and EvaluatorAgent temperature is `0.1`.

The RefugeeAgent uses task-specific temperatures defined in `agents/refugee.py`.

---

## Retrieval configuration

The local retrieval layer uses:

| Parameter | Configuration |
|---|---|
| Embedding model | `text-embedding-3-small` |
| Vector store | Local FAISS |
| Similarity configuration | FAISS default L2 / Euclidean distance |
| Semantic chunking | Percentile breakpoint threshold = 95 |
| Post-semantic size guard | Maximum chunk size = 1,500 characters |
| Chunk overlap | 100 characters |
| Consultant retrieval | top-k = 5 |
| Refugee-context retrieval | top-k = 4 |
| Reranking | Not implemented |
| Retrieval-score threshold | Not implemented |

The size guard uses `RecursiveCharacterTextSplitter` after semantic chunking.

The ConsultantAgent can also use curated live institutional retrieval. By default, up to four live sources can be fetched per query. The configured domain allow-list includes:

- `gov.pl`
- `zus.pl`
- `podatki.gov.pl`
- `biznes.gov.pl`
- `euraxess.pl`
- `ec.europa.eu`

Tavily is disabled by default and is used only when explicitly enabled. When enabled, it is restricted to the same domain allow-list.

No explicit reranking, retrieval-score thresholding, source-deduplication, or contradiction-reconciliation stage is implemented.

---

## Evaluation

Evaluation is performed in two stages.

### Stage 1: deterministic checks

The deterministic layer checks:

| Check | Purpose |
|---|---|
| Arithmetic consistency | Compare reported budget values with values recomputed from `profile_json` |
| Citation presence | Detect missing citation support for Poland-specific claims |
| Citation validity | Detect citation identifiers that do not correspond to supplied source objects |
| Source trust | Detect live or open-web sources outside the configured allow-list |
| Structural completeness | Detect missing required sections |
| Private-program support | Detect unsupported mentions of configured private pension products |

Arithmetic consistency uses a tolerance of `± PLN 30`.

### Stage 2: LLM rubric

The EvaluatorAgent assigns model-based scores from 0 to 10 for:

| Dimension | Interpretation |
|---|---|
| Groundedness | Support for factual and Poland-specific claims |
| Arithmetic consistency | Consistency of reported numerical information |
| Actionability | Specificity and practical usefulness of next steps |
| Clarity | Accessibility and comprehensibility |
| Safety / ethics | Caution, appropriateness, and avoidance of overconfident guidance |

The arithmetic score should be interpreted together with the deterministic budget safeguard. Near-perfect arithmetic consistency primarily reflects engineered recomputation and validation rather than unaided numerical reasoning by the ConsultantAgent.

### Score aggregation and penalties

The base score is the rounded mean of the five rubric dimensions.

```text
base_score = round(mean(
    groundedness,
    arithmetic_consistency,
    actionability,
    clarity,
    safety_ethics
))

final_score = clip(base_score - deterministic_penalties, 0, 10)
```

Configured penalty weights are:

| Trigger | Penalty |
|---|---:|
| High-severity arithmetic error | -4 |
| Missing citation | -1 |
| Missing required section | -2 |
| Untrusted source | -1 |
| Unsupported private-program mention | -4 |
| High-severity unsupported claim | -2 |
| Medium-severity unsupported claim | -1 |

Multiple applicable penalties are combined additively.

### Case routing

Each case is assigned one of three operational routing outcomes:

| Label | Condition |
|---|---|
| `accepted` | `allow_to_show = True`, `grounded_only_pass = True`, and `overall_score >= 9` |
| `needs_review` | All cases that are neither accepted nor flagged |
| `flagged` | `allow_to_show = False` or `overall_score < 7` |

`allow_to_show` is a deterministic display gate. It becomes false when a high-severity issue prevents automatic display.

`grounded_only_pass` is a separate grounding gate. It becomes false when configured high-severity unsupported or arithmetic issues are present or when medium-or-higher citation, structural, or source-trust issues are detected.

An `accepted` case therefore means only that the case passed the predefined automated criteria. It does not mean professional approval or demonstrated real-world deployment safety.

---

## Experiments

Additional robustness and ablation analyses are available in the `experiments/` directory.

```text
experiments/
├── figures/
├── outputs/
│   ├── baseline_no_rag.jsonl
│   ├── repeated_runs.jsonl
│   ├── analysis_summary.json
│   └── analysis_tables.xlsx
├── analysis.py
├── repeated_runs.py
├── analysis.ipynb
├── test_analysis.py
└── README.md
```

The directory contains three main analysis components.

### Scoring and routing audit

`analysis.py` reconstructs composite scores and routing outcomes from stored evaluator outputs. It also summarizes issue severity, penalty application, subgroup comparisons, retrieval metadata, and threshold sensitivity.

### Paired no-RAG ablation

The no-RAG experiment uses fixed synthetic profiles and compares the same cases with RAG and without RAG.

The profile, query, clarification sequence, model configuration, deterministic arithmetic safeguard, EvaluatorAgent, scoring logic, and routing thresholds are held fixed.

In the condition without RAG:

- retrieval is disabled
- no external source objects are passed to the ConsultantAgent or EvaluatorAgent
- citation requirements that cannot be satisfied without retrieval are removed from the output contract
- the recommendation is generated using the model and supplied client profile only

This experiment compares the implemented with RAG and without RAG conditions. Because disabling retrieval also required accompanying changes to the Consultant prompt and citation contract, the comparison should not be interpreted as a pure retrieval-only causal intervention. It does not provide a single-agent or no-evaluator comparison.

### Repeated-run stability

The repeated-run experiment selects fixed original profiles and replays the final ConsultantAgent and EvaluatorAgent stages multiple times.

The fixed inputs are:

- structured profile
- textual profile
- user query
- clarification question-answer sequence

Consultant final generation, retrieval, EvaluatorAgent scoring, and routing are rerun. This measures end-to-end operational stability under fixed inputs rather than deterministic reproducibility under a frozen evidence set.

The default full experiment uses 100 profiles with 5 replays per profile and selection seed `42`.

---

## Scope and interpretation

This repository supports a computational research testbed for retirement-oriented financial guidance in the context of displaced Ukrainians in Poland.

The framework does **not** constitute a validated autonomous financial advisor.

The current evaluation does not establish:

- human-expert agreement with EvaluatorAgent scores
- real-world user benefit
- regulatory compliance
- fairness across protected groups
- privacy guarantees
- cybersecurity robustness
- longitudinal reliability
- generalization to other countries or displaced populations

The term **trustworthiness** is therefore used operationally and is limited to the dimensions explicitly measured in the testbed.

---

## Synthetic profile calibration

The principal empirical anchor is the **NBP survey conducted in 2025 and published in 2026**, based on **3,965 respondents**.

The generator intentionally distinguishes empirical anchors from modelling assumptions.

| Provenance category | Variables |
|---|---|
| Directly survey-derived | Employment status, base remittance rate |
| Survey-anchored with modelling assumptions | Gender, monthly-income bands and within-band sampling, Polish-language categories with researcher-defined conditional adjustments |
| Researcher-defined modelling assumptions | Number of dependents, duration of residence in Poland, savings-generation rules, expenditure shares and minimum floors, conditional remittance adjustments by employment status and household-support proxy |

The conditional Polish-language model is researcher-defined. It should not be interpreted as a direct reconstruction of the marginal language-proficiency distribution reported by NBP.

Monthly-income band boundaries are survey-anchored. Within-band sampling weights are modelling choices.

Generated profiles are synthetic scenarios informed by survey evidence. They are not population estimates.

---

## Budget simulation

The budget generator computes income, housing, utilities, food, transport, healthcare, other expenditure, remittances, savings, and monthly surplus algorithmically.

When generated expenditure exceeds the feasible level implied by the minimum-surplus rule, the algorithm first reduces miscellaneous spending toward its minimum floor. If this is insufficient, flexible components are proportionally scaled while core expenditure floors are preserved.

This feasibility-preserving procedure improves internal consistency but can make some synthetic households more solvent than comparable real-world households. Deficit rates and surplus distributions produced by the generator should therefore be interpreted as properties of the computational testbed rather than population estimates.

---

## Setup

### 1. Obtain API credentials

| Key | Purpose |
|---|---|
| `OPENAI_API_KEY` | RefugeeAgent and ConsultantAgent |
| `DEEPSEEK_API_KEY` | EvaluatorAgent |
| `TAVILY_API_KEY` | Optional open-web retrieval when Tavily is enabled |

### 2. Configure the environment

```bash
cp .env.example .env
```

Add the required API keys to `.env`.

### 3. Install dependencies

```bash
pip install poetry
poetry install
```

### 4. Add source documents

Place source files in:

```text
books_refuge/
books_consultant/
```

Supported local formats include PDF, HTML, and XLSX.

### 5. Run the main pipeline

```bash
poetry run python run_langgraph.py
```

To set a custom batch size:

```bash
N_CASES=50 poetry run python run_langgraph.py
```

Main pipeline results are written to:

```text
runs/runs.jsonl
```

---

## Docker

### Build

```bash
docker compose build
```

### Run

```bash
docker compose run --rm pipeline
```

Custom batch size:

```bash
N_CASES=50 docker compose run --rm pipeline
```

Large batch in the background:

```bash
N_CASES=500 docker compose run -d pipeline
```

View logs:

```bash
docker compose logs -f pipeline
```

Stop containers:

```bash
docker compose down
```

---

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | yes | | Used by RefugeeAgent and ConsultantAgent |
| `DEEPSEEK_API_KEY` | yes | | Used by EvaluatorAgent |
| `DEEPSEEK_BASE_URL` | no | `https://api.deepseek.com/v1` | DeepSeek endpoint |
| `N_CASES` | no | `5` | Cases generated per main run |
| `CONSULTANT_ENABLE_WEB` | no | `true` | Curated live-web retrieval |
| `CONSULTANT_ALLOWED_DOMAINS` | no | configured allow-list | Trusted institutional domains |
| `CONSULTANT_WEB_TIMEOUT_MS` | no | `20000` | Web-fetch timeout in milliseconds |
| `CONSULTANT_WEB_MAX_SOURCES` | no | `4` | Maximum curated live-web sources per retrieval call |
| `CONSULTANT_WEB_CACHE_DIR` | no | `books_consultant/live_web_cache` | Directory used for fetched page artefacts |
| `CONSULTANT_CURATED_URLS` | no | | Additional curated URLs |
| `CONSULTANT_ENABLE_TAVILY` | no | `false` | Optional Tavily retrieval |
| `TAVILY_API_KEY` | no | | Required only when Tavily is enabled |

---

## Main output format

Each line in `runs/runs.jsonl` represents one advisory cycle.

```text
run_id
ts_unix
case_status

refugee
  anchors
  profile_json
  persona_json
  profile_text
  user_query

consultant_draft
  draft_answer
  clarifying_questions

refugee_clarifying_qa

consultant_final
  final_answer
  final_answer_struct
  final_sources
  repair_issues

evaluator
  deterministic_checks
  llm_rubric
  flagged_spans
  final
```

The `final` evaluator block contains the operational score and routing gates used to derive `case_status`.

---

## Reproducibility

For the primary dataset, analysis scripts use the first 500 chronological records from `runs/runs.jsonl`.

Additional experiment outputs are stored separately under `experiments/outputs/` and do not overwrite the primary dataset.

The analysis and test scripts are designed to keep the original `runs/runs.jsonl` file read-only.

---

## Data and code availability

The generated advisory dataset is available on Hugging Face:

[Uliana333/ukrainian-refugees-financial-advisory](https://huggingface.co/datasets/Uliana333/ukrainian-refugees-financial-advisory)

The repository contains the source code, retrieval configuration, analysis scripts, ablation experiment, repeated-run experiment, and reproducibility checks used in the study.
