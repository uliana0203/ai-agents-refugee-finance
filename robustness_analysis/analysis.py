"""
experiments/analysis.py
==================================

Robustness and ablation analytics for "Toward Trustworthy AI Financial Advisors".

This module ONLY reads existing artefacts. It never calls an LLM API and never
writes to runs/. All original-dataset analyses use the FIRST 500 records of
runs/runs.jsonl in chronological order (by ts_unix); those 500 reproduce the
manuscript counts (327 accepted / 170 needs_review / 3 flagged). The 501st
appended record is excluded.

Sub-commands
------------
    python experiments/analysis.py all
        Run every no-API analysis (TASK E, F-audit, G, H, I, J, K, L, M) and
        write experiments/outputs/analysis_tables.xlsx +
        experiments/outputs/analysis_summary.json + no-API figures.

    python experiments/analysis.py analyze-repeated
        Summarise experiments/outputs/repeated_runs.jsonl (TASK B).

    python experiments/analysis.py analyze-baseline
        Summarise experiments/outputs/baseline_no_rag.jsonl (TASK D).

Every sub-command merges its sheets into the single workbook analysis_tables.xlsx
(existing sheets are replaced, others kept).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_PATH = REPO_ROOT / "runs" / "runs.jsonl"
OUT_DIR = REPO_ROOT / "experiments" / "outputs"
FIG_DIR = OUT_DIR / "figures"
XLSX_PATH = OUT_DIR / "analysis_tables.xlsx"
SUMMARY_PATH = OUT_DIR / "analysis_summary.json"
REPEATED_JSONL = OUT_DIR / "repeated_runs.jsonl"
BASELINE_JSONL = OUT_DIR / "baseline_no_rag.jsonl"

N_ORIGINAL = 500
RUBRIC_DIMS = ["groundedness", "arithmetic_consistency", "actionability", "clarity", "safety_ethics"]
EXPENSE_KEYS = ["housing", "utilities", "food", "transport", "healthcare", "other", "remittances"]

# Penalty weights exactly as in agents/evaluator.py::EvaluatorConfig (documented, not changed).
PENALTY_WEIGHTS = {
    "math_error_high": 4,
    "missing_citation": 1,
    "missing_required_section": 2,
    "untrusted_source": 1,
    "private_program_hallucination": 4,
    "unsupported_claim_high": 2,
    "unsupported_claim_medium": 1,
}


# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------

def load_all_records(path: Path = RUNS_PATH) -> List[Dict[str, Any]]:
    """Load every JSON line from runs.jsonl (read-only)."""
    recs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def load_original_500(path: Path = RUNS_PATH) -> List[Dict[str, Any]]:
    """Return the first N_ORIGINAL records in chronological order (ts_unix, then file order)."""
    recs = load_all_records(path)
    order = sorted(range(len(recs)), key=lambda i: (recs[i].get("ts_unix", 0), i))
    chosen = [recs[i] for i in order[:N_ORIGINAL]]
    if len(chosen) < N_ORIGINAL:
        print(f"[warn] only {len(chosen)} records available (< {N_ORIGINAL})")
    return chosen


def _issues_of(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    iss = (((rec.get("evaluator") or {}).get("llm_rubric") or {}).get("issues")) or []
    return [it for it in iss if isinstance(it, dict)]


def flatten(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten one run record into a flat analysis row."""
    ref = rec.get("refugee", {})
    prof = ref.get("profile_json", {})
    anch = ref.get("anchors", {})
    ev = rec.get("evaluator", {})
    final = ev.get("final", {})
    rub = ev.get("llm_rubric", {})
    scores = rub.get("scores", {})
    det = ev.get("deterministic_checks", {})
    cfinal = rec.get("consultant_final", {})

    income = float(prof.get("income", 0.0) or 0.0)
    expenses = sum(float(prof.get(k, 0.0) or 0.0) for k in EXPENSE_KEYS)
    surplus = income - expenses

    issues = _issues_of(rec)
    itypes = [it.get("type", "unknown") for it in issues]
    isevs = [it.get("severity", "unknown") for it in issues]

    row: Dict[str, Any] = {
        "run_id": rec.get("run_id"),
        "ts_unix": rec.get("ts_unix"),
        "case_status": rec.get("case_status"),
        # profile / anchors
        "gender": anch.get("gender", prof.get("gender")),
        "employment_status": anch.get("employment_status", prof.get("employment_status")),
        "months": anch.get("months", prof.get("months")),
        "dependents": int(prof.get("dependents", anch.get("dependents", 0)) or 0),
        "polish_language_level": anch.get("polish_language_level", prof.get("polish_language_level")),
        "remittances_flag": int(anch.get("remittances_flag", 0) or 0),
        "income": income,
        "savings": float(prof.get("savings", 0.0) or 0.0),
        "expenses_total": expenses,
        "surplus": surplus,
        "is_deficit": surplus <= 0,
        "has_dependents": int(prof.get("dependents", 0) or 0) > 0,
        # evaluator scalar outcomes
        "overall_score": final.get("overall_score"),
        "allow_to_show": bool(final.get("allow_to_show", False)),
        "grounded_only_pass": bool(final.get("grounded_only_pass", False)),
        # rubric dims
        **{f"score_{d}": scores.get(d) for d in RUBRIC_DIMS},
        # issues
        "n_issues": len(issues),
        "issue_types": itypes,
        "issue_sevs": isevs,
        "has_missing_citation": "missing_citation" in itypes,
        "has_unsupported_claim": "unsupported_claim" in itypes,
        "has_untrusted_source": "untrusted_source" in itypes,
        "has_missing_required_section": "missing_required_section" in itypes,
        "has_unsafe_overconfidence": "unsafe_overconfidence" in itypes,
        "has_math_error": "math_error" in itypes,
        # deterministic checks
        "det_missing_citations_flag": bool(det.get("missing_citations_flag", False)),
        "det_private_programs_flag": bool(det.get("private_programs_flag", False)),
        "det_unknown_citations": bool(det.get("unknown_citations")),
        "det_untrusted_web_sources": bool(det.get("untrusted_web_sources")),
        "det_missing_required_sections": bool(det.get("missing_required_sections")),
        "det_expenses_mismatch": bool(det.get("expenses_total_mismatch")),
        "det_surplus_mismatch": bool(det.get("surplus_mismatch")),
        "det_arith_mismatch": bool(det.get("expenses_total_mismatch") or det.get("surplus_mismatch")),
        "n_repair_issues": len(det.get("repair_issues") or []) or len(cfinal.get("repair_issues") or []),
        "citation_count": det.get("citation_count"),
        "inline_citation_count": det.get("inline_citation_count"),
        # source channel mix (final answer)
        **_source_channel_counts(cfinal.get("final_sources") or []),
    }
    return row


def _source_channel_counts(sources: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    c = Counter(str(s.get("source_type", "unknown")) for s in sources)
    return {
        "n_src_total": sum(c.values()),
        "n_src_vectorstore": c.get("vectorstore", 0),
        "n_src_live_web": c.get("live_web", 0),
        "n_src_tavily": c.get("tavily_search", 0),
    }


def frame_original(path: Path = RUNS_PATH) -> pd.DataFrame:
    return pd.DataFrame([flatten(r) for r in load_original_500(path)])


# --------------------------------------------------------------------------------------
# Pure helpers (imported by test_analysis.py)
# --------------------------------------------------------------------------------------

SEV_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}
SEV_NAME = {v: k for k, v in SEV_RANK.items()}


def normalise_severity(sev: Optional[str]) -> str:
    s = str(sev or "").strip().lower()
    return s if s in SEV_RANK else "low"


def case_max_severity(severities: Sequence[Optional[str]]) -> str:
    """Return the highest severity label among a case's issues ('none' if empty)."""
    if not severities:
        return "none"
    return SEV_NAME[max(SEV_RANK[normalise_severity(s)] for s in severities)]


def severity_counts(severities: Sequence[Optional[str]]) -> Dict[str, int]:
    """Count issue-level severities into the fixed buckets low/medium/high."""
    out = {"low": 0, "medium": 0, "high": 0}
    for s in severities:
        out[normalise_severity(s)] = out.get(normalise_severity(s), 0) + 1
    return out


def recompute_routing(overall_score: Optional[int], allow_to_show: bool,
                      grounded_only_pass: bool, accept_threshold: int = 9,
                      flag_below: int = 7) -> str:
    """Replicate run_langgraph.route logic with configurable thresholds (TASK G)."""
    s = -1 if overall_score is None else int(overall_score)
    if (not allow_to_show) or s < flag_below:
        return "flagged"
    if allow_to_show and grounded_only_pass and s >= accept_threshold:
        return "accepted"
    return "needs_review"


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 1.0


# --------------------------------------------------------------------------------------
# Stats helpers (scipy optional -> graceful degradation)
# --------------------------------------------------------------------------------------

try:
    from scipy import stats as _sps  # type: ignore
    SCIPY = True
except Exception:  # pragma: no cover
    _sps = None
    SCIPY = False


def mean_ci(x: Sequence[float], conf: float = 0.95) -> Tuple[float, float, float, float, int]:
    """Return (mean, sd, ci_low, ci_high, n) using a t-interval; degrades to +-1.96*se without scipy."""
    a = np.asarray([v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=float)
    n = int(a.size)
    if n == 0:
        return (math.nan, math.nan, math.nan, math.nan, 0)
    m = float(a.mean())
    sd = float(a.std(ddof=1)) if n > 1 else 0.0
    if n < 2 or sd == 0.0:
        return (m, sd, m, m, n)
    se = sd / math.sqrt(n)
    if SCIPY:
        crit = float(_sps.t.ppf(0.5 + conf / 2.0, n - 1))
    else:
        crit = 1.959963985
    return (m, sd, m - crit * se, m + crit * se, n)


def hedges_g(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Cohen's d (pooled SD) and small-sample bias-corrected Hedges' g for two groups."""
    x = np.asarray([v for v in a if v is not None], dtype=float)
    y = np.asarray([v for v in b if v is not None], dtype=float)
    n1, n2 = x.size, y.size
    if n1 < 2 or n2 < 2:
        return {"cohens_d": math.nan, "hedges_g": math.nan, "n1": n1, "n2": n2}
    s1, s2 = x.var(ddof=1), y.var(ddof=1)
    sp = math.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)) if (n1 + n2 - 2) > 0 else 0.0
    if sp == 0.0:
        return {"cohens_d": 0.0, "hedges_g": 0.0, "n1": n1, "n2": n2}
    d = (x.mean() - y.mean()) / sp
    J = 1.0 - (3.0 / (4.0 * (n1 + n2) - 9.0))
    return {"cohens_d": float(d), "hedges_g": float(d * J), "n1": int(n1), "n2": int(n2)}


def two_group_tests(a: Sequence[float], b: Sequence[float]) -> Dict[str, Any]:
    """Welch t-test + Mann-Whitney U for two independent groups."""
    x = np.asarray([v for v in a if v is not None], dtype=float)
    y = np.asarray([v for v in b if v is not None], dtype=float)
    out: Dict[str, Any] = {"n1": int(x.size), "n2": int(y.size),
                           "welch_t": math.nan, "welch_p": math.nan,
                           "mwu_U": math.nan, "mwu_p": math.nan, "note": ""}
    if x.size < 2 or y.size < 2:
        out["note"] = "n too small"
        return out
    if x.std() == 0 and y.std() == 0 and x.mean() == y.mean():
        out["note"] = "both groups constant & equal; tests uninformative"
        return out
    if not SCIPY:
        out["note"] = "scipy not installed; descriptive only"
        return out
    if np.concatenate([x, y]).std() < 0.05:
        out["note"] = "near-constant outcome; parametric p-values unreliable"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            t, p = _sps.ttest_ind(x, y, equal_var=False)
            out["welch_t"], out["welch_p"] = float(t), float(p)
        except Exception as e:  # pragma: no cover
            out["note"] += f" welch_err={e!r}"
        try:
            U, p = _sps.mannwhitneyu(x, y, alternative="two-sided")
            out["mwu_U"], out["mwu_p"] = float(U), float(p)
        except Exception as e:  # pragma: no cover
            out["note"] += f" mwu_err={e!r}"
    return out


def multi_group_tests(groups: Dict[str, Sequence[float]]) -> Dict[str, Any]:
    """One-way ANOVA + Kruskal-Wallis with eta^2 / epsilon^2 effect sizes for k>=2 groups."""
    arrays = {k: np.asarray([v for v in vs if v is not None], dtype=float) for k, vs in groups.items()}
    arrays = {k: v for k, v in arrays.items() if v.size >= 2}
    out: Dict[str, Any] = {"k_groups": len(arrays), "group_sizes": {k: int(v.size) for k, v in arrays.items()},
                           "anova_F": math.nan, "anova_p": math.nan, "eta_squared": math.nan,
                           "kruskal_H": math.nan, "kruskal_p": math.nan, "epsilon_squared": math.nan, "note": ""}
    if len(arrays) < 2:
        out["note"] = "fewer than 2 usable groups"
        return out
    allv = np.concatenate(list(arrays.values()))
    if allv.std() == 0:
        out["note"] = "outcome constant across all groups; tests uninformative"
        return out
    # eta^2 (classic between/total SS) computed directly so it works without scipy
    grand = allv.mean()
    ss_total = float(((allv - grand) ** 2).sum())
    ss_between = float(sum(v.size * (v.mean() - grand) ** 2 for v in arrays.values()))
    out["eta_squared"] = ss_between / ss_total if ss_total > 0 else math.nan
    if not SCIPY:
        out["note"] = "scipy not installed; ANOVA/KW p-values omitted"
        return out
    if allv.std() < 0.05:
        out["note"] = "near-constant outcome; parametric p-values unreliable"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            F, p = _sps.f_oneway(*arrays.values())
            out["anova_F"], out["anova_p"] = float(F), float(p)
        except Exception as e:  # pragma: no cover
            out["note"] += f" anova_err={e!r}"
        try:
            H, p = _sps.kruskal(*arrays.values())
            n = int(allv.size)
            k = len(arrays)
            out["kruskal_H"], out["kruskal_p"] = float(H), float(p)
            out["epsilon_squared"] = float((H - k + 1) / (n - k)) if n > k else math.nan
        except Exception as e:  # pragma: no cover
            out["note"] += f" kruskal_err={e!r}"
    return out


def icc21(matrix: np.ndarray) -> Dict[str, float]:
    """ICC(2,1): two-way random effects, absolute agreement, single measurement.

    matrix shape = (n_subjects, k_raters). Here subjects = fixed profiles,
    raters = repeat slots. NaN rows/cols are dropped listwise.
    """
    m = np.asarray(matrix, dtype=float)
    m = m[~np.isnan(m).any(axis=1)]
    n, k = m.shape
    if n < 2 or k < 2:
        return {"icc": math.nan, "n_subjects": int(n), "k_raters": int(k), "note": "need n>=2,k>=2"}
    grand = m.mean()
    ss_total = ((m - grand) ** 2).sum()
    ms_rows = k * ((m.mean(axis=1) - grand) ** 2).sum() / (n - 1)          # between subjects
    ms_cols = n * ((m.mean(axis=0) - grand) ** 2).sum() / (k - 1)          # between raters
    ss_err = ss_total - (k * ((m.mean(axis=1) - grand) ** 2).sum()) - (n * ((m.mean(axis=0) - grand) ** 2).sum())
    ms_err = ss_err / ((n - 1) * (k - 1))
    denom = ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n
    icc = (ms_rows - ms_err) / denom if denom != 0 else math.nan
    return {"icc": float(icc), "n_subjects": int(n), "k_raters": int(k),
            "ms_between_subjects": float(ms_rows), "ms_between_raters": float(ms_cols),
            "ms_error": float(ms_err), "note": ""}


def fleiss_kappa(rows: Sequence[Sequence[str]], categories: Sequence[str]) -> Dict[str, float]:
    """Fleiss' kappa over a list of per-subject rating lists (each length = n_raters)."""
    cat_idx = {c: i for i, c in enumerate(categories)}
    counts = []
    n_raters = None
    for r in rows:
        if n_raters is None:
            n_raters = len(r)
        if len(r) != n_raters:
            continue
        row = [0] * len(categories)
        for v in r:
            if v in cat_idx:
                row[cat_idx[v]] += 1
        counts.append(row)
    if not counts or not n_raters or n_raters < 2:
        return {"fleiss_kappa": math.nan, "n_subjects": len(counts), "n_raters": n_raters or 0,
                "note": "need >=2 raters"}
    mat = np.asarray(counts, dtype=float)
    N = mat.shape[0]
    n = n_raters
    p_j = mat.sum(axis=0) / (N * n)
    P_i = ((mat ** 2).sum(axis=1) - n) / (n * (n - 1))
    P_bar = P_i.mean()
    P_e = float((p_j ** 2).sum())
    kappa = (P_bar - P_e) / (1 - P_e) if (1 - P_e) != 0 else math.nan
    return {"fleiss_kappa": float(kappa), "n_subjects": int(N), "n_raters": int(n),
            "P_bar": float(P_bar), "P_e": P_e, "note": ""}


# --------------------------------------------------------------------------------------
# Excel writer (merge into single workbook)
# --------------------------------------------------------------------------------------

def write_sheets(sheets: Dict[str, pd.DataFrame]) -> None:
    """Write/replace the given sheets in analysis_tables.xlsx, keeping other sheets intact."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mode = "a" if XLSX_PATH.exists() else "w"
    kwargs: Dict[str, Any] = {"engine": "openpyxl", "mode": mode}
    if mode == "a":
        kwargs["if_sheet_exists"] = "replace"
    with pd.ExcelWriter(XLSX_PATH, **kwargs) as xw:
        for name, df in sheets.items():
            safe = name[:31]
            (df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)).to_excel(xw, sheet_name=safe, index=False)
    print(f"[xlsx] wrote {len(sheets)} sheet(s) -> {XLSX_PATH.relative_to(REPO_ROOT)}")


def update_summary(patch: Dict[str, Any]) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    if SUMMARY_PATH.exists():
        try:
            data = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data.update(patch)
    SUMMARY_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[json] updated {SUMMARY_PATH.relative_to(REPO_ROOT)}")
    return data


# --------------------------------------------------------------------------------------
# TASK: dataset overview + routing
# --------------------------------------------------------------------------------------

def analysis_overview(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    n = len(df)
    vc = df["case_status"].value_counts()
    overview = pd.DataFrame({
        "Metric": ["Total analysed cases", "Accepted", "Needs review", "Flagged",
                   "Cases with rubric scores", "Excluded appended records (record 501)"],
        "Value": [n, int(vc.get("accepted", 0)), int(vc.get("needs_review", 0)),
                  int(vc.get("flagged", 0)), int(df["overall_score"].notna().sum()),
                  len(load_all_records()) - n],
    })
    routing = pd.DataFrame({
        "case_status": vc.index,
        "n_cases": vc.values,
        "pct": (vc.values / n * 100).round(1),
    })
    score_dist = (df["overall_score"].value_counts().sort_index()
                  .rename_axis("overall_score").reset_index(name="n_cases"))
    return {"Dataset_Overview": overview, "Routing": routing, "Routing_Score_Dist": score_dist}


# --------------------------------------------------------------------------------------
# TASK E: issue severity x routing
# --------------------------------------------------------------------------------------

def analysis_issue_severity(df: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    routes = ["accepted", "needs_review", "flagged"]

    # 1. issue-level severity x routing  (counts ISSUES, not cases)
    il_rows = []
    for sev in ["low", "medium", "high"]:
        row = {"severity": sev}
        tot = 0
        for rt in routes:
            c = 0
            for _, r in df[df.case_status == rt].iterrows():
                c += sum(1 for s in r["issue_sevs"] if normalise_severity(s) == sev)
            row[rt] = c
            tot += c
        row["total_issues"] = tot
        il_rows.append(row)
    issue_level = pd.DataFrame(il_rows)

    # 2. case-level maximum severity x routing  (counts CASES)
    df = df.copy()
    df["max_sev"] = df["issue_sevs"].apply(case_max_severity)
    ml_rows = []
    for sev in ["none", "low", "medium", "high"]:
        sub = df[df.max_sev == sev]
        row = {"max_severity": sev}
        for rt in routes:
            row[rt] = int((sub.case_status == rt).sum())
        row["total_cases"] = int(len(sub))
        ml_rows.append(row)
    max_sev = pd.DataFrame(ml_rows)

    # 3. source-support issue types x routing  (cases containing >=1 of that type)
    ss_rows = []
    for t in ["unsupported_claim", "missing_citation", "untrusted_source"]:
        row = {"issue_type": t}
        n_cases = 0
        n_issues = 0
        for rt in routes:
            sub = df[df.case_status == rt]
            cc = int(sub["issue_types"].apply(lambda ts: t in ts).sum())
            ic = int(sub["issue_types"].apply(lambda ts: sum(1 for x in ts if x == t)).sum())
            row[f"{rt}_cases"] = cc
            n_cases += cc
            n_issues += ic
        row["total_cases_with_type"] = n_cases
        row["total_issues_of_type"] = n_issues
        ss_rows.append(row)
    source_support = pd.DataFrame(ss_rows)

    # 4. issue type x routing (issue counts + case counts)
    all_types = sorted({t for ts in df["issue_types"] for t in ts})
    t_rows = []
    for t in all_types:
        row = {"issue_type": t}
        for rt in routes:
            sub = df[df.case_status == rt]
            row[f"{rt}_issues"] = int(sub["issue_types"].apply(lambda ts: sum(1 for x in ts if x == t)).sum())
            row[f"{rt}_cases"] = int(sub["issue_types"].apply(lambda ts: t in ts).sum())
        row["total_issues"] = int(df["issue_types"].apply(lambda ts: sum(1 for x in ts if x == t)).sum())
        row["total_cases"] = int(df["issue_types"].apply(lambda ts: t in ts).sum())
        t_rows.append(row)
    issue_type_routing = pd.DataFrame(t_rows)

    # 5. issue type x severity (issue counts)
    ts_counter: Counter = Counter()
    for _, r in df.iterrows():
        for t, s in zip(r["issue_types"], r["issue_sevs"]):
            ts_counter[(t, normalise_severity(s))] += 1
    ts_rows = []
    for t in all_types:
        row = {"issue_type": t}
        for sev in ["low", "medium", "high"]:
            row[sev] = ts_counter.get((t, sev), 0)
        row["total"] = sum(row[s] for s in ["low", "medium", "high"])
        ts_rows.append(row)
    issue_type_severity = pd.DataFrame(ts_rows)

    n_cases_with_issue = int((df["n_issues"] > 0).sum())
    summary = {
        "issue_severity": {
            "n_cases": int(len(df)),
            "n_cases_with_at_least_one_issue": n_cases_with_issue,
            "n_issues_total": int(df["n_issues"].sum()),
            "issue_level_severity_totals": {s: int(issue_level.loc[issue_level.severity == s, "total_issues"].iloc[0])
                                            for s in ["low", "medium", "high"]},
            "case_max_severity_distribution": df["max_sev"].value_counts().to_dict(),
            "note": ("No high-severity issue occurs in the 500-case dataset; the high-severity "
                     "routing path in Table 3 is defined but was never triggered."),
        }
    }
    sheets = {
        "Issue_Severity": issue_level,
        "Max_Severity": max_sev,
        "Source_Support": source_support,
        "Issue_Type_Routing": issue_type_routing,
        "Issue_Type_Severity": issue_type_severity,
    }
    return sheets, summary


# --------------------------------------------------------------------------------------
# TASK F: penalty audit recomputed over the stored 500
# --------------------------------------------------------------------------------------

def _penalty_for_issue(it: Dict[str, Any], private_flag: bool) -> Tuple[str, str, int]:
    t = it.get("type")
    s = it.get("severity")
    if t == "math_error" and s == "high":
        return t, s, PENALTY_WEIGHTS["math_error_high"]
    if t == "missing_citation":
        return t, s, PENALTY_WEIGHTS["missing_citation"]
    if t == "missing_required_section":
        return t, s, PENALTY_WEIGHTS["missing_required_section"]
    if t == "untrusted_source":
        return t, s, PENALTY_WEIGHTS["untrusted_source"]
    if t == "unsupported_claim" and s == "high":
        return t, s, PENALTY_WEIGHTS["private_program_hallucination" if private_flag else "unsupported_claim_high"]
    if t == "unsupported_claim" and s == "medium":
        return t, s, PENALTY_WEIGHTS["unsupported_claim_medium"]
    return t, s, 0


def analysis_penalty_audit(df_none: None = None) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    recs = load_original_500()
    applied: Counter = Counter()
    points: Counter = Counter()
    total_pen_hist: Counter = Counter()
    mismatches = 0
    for rec in recs:
        ev = rec.get("evaluator", {})
        det = ev.get("deterministic_checks", {})
        rub = ev.get("llm_rubric", {})
        scores = rub.get("scores", {}) or {}
        vals = [scores.get(k, 0) for k in RUBRIC_DIMS]
        base = int(round(sum(vals) / len(vals))) if vals else 0
        overall = base
        tp = 0
        for it in _issues_of(rec):
            t, s, p = _penalty_for_issue(it, bool(det.get("private_programs_flag")))
            if p:
                overall -= p
                tp += p
                applied[(t, normalise_severity(s))] += 1
                points[(t, normalise_severity(s))] += p
        overall = max(0, min(10, overall))
        total_pen_hist[tp] += 1
        if overall != ev.get("final", {}).get("overall_score"):
            mismatches += 1

    rows = []
    for (t, s) in sorted(applied):
        rows.append({
            "issue_type": t, "severity": s,
            "penalty_weight_pts": PENALTY_WEIGHTS.get(
                {"math_error": "math_error_high", "missing_citation": "missing_citation",
                 "missing_required_section": "missing_required_section",
                 "untrusted_source": "untrusted_source"}.get(
                    t, f"unsupported_claim_{s}" if t == "unsupported_claim" else t), 0),
            "n_issues_penalised": applied[(t, s)],
            "total_points_deducted": points[(t, s)],
            "applied_empirically_or_heuristically": "heuristic (fixed weights in EvaluatorConfig; not tuned on data)",
        })
    penalty_tbl = pd.DataFrame(rows)

    weights_tbl = pd.DataFrame([
        {"penalty_key": k, "points": v,
         "trigger": {
             "math_error_high": "deterministic arithmetic mismatch vs profile_json (>PLN30) OR rubric math_error/high",
             "missing_citation": "Poland-specific claim w/o inline [S#] before 'Sources used' (any severity)",
             "missing_required_section": "structured answer missing a required section OR needed deterministic repair",
             "untrusted_source": "final answer cites live/Tavily source outside allowed domains",
             "private_program_hallucination": "PPK/IKE/IKZE named in answer but absent from retrieved sources",
             "unsupported_claim_high": "unknown citation id / high-severity unsupported claim (non-private-program)",
             "unsupported_claim_medium": "medium-severity unsupported claim from rubric",
         }[k]} for k, v in PENALTY_WEIGHTS.items()
    ])

    hist_tbl = (pd.Series(total_pen_hist).sort_index()
                .rename_axis("total_penalty_points").reset_index(name="n_cases"))

    combine_note = pd.DataFrame([{
        "aspect": "combination rule",
        "detail": "Penalties are summed (additive) over all issues; each issue contributes independently; "
                  "final = clip(base_score - sum(penalties), 0, 10)."},
        {"aspect": "multiple simultaneous issues",
         "detail": "e.g. missing_citation(-1) + missing_required_section(-2) => -3 total."},
        {"aspect": "threshold sensitivity",
         "detail": "Penalty weights were not selected by tuning against outcomes; see Threshold_Sensitivity "
                   "sheet for routing sensitivity to the acceptance cutoff."},
        {"aspect": "audit reproduction",
         "detail": f"Recomputed overall_score matches stored value for {N_ORIGINAL - mismatches}/{N_ORIGINAL} cases "
                   f"({mismatches} mismatches)."},
    ])

    summary = {
        "penalty_audit": {
            "recomputed_matches_stored": N_ORIGINAL - mismatches,
            "mismatches": mismatches,
            "penalised_issue_counts": {f"{t}/{s}": applied[(t, s)] for (t, s) in applied},
            "total_points_by_type": {f"{t}/{s}": points[(t, s)] for (t, s) in points},
            "cases_with_zero_penalty": int(total_pen_hist.get(0, 0)),
            "note": ("Only missing_citation, missing_required_section and unsupported_claim/medium "
                     "ever reduce the score in the 500-case dataset. unsupported_claim/low, "
                     "unsafe_overconfidence and 'other' issues carry no score penalty."),
        }
    }
    return {"Penalty_Audit": penalty_tbl, "Penalty_Weights": weights_tbl,
            "Penalty_Hist": hist_tbl, "Penalty_Notes": combine_note}, summary


# --------------------------------------------------------------------------------------
# TASK G: threshold sensitivity
# --------------------------------------------------------------------------------------

def analysis_threshold_sensitivity(df: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    rows = []
    scenarios = [
        ("accept>=8, flag<7 (looser accept)", 8, 7),
        ("accept>=9, flag<7 (ACTUAL)", 9, 7),
        ("accept>=10, flag<7 (stricter accept)", 10, 7),
        ("accept>=9, flag<6 (looser flag)", 9, 6),
        ("accept>=9, flag<8 (stricter flag)", 9, 8),
    ]
    for label, acc, flg in scenarios:
        routed = df.apply(lambda r: recompute_routing(r["overall_score"], r["allow_to_show"],
                                                      r["grounded_only_pass"], acc, flg), axis=1)
        vc = routed.value_counts()
        rows.append({
            "scenario": label, "accept_threshold": acc, "flag_below": flg,
            "accepted": int(vc.get("accepted", 0)),
            "needs_review": int(vc.get("needs_review", 0)),
            "flagged": int(vc.get("flagged", 0)),
            "accepted_pct": round(vc.get("accepted", 0) / len(df) * 100, 1),
        })
    tbl = pd.DataFrame(rows)
    actual = tbl[tbl.scenario.str.contains("ACTUAL")].iloc[0]
    summary = {"threshold_sensitivity": {
        "actual": {"accepted": int(actual.accepted), "needs_review": int(actual.needs_review),
                   "flagged": int(actual.flagged)},
        "scenarios": tbl.to_dict(orient="records"),
        "note": ("Post-hoc recomputation only; the reported experiment is unchanged. "
                 "Acceptance requires composite score >= threshold AND grounded_only_pass AND allow_to_show."),
    }}
    return {"Threshold_Sensitivity": tbl}, summary


# --------------------------------------------------------------------------------------
# TASK H: RAG configuration audit
# --------------------------------------------------------------------------------------

def analysis_rag_config() -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    manifests = {}
    for name in ("refuge", "consultant"):
        mp = REPO_ROOT / "indexes" / name / "manifest.json"
        if mp.exists():
            manifests[name] = json.loads(mp.read_text(encoding="utf-8"))

    cons_cfg = (manifests.get("consultant") or {}).get("cfg", {})
    rows = [
        ("embedding_model", cons_cfg.get("embeddings_model", "text-embedding-3-small"),
         "src/tools.py RagConfig.embeddings_model; indexes/*/manifest.json", "yes"),
        ("semantic_chunking", cons_cfg.get("semantic_chunking", True),
         "src/tools.py split_into_chunks() -> SemanticChunker", "yes"),
        ("semantic_breakpoint_threshold_type", cons_cfg.get("semantic_breakpoint_threshold_type", "percentile"),
         "src/tools.py RagConfig", "yes"),
        ("semantic_breakpoint_threshold_amount", cons_cfg.get("semantic_breakpoint_threshold_amount", 95.0),
         "src/tools.py RagConfig", "yes"),
        ("chunk_size (post-semantic size guard)", cons_cfg.get("chunk_size", 1500),
         "run_langgraph.py make_runtime() passes RagConfig(chunk_size=1500)", "yes"),
        ("chunk_overlap (ACTUAL, not the RagConfig default 200)", cons_cfg.get("chunk_overlap", 100),
         "run_langgraph.py make_runtime() passes chunk_overlap=100; manifest confirms 100", "yes"),
        ("size_guard_splitter", "RecursiveCharacterTextSplitter(separators=['\\n\\n','\\n',' ',''])",
         "src/tools.py split_into_chunks()", "yes"),
        ("vector_store", "FAISS (local, faiss-cpu)", "src/tools.py build_index_from_folder()/load_index()", "yes"),
        ("similarity_function", "FAISS default L2 / Euclidean distance on embedding vectors (cosine NOT explicitly set)",
         "agents/consultant.py _retrieve() -> vectorstore.similarity_search()", "yes"),
        ("retrieval_top_k_consultant", 5,
         "agents/consultant.py _combined_sources(..., k=5) in draft() and final()", "yes"),
        ("retrieval_top_k_refugee_context", 4,
         "agents/refugee.py RefugeeAgentConfig.k = 4 (qualitative constraints only, not advice)", "yes"),
        ("reranking", "NOT IMPLEMENTED", "no rerank stage anywhere in agents/ or src/", "no"),
        ("retrieval_score_threshold", "NOT IMPLEMENTED (similarity_search returns top-k unconditionally, no score cutoff)",
         "agents/consultant.py _retrieve()", "no"),
        ("duplicate_source_handling", "NOT IMPLEMENTED as explicit dedup; IDs namespaced S1.. / S100.. (web) / S200.. (Tavily); "
         "final_sources filtered to tags actually cited (_extract_used_source_ids)",
         "agents/consultant.py _combined_sources(), _extract_used_source_ids()", "partial"),
        ("conflicting_source_handling", "NOT IMPLEMENTED as reconciliation; prompt instructs model to prefer retrieved "
         "evidence for Poland-specific claims and to avoid unsupported pension vehicles",
         "agents/consultant.py draft()/final() prompt text", "no"),
        ("live_web_retrieval", "ENABLED by default (CONSULTANT_ENABLE_WEB=true); MCP fetch of curated official URLs, "
         "ranked by _score_url, capped at web_max_sources=4; re-fetched every call (cache written, not read)",
         "agents/consultant.py _live_web_sources(); src/tools.py mcp_fetch_to_file()", "yes"),
        ("allowed_domains", ",".join(["gov.pl", "zus.pl", "podatki.gov.pl", "biznes.gov.pl", "euraxess.pl", "ec.europa.eu"]),
         "agents/consultant.py ConsultantAgentConfig.allowed_domains / .env.example", "yes"),
        ("open_web_search_tavily", "DISABLED by default (CONSULTANT_ENABLE_TAVILY=false); when on, restricted to allowed_domains",
         "agents/consultant.py _tavily_sources(); src/tools.py tavily_search()", "off"),
        ("live_vs_open_web_criterion", "Curated institutional URLs are topic-routed (_detect_topics -> _topic_url_map); "
         "open-web (Tavily) is opt-in and domain-restricted; used only 5x across the 500-case dataset",
         "agents/consultant.py _candidate_urls_for_query(), _tavily_sources()", "yes"),
        ("source_id_namespaces", "S1-S99 vectorstore, S100+ live web, S200+ Tavily", "agents/consultant.py", "yes"),
    ]
    df = pd.DataFrame(rows, columns=["parameter", "value", "code_location", "implemented"])

    idx_rows = []
    for name, man in manifests.items():
        idx_rows.append({
            "index": name,
            "n_docs": man.get("n_docs"),
            "n_chunks": man.get("n_chunks"),
            "built_at_unix": man.get("built_at_unix"),
            "build_seconds": man.get("build_seconds"),
            "n_pdf": len(man.get("inputs", {}).get("pdf", [])),
            "n_html": len(man.get("inputs", {}).get("html", [])),
            "n_xlsx": len(man.get("inputs", {}).get("xlsx", [])),
        })
    idx_df = pd.DataFrame(idx_rows)

    # observed source-channel mix in the 500-case final answers
    fo = frame_original()
    mix = pd.DataFrame([{
        "channel": "vectorstore (local FAISS)", "n_source_objects_in_final_answers": int(fo["n_src_vectorstore"].sum())},
        {"channel": "live_web (MCP fetch)", "n_source_objects_in_final_answers": int(fo["n_src_live_web"].sum())},
        {"channel": "tavily_search", "n_source_objects_in_final_answers": int(fo["n_src_tavily"].sum())},
    ])

    summary = {"rag_config": {r[0]: r[1] for r in rows},
               "rag_index_manifests": idx_rows,
               "rag_observed_source_mix_500": mix.set_index("channel").iloc[:, 0].to_dict()}
    return {"RAG_Config": df, "RAG_Index_Manifests": idx_df, "RAG_Source_Mix": mix}, summary


# --------------------------------------------------------------------------------------
# TASK I: subgroup statistics
# --------------------------------------------------------------------------------------

OUTCOMES = ["overall_score"] + [f"score_{d}" for d in RUBRIC_DIMS]


def _descr_rows(df: pd.DataFrame, group_col: str, groups: Sequence[Any], labels: Sequence[str]) -> List[Dict[str, Any]]:
    rows = []
    for g, lab in zip(groups, labels):
        sub = df[df[group_col] == g] if not callable(g) else df[g(df)]
        for oc in OUTCOMES:
            m, sd, lo, hi, n = mean_ci(sub[oc].tolist())
            rows.append({"grouping": group_col, "group": lab, "outcome": oc, "n": n,
                         "mean": round(m, 3), "sd": round(sd, 3),
                         "median": round(float(np.nanmedian(sub[oc].astype(float))), 3) if n else math.nan,
                         "ci95_low": round(lo, 3), "ci95_high": round(hi, 3)})
    return rows


def analysis_subgroups(df: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    df = df.copy()
    descr: List[Dict[str, Any]] = []
    tests: List[Dict[str, Any]] = []
    effects: List[Dict[str, Any]] = []

    # --- employment status (k groups) ---
    emp_levels = ["permanent_job", "self_employed", "other_work", "unemployed", "economically_inactive"]
    emp_present = [e for e in emp_levels if e in set(df.employment_status)]
    descr += _descr_rows(df, "employment_status", emp_present, emp_present)
    for oc in OUTCOMES:
        groups = {e: df[df.employment_status == e][oc].tolist() for e in emp_present}
        mt = multi_group_tests(groups)
        tests.append({"grouping": "employment_status", "outcome": oc, "test": "ANOVA + Kruskal-Wallis",
                      **{k: mt[k] for k in ("k_groups", "anova_F", "anova_p", "kruskal_H", "kruskal_p", "note")}})
        effects.append({"grouping": "employment_status", "outcome": oc,
                        "eta_squared": round(mt["eta_squared"], 4) if not math.isnan(mt["eta_squared"]) else math.nan,
                        "epsilon_squared": round(mt["epsilon_squared"], 4) if not math.isnan(mt["epsilon_squared"]) else math.nan})

    # --- deficit vs positive surplus (2 groups) ---
    for gcol, a_lab, b_lab, a_mask, b_mask in [
        ("surplus_position", "deficit(surplus<=0)", "positive_surplus", df.is_deficit, ~df.is_deficit),
        ("dependents", "has_dependents", "no_dependents", df.has_dependents, ~df.has_dependents),
    ]:
        for lab, mask in [(a_lab, a_mask), (b_lab, b_mask)]:
            sub = df[mask]
            for oc in OUTCOMES:
                m, sd, lo, hi, n = mean_ci(sub[oc].tolist())
                descr.append({"grouping": gcol, "group": lab, "outcome": oc, "n": n,
                              "mean": round(m, 3), "sd": round(sd, 3),
                              "median": round(float(np.nanmedian(sub[oc].astype(float))), 3) if n else math.nan,
                              "ci95_low": round(lo, 3), "ci95_high": round(hi, 3)})
        for oc in OUTCOMES:
            a = df[a_mask][oc].tolist()
            b = df[b_mask][oc].tolist()
            tt = two_group_tests(a, b)
            es = hedges_g(a, b)
            am, *_ = mean_ci(a)
            bm, *_ = mean_ci(b)
            tests.append({"grouping": gcol, "outcome": oc, "test": "Welch t + Mann-Whitney U",
                          "mean_diff(a_minus_b)": round((am - bm), 3) if not (math.isnan(am) or math.isnan(bm)) else math.nan,
                          **{k: (round(tt[k], 4) if isinstance(tt[k], float) and not math.isnan(tt[k]) else tt[k])
                             for k in ("n1", "n2", "welch_t", "welch_p", "mwu_U", "mwu_p", "note")}})
            effects.append({"grouping": gcol, "outcome": oc,
                            "cohens_d": round(es["cohens_d"], 4) if not math.isnan(es["cohens_d"]) else math.nan,
                            "hedges_g": round(es["hedges_g"], 4) if not math.isnan(es["hedges_g"]) else math.nan})

    sheets = {
        "Subgroup_Descr": pd.DataFrame(descr),
        "Subgroup_Tests": pd.DataFrame(tests),
        "Effect_Sizes": pd.DataFrame(effects),
    }
    summary = {"subgroups": {
        "scipy_available": SCIPY,
        "framing": "exploratory; synthetic profiles generated by interdependent simulation rules; no causal claims",
        "employment_levels": emp_present,
        "deficit_n": int(df.is_deficit.sum()),
        "has_dependents_n": int(df.has_dependents.sum()),
    }}
    return sheets, summary


# --------------------------------------------------------------------------------------
# TASK J: synthetic budget-generation audit
# --------------------------------------------------------------------------------------

def analysis_budget_generation(df: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    logic = pd.DataFrame([
        {"step": "monthly income", "code_location": "refugee.py generate_profile_numbers()",
         "implementation": "uniform(income_min, income_max) from the gender x employment band, rounded to nearest PLN 10"},
        {"step": "expense categories", "code_location": "refugee.py generate_profile_numbers()",
         "implementation": "each = income * triangular(low,mode,high) share; food share +.02/.03 if dependents>=1; "
                           "transport share +.01/.02 if employed; then rounded to PLN 10"},
        {"step": "hard expense floors", "code_location": "refugee.py generate_profile_numbers()",
         "implementation": "housing>=500+100*dep; food>=400+200*dep; other>=120+80*dep; utilities>=50; transport>=30"},
        {"step": "remittances", "code_location": "refugee.py generate_profile_numbers()",
         "implementation": "0 if remittances_flag==0 else income*triangular(.05,.10,.18), +.02 if dep>1, capped at 22% income"},
        {"step": "minimum desired surplus target", "code_location": "refugee.py generate_profile_numbers()",
         "implementation": "max(min_retirement_surplus_pln=100, income*triangular(...)); band depends on employment "
                           "(permanent/self .05-.10; other_work .03-.07; unemployed/inactive .02-.05)"},
        {"step": "feasibility adjustment 1", "code_location": "refugee.py generate_profile_numbers()",
         "implementation": "if expenses > income - surplus_target: cut 'other' down toward other_min only"},
        {"step": "feasibility adjustment 2", "code_location": "refugee.py generate_profile_numbers()",
         "implementation": "if still over: proportionally scale ONLY flexible items (utilities>min, transport>min, "
                           "healthcare, remittances); housing/food/other/utilities_min/transport_min kept fixed"},
        {"step": "final surplus", "code_location": "refugee.py generate_profile_numbers()",
         "implementation": "surplus = max(0.0, income - expenses)  -> NOTE: clamped at 0, cannot go negative here"},
        {"step": "savings", "code_location": "refugee.py generate_profile_numbers()",
         "implementation": "uniform(0.5,2.0)*surplus + uniform(0,0.5)*(months/12)*200; unemployed/inactive & surplus<200 "
                           "-> uniform(0,350) low-savings draw"},
        {"step": "deficit possibility", "code_location": "refugee.py + consultant.py",
         "implementation": "generate_profile_numbers() surplus is >=0; DEFICIT in analysis = recomputed "
                           "income - sum(expenses incl. remittances) <= 0 (consultant/evaluator recompute, no 0-clamp)"},
    ])

    s = df["surplus"].astype(float)
    pct = {f"p{p}": round(float(np.percentile(s, p)), 1) for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)}
    stats_tbl = pd.DataFrame([
        {"metric": "n cases", "value": int(len(df))},
        {"metric": "deficit cases (surplus<=0)", "value": int(df.is_deficit.sum())},
        {"metric": "deficit pct", "value": round(float(df.is_deficit.mean() * 100), 1)},
        {"metric": "surplus mean (PLN)", "value": round(float(s.mean()), 1)},
        {"metric": "surplus median (PLN)", "value": round(float(s.median()), 1)},
        {"metric": "surplus sd (PLN)", "value": round(float(s.std(ddof=1)), 1)},
        {"metric": "surplus min (PLN)", "value": round(float(s.min()), 1)},
        {"metric": "surplus max (PLN)", "value": round(float(s.max()), 1)},
        *[{"metric": f"surplus {k}", "value": v} for k, v in pct.items()],
        {"metric": "income median (PLN)", "value": round(float(df.income.median()), 1)},
        {"metric": "expenses_total median (PLN)", "value": round(float(df.expenses_total.median()), 1)},
        {"metric": "savings median (PLN)", "value": round(float(df.savings.median()), 1)},
    ])

    summary = {"budget_generation": {
        "deficit_cases": int(df.is_deficit.sum()),
        "deficit_pct": round(float(df.is_deficit.mean() * 100), 1),
        "surplus_mean": round(float(s.mean()), 1),
        "surplus_median": round(float(s.median()), 1),
        "surplus_min": round(float(s.min()), 1),
        "surplus_max": round(float(s.max()), 1),
        "surplus_percentiles": pct,
        "feasibility_note": ("The generator preserves core floors (housing/food/other/min utilities/min transport) "
                             "and trims discretionary 'other' then scales flexible items to hit a small minimum "
                             "surplus target; this can make synthetic households more solvent than some real "
                             "refugee households. Numbers are not population estimates."),
    }}
    return {"Budget_Generation": logic, "Budget_Stats": stats_tbl}, summary


# --------------------------------------------------------------------------------------
# TASK K: calibration provenance audit
# --------------------------------------------------------------------------------------

def analysis_calibration_provenance() -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    NBP = "Narodowy Bank Polski (2025), 'The living and economic situation of migrants from Ukraine in Poland in 2024'"
    rows = [
        {"variable": "gender", "implemented_rule": "female prior 0.66 (Bernoulli)",
         "source": NBP, "type": "survey-derived", "year": "2024 survey / 2025 report", "sample_size": "3,965",
         "code_location": "refugee.py RefugeeAgentConfig.female_prior=0.66; sample_anchors()",
         "notes": "README & manuscript Table 1 agree (female 0.66 / male 0.34)."},
        {"variable": "employment_status", "implemented_rule": "categorical [.54 permanent, .04 self, .17 other_work, "
         ".14 unemployed, .11 economically_inactive]",
         "source": NBP, "type": "survey-derived", "year": "2024 survey / 2025 report", "sample_size": "3,965",
         "code_location": "refugee.py sample_anchors()",
         "notes": "Matches README and manuscript Table 1 (54/4/17/14/11%)."},
        {"variable": "monthly income", "implemented_rule": "uniform() within gender x employment PLN bands "
         "(unemployed/inactive 800-3500; female 2000-7000; male 2500-8500), sub-band weights per code",
         "source": NBP + " (medians: female PLN 3,872; male PLN 4,665)", "type": "survey-anchored + modelling assumption",
         "year": "2024 survey / 2025 report", "sample_size": "3,965",
         "code_location": "refugee.py sample_anchors() income_band; generate_profile_numbers()",
         "notes": "Band EDGES and sub-band weights are researcher-set to reproduce reported medians; manuscript "
                  "Table 1 only states 'PLN 800-8,500'. Not a direct survey distribution."},
        {"variable": "polish_language_level", "implemented_rule": "categorical weights CONDITIONAL on months_in_poland "
         "(<=12 / <=24 / >24) and employment; NOT the marginal 5/39/42/14 split",
         "source": "README cites NBP 2025 Fig. 11 marginal none 5% / basic 39% / intermediate 42% / advanced 14%",
         "type": "modelling assumption (survey-informed)", "year": "2024 survey / 2025 report", "sample_size": "3,965",
         "code_location": "refugee.py sample_anchors() lang_weights",
         "notes": "DISCREPANCY: code implements a month/employment-conditional model, not the marginal survey "
                  "proportions the README/manuscript imply. Manuscript wording ('calibrated to NBP 2025 "
                  "language-proficiency proportions') should be softened to 'survey-informed'."},
        {"variable": "remittances", "implemented_rule": "P(send)=0.20 if unemployed/inactive else 0.36-0.50 via a "
         "household-support proxy; amount = income*triangular(.05,.10,.18) capped 22%",
         "source": NBP + " (~36% send remittances)", "type": "survey-anchored + modelling assumption",
         "year": "2024 survey / 2025 report", "sample_size": "3,965",
         "code_location": "refugee.py sample_anchors() rem_flag; generate_profile_numbers() rem_share",
         "notes": "Base 36% is survey-derived; the employment/proxy split and the 22% cap are researcher-set."},
        {"variable": "dependents", "implemented_rule": "categorical [.35, .40, .20, .05] for 0/1/2/3",
         "source": "not tied to a specific NBP figure in code comments",
         "type": "modelling assumption", "year": "n/a", "sample_size": "n/a",
         "code_location": "refugee.py sample_anchors()",
         "notes": "Manuscript Table 1 lists these % as a 'weighted categorical draw'; text (Sec 3.3) lists dependents "
                  "under modelling assumptions. Keep consistent -> modelling assumption."},
        {"variable": "months_in_poland", "implemented_rule": "randint(6, 48) discrete uniform",
         "source": "researcher-set range", "type": "modelling assumption", "year": "n/a", "sample_size": "n/a",
         "code_location": "refugee.py RefugeeAgentConfig.min_months_in_poland=6; sample_anchors()",
         "notes": "Matches manuscript Table 1 (6-48 months, discrete uniform)."},
        {"variable": "savings", "implemented_rule": "uniform(0.5,2.0)*surplus + uniform(0,0.5)*(months/12)*200; "
         "low-savings draw uniform(0,350) for unemployed/inactive with surplus<200",
         "source": "researcher-set", "type": "modelling assumption", "year": "n/a", "sample_size": "n/a",
         "code_location": "refugee.py generate_profile_numbers()",
         "notes": "Fully synthetic; <PLN1,500 -> 'no dedicated retirement plan' flag (persona controls)."},
        {"variable": "expense shares (housing/utilities/food/transport/healthcare/other)",
         "implemented_rule": "triangular() shares of income with the modes/floors listed in Budget_Generation sheet",
         "source": "manuscript: 'reflect Polish household budget proportions', not from the NBP refugee survey",
         "type": "modelling assumption", "year": "n/a", "sample_size": "n/a",
         "code_location": "refugee.py generate_profile_numbers()",
         "notes": "Shares, modes and floors are researcher-set for internal consistency."},
        {"variable": "NBP survey provenance", "implemented_rule": "n = 3,965 respondents",
         "source": "refugee.py comment 'NBP 2025 survey ... (n=3,965)'; README 'n = 3,965 respondents, April-June 2025'; "
                   "manuscript 'included 3,965 respondents'",
         "type": "reference-consistency check", "year": "manuscript says '2025 National Bank of Poland survey'; "
         "the cited NBP (2025) report covers the situation IN 2024", "sample_size": "3,965",
         "code_location": "refugee.py header comments; README.md 'Profile Variables (NBP 2025)'",
         "notes": "n=3,965 is consistent across code / README / manuscript. Minor wording issue: 'NBP 2025 survey' vs "
                  "report title year; README adds 'April-June 2025' fieldwork which is not stated in the manuscript."},
    ]
    df = pd.DataFrame(rows)
    summary = {"calibration_provenance": {
        "survey_derived": ["gender", "employment_status", "remittances(base rate)"],
        "survey_anchored_plus_assumptions": ["monthly income"],
        "modelling_assumptions": ["dependents", "months_in_poland", "savings", "expense shares",
                                  "polish_language_level (conditional model)"],
        "discrepancies": [
            "polish_language_level: code uses a months/employment-conditional model, not the marginal "
            "NBP Fig.11 proportions implied by README/manuscript.",
            "monthly income: only band edges/medians are survey-anchored; sub-band weights are researcher-set.",
            "NBP wording: manuscript 'NBP 2025 survey' vs report describing 2024; README adds 'April-June 2025' "
            "fieldwork window absent from the manuscript.",
        ],
        "consistent": ["n=3,965 sample size across code/README/manuscript",
                       "employment 54/4/17/14/11%", "gender 66/34%", "months 6-48 uniform"],
    }}
    return {"Calibration": df}, summary


# --------------------------------------------------------------------------------------
# TASK L: arithmetic safeguard analysis
# --------------------------------------------------------------------------------------

def analysis_arithmetic_safeguard(df: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    recs = load_original_500()
    n = len(recs)
    n_repair = 0
    n_arith_mismatch = 0
    n_struct_budget_present = 0
    mismatch_magnitudes: List[float] = []
    for rec in recs:
        ev = rec.get("evaluator", {})
        det = ev.get("deterministic_checks", {})
        cf = rec.get("consultant_final", {})
        ri = (cf.get("repair_issues") or []) + (det.get("repair_issues") or [])
        if ri:
            n_repair += 1
        if det.get("expenses_total_mismatch") or det.get("surplus_mismatch"):
            n_arith_mismatch += 1
        qb = (cf.get("final_answer_struct") or {}).get("quick_budget_check")
        if isinstance(qb, dict) and qb.get("monthly_income") is not None:
            n_struct_budget_present += 1
        ce, cc = det.get("computed_expenses"), det.get("claimed_expenses_total")
        if isinstance(ce, (int, float)) and isinstance(cc, (int, float)):
            mismatch_magnitudes.append(abs(float(cc) - float(ce)))

    responsibilities = pd.DataFrame([
        {"stage": "Consultant generation",
         "role": "LLM writes narrative + advice; it is TOLD the arithmetic (income/expenses/surplus) in the prompt "
                 "but its numeric output is not trusted."},
        {"stage": "Deterministic recomputation (Consultant)",
         "role": "consultant.final() OVERWRITES quick_budget_check.monthly_income/_expenses_total/_surplus with values "
                 "recomputed from profile_json before the answer is returned (agents/consultant.py final())."},
        {"stage": "Deterministic check (Evaluator)",
         "role": "evaluator.evaluate() recomputes income/expenses/surplus from profile_json and compares to the "
                 "answer's stated figures with +-PLN 30 tolerance; a mismatch => math_error issue."},
        {"stage": "Rubric score (Evaluator LLM)",
         "role": "arithmetic_consistency dimension (0-10) scored by DeepSeek AFTER the deterministic overwrite, so it "
                 "mostly reflects the injected numbers, not the Consultant's unaided arithmetic."},
    ])
    summary_tbl = pd.DataFrame([
        {"metric": "total cases", "value": n},
        {"metric": "cases with deterministic section repair (repair_issues)", "value": n_repair},
        {"metric": "cases with deterministic arithmetic mismatch flagged", "value": n_arith_mismatch},
        {"metric": "cases with NO arithmetic mismatch", "value": n - n_arith_mismatch},
        {"metric": "pct with arithmetic mismatch", "value": round(n_arith_mismatch / n * 100, 2)},
        {"metric": "pct with section repair", "value": round(n_repair / n * 100, 2)},
        {"metric": "cases with structured quick_budget_check present", "value": n_struct_budget_present},
        {"metric": "max |claimed-computed expenses| (PLN) across cases", "value": round(max(mismatch_magnitudes), 1) if mismatch_magnitudes else None},
        {"metric": "mean |claimed-computed expenses| (PLN)", "value": round(float(np.mean(mismatch_magnitudes)), 2) if mismatch_magnitudes else None},
    ])
    summary = {"arithmetic_safeguard": {
        "n_cases": n,
        "arithmetic_mismatch_cases": n_arith_mismatch,
        "arithmetic_mismatch_pct": round(n_arith_mismatch / n * 100, 2),
        "section_repair_cases": n_repair,
        "interpretation": ("The near-perfect arithmetic_consistency score (mean 9.99) primarily reflects the "
                           "deterministic recomputation/overwrite of the budget block, not unaided LLM numerical "
                           "reasoning. This should be stated in Results, not only Discussion."),
    }}
    return {"Arithmetic": summary_tbl, "Arithmetic_Responsibilities": responsibilities}, summary


# --------------------------------------------------------------------------------------
# TASK M: representative sample interaction
# --------------------------------------------------------------------------------------

def _pick_representative(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    cand = []
    for r in recs:
        f = flatten(r)
        if (f["case_status"] == "accepted" and f["overall_score"] == 9 and f["grounded_only_pass"]
                and f["has_dependents"] and f["employment_status"] == "permanent_job"
                and not f["is_deficit"] and f["n_repair_issues"] == 0
                and len(r.get("refugee_clarifying_qa") or []) == 3):
            cand.append(r)
    if not cand:  # relax
        cand = [r for r in recs if flatten(r)["case_status"] == "accepted" and flatten(r)["overall_score"] == 9]
    cand.sort(key=lambda r: r.get("ts_unix", 0))
    return cand[0]


def analysis_sample_interaction() -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    recs = load_original_500()
    r = _pick_representative(recs)
    ref = r["refugee"]
    draft = r.get("consultant_draft", {})
    cf = r.get("consultant_final", {})
    ev = r.get("evaluator", {})
    rub = ev.get("llm_rubric", {})
    qa = r.get("refugee_clarifying_qa", [])

    parts = [
        ("run_id", r.get("run_id")),
        ("case_status", r.get("case_status")),
        ("PROFILE (text)", ref.get("profile_text")),
        ("PERSONA (json)", json.dumps(ref.get("persona_json", {}), ensure_ascii=False)),
        ("CLIENT QUESTION", ref.get("user_query")),
        ("CONSULTANT DRAFT", (draft.get("draft_answer") or "")[:1800]),
        ("CLARIFYING QUESTIONS", " | ".join(draft.get("clarifying_questions") or [])),
        ("REFUGEE ANSWERS", " || ".join(f"Q: {x.get('question','')} A: {x.get('answer','')}" for x in qa)),
        ("FINAL RECOMMENDATION", cf.get("final_answer")),
        ("SOURCES USED", "; ".join(f"[{s.get('id')}] {s.get('source')} (p.{s.get('page')}, {s.get('source_type')})"
                                   for s in (cf.get("final_sources") or []))),
        ("EVALUATOR SCORES", json.dumps(rub.get("scores", {}), ensure_ascii=False)),
        ("EVALUATOR ISSUES", json.dumps(_issues_of(r), ensure_ascii=False)),
        ("SCORING AUDIT (if present)", json.dumps(rub.get("scoring_audit", "not in original record"), ensure_ascii=False)),
        ("FINAL VERDICT", json.dumps(ev.get("final", {}), ensure_ascii=False)),
    ]
    df = pd.DataFrame(parts, columns=["field", "value"])
    summary = {"sample_interaction": {k: v for k, v in parts}}
    return {"Sample_Interaction": df}, summary


# --------------------------------------------------------------------------------------
# TASK B: repeated-run stability analysis
# --------------------------------------------------------------------------------------

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def analyze_repeated(path: Path = REPEATED_JSONL) -> None:
    if not path.exists():
        print(f"[skip] {path} not found - run repeated_runs.py first.")
        return
    rows = _load_jsonl(path)
    n_error = sum(1 for x in rows if x.get("case_status") == "error")
    recs = []
    for x in rows:
        if x.get("case_status") == "error":
            continue
        ev = x.get("evaluator", {})
        rub = ev.get("llm_rubric", {})
        sc = rub.get("scores", {})
        recs.append({
            "group": x.get("profile_repeat_group_id"),
            "original_run_id": x.get("original_run_id"),
            "repeat_id": x.get("repeat_id"),
            "overall_score": ev.get("final", {}).get("overall_score"),
            "case_status": x.get("case_status"),
            "issue_types": tuple(sorted({i.get("type") for i in rub.get("issues", []) if isinstance(i, dict)})),
            **{f"score_{d}": sc.get(d) for d in RUBRIC_DIMS},
        })
    d = pd.DataFrame(recs)
    if d.empty:
        print("[skip] repeated jsonl empty")
        return
    if n_error:
        print(f"[note] excluded {n_error} error stub(s) from the repeated-run analysis")

    # per-profile score stability
    prof_rows = []
    for g, sub in d.groupby("group"):
        vals = sub["overall_score"].astype(float).to_numpy()
        modal = sub["case_status"].mode().iloc[0]
        prof_rows.append({
            "group": g, "original_run_id": sub["original_run_id"].iloc[0], "n_repeats": len(sub),
            "score_mean": round(float(np.mean(vals)), 3), "score_sd": round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 3),
            "score_min": float(np.min(vals)), "score_max": float(np.max(vals)),
            "score_range": float(np.max(vals) - np.min(vals)),
            "score_mad": round(float(np.mean(np.abs(vals - np.mean(vals)))), 3),
            "modal_routing": modal,
            "pct_matching_modal": round(float((sub["case_status"] == modal).mean() * 100), 1),
            "all_repeats_identical_routing": bool(sub["case_status"].nunique() == 1),
        })
    prof_df = pd.DataFrame(prof_rows)

    # per-dimension within-profile variability
    dim_rows = []
    for dim in ["overall_score"] + [f"score_{x}" for x in RUBRIC_DIMS]:
        within_sd = d.groupby("group")[dim].apply(lambda s: np.std(s.astype(float), ddof=1) if len(s) > 1 else 0.0)
        mat = (d.pivot_table(index="group", columns="repeat_id", values=dim)).to_numpy(dtype=float)
        icc = icc21(mat)
        dim_rows.append({
            "dimension": dim,
            "grand_mean": round(float(d[dim].astype(float).mean()), 3),
            "mean_within_profile_sd": round(float(np.nanmean(within_sd)), 4),
            "max_within_profile_sd": round(float(np.nanmax(within_sd)), 4),
            "pct_profiles_zero_variance": round(float((within_sd == 0).mean() * 100), 1),
            "icc21": round(icc["icc"], 4) if not math.isnan(icc["icc"]) else math.nan,
            "icc_note": icc["note"],
        })
    dim_df = pd.DataFrame(dim_rows)

    # routing agreement
    routing_rows = d.groupby("group")["case_status"].apply(list).tolist()
    cats = ["accepted", "needs_review", "flagged"]
    fk = fleiss_kappa(routing_rows, cats)
    pct_identical = float(prof_df["all_repeats_identical_routing"].mean() * 100)
    pct_modal = float(prof_df["pct_matching_modal"].mean())
    overall_route_dist = d["case_status"].value_counts().to_dict()

    # pairwise Jaccard of issue-type sets
    jac = []
    for g, sub in d.groupby("group"):
        sets = [set(x) for x in sub["issue_types"]]
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                jac.append(jaccard(sets[i], sets[j]))
    mean_jac = float(np.mean(jac)) if jac else math.nan

    agree_df = pd.DataFrame([
        {"metric": "n profiles", "value": int(prof_df["group"].nunique())},
        {"metric": "repeats per profile", "value": int(d.groupby('group').size().mode().iloc[0])},
        {"metric": "total repeated runs", "value": int(len(d))},
        {"metric": "% profiles identical routing in all repeats", "value": round(pct_identical, 1)},
        {"metric": "mean % repeats matching modal routing", "value": round(pct_modal, 1)},
        {"metric": "Fleiss kappa (routing)", "value": round(fk["fleiss_kappa"], 4) if not math.isnan(fk["fleiss_kappa"]) else None},
        {"metric": "mean pairwise Jaccard (issue-type sets)", "value": round(mean_jac, 4) if not math.isnan(mean_jac) else None},
        {"metric": "routing distribution (all repeats)", "value": json.dumps(overall_route_dist)},
        {"metric": "mean within-profile SD of overall_score", "value": round(float(prof_df['score_sd'].mean()), 4)},
        {"metric": "% profiles with 0 overall_score variance", "value": round(float((prof_df['score_sd'] == 0).mean() * 100), 1)},
    ])

    write_sheets({"Repeated_Profile": prof_df, "Repeated_Scores": dim_df, "Routing_Agreement": agree_df})
    update_summary({"repeated_runs_analysis": {
        "n_profiles": int(prof_df["group"].nunique()),
        "total_runs": int(len(d)),
        "error_stubs_excluded": int(n_error),
        "pct_profiles_identical_routing": round(pct_identical, 1),
        "mean_pct_matching_modal": round(pct_modal, 1),
        "fleiss_kappa_routing": fk["fleiss_kappa"],
        "mean_within_profile_sd_overall": round(float(prof_df["score_sd"].mean()), 4),
        "mean_pairwise_jaccard_issue_types": mean_jac,
        "icc21_by_dimension": {r["dimension"]: r["icc21"] for r in dim_rows},
    }})
    _fig_repeated(prof_df, dim_df)


# --------------------------------------------------------------------------------------
# TASK D: no-RAG baseline analysis
# --------------------------------------------------------------------------------------

_ROUTE_RANK = {"flagged": 0, "needs_review": 1, "accepted": 2}
_CITE_TAG_RE = __import__("re").compile(r"\[(?:S|W)\d+\]")
_NO_SOURCES_NOTE = "No external sources were retrieved in this experimental condition."


def _baseline_flat(x: Dict[str, Any]) -> Dict[str, Any]:
    """One baseline run -> flat row. retrieved_sources_count for without_rag is forced to 0 (item 7)."""
    ev = x.get("evaluator", {})
    rub = ev.get("llm_rubric", {})
    sc = rub.get("scores", {}) or {}
    final = ev.get("final", {})
    det = ev.get("deterministic_checks", {})
    cf = x.get("consultant_final", {})
    issues = [i for i in rub.get("issues", []) if isinstance(i, dict)]
    itypes = [i.get("type") for i in issues]
    cond = x.get("condition")
    rsc = x.get("retrieved_sources_count")
    if rsc is None:
        rsc = len(cf.get("sources") or [])
    if cond == "without_rag":
        rsc = 0
    repair_fields = {str(r.get("field")) for r in (cf.get("repair_issues") or []) if isinstance(r, dict)}
    ans = cf.get("final_answer") or ""
    return {
        "pair": x.get("profile_pair_id"),
        "original_run_id": x.get("original_run_id"),
        "condition": cond,
        "overall_score": final.get("overall_score"),
        "case_status": x.get("case_status"),
        "grounded_only_pass": bool(final.get("grounded_only_pass")),
        "allow_to_show": bool(final.get("allow_to_show")),
        "retrieved_sources_count": int(rsc),
        **{f"score_{dm}": sc.get(dm) for dm in RUBRIC_DIMS},
        "n_issues": len(issues),
        "n_missing_citation": sum(1 for t in itypes if t == "missing_citation"),
        "n_unsupported_claim": sum(1 for t in itypes if t == "unsupported_claim"),
        "n_unsupported_claim_high": sum(1 for i in issues if i.get("type") == "unsupported_claim" and i.get("severity") == "high"),
        "n_unsafe_overconfidence": sum(1 for t in itypes if t == "unsafe_overconfidence"),
        "n_missing_required_section": sum(1 for t in itypes if t == "missing_required_section"),
        "n_untrusted_source": sum(1 for t in itypes if t == "untrusted_source"),
        "n_source_support_issues": sum(1 for t in itypes if t in ("unsupported_claim", "missing_citation", "untrusted_source")),
        "has_missing_citation": "missing_citation" in itypes,
        "has_unsupported_claim": "unsupported_claim" in itypes,
        "has_unsafe_overconfidence": "unsafe_overconfidence" in itypes,
        "has_missing_required_section": "missing_required_section" in itypes,
        # artefact probes (item 8) -- only meaningful for without_rag
        "hallucinated_cite_tag_in_answer": bool(_CITE_TAG_RE.search(ans)),
        "det_citations": list(det.get("citations") or []),
        "det_unknown_citations": list(det.get("unknown_citations") or []),
        "repair_summary_inline_citation": "summary_inline_citation" in repair_fields,
        "repair_sources_used": "sources_used" in repair_fields,
        "note_flagged_as_citation": _NO_SOURCES_NOTE in list(det.get("citations") or []),
    }


def analyze_baseline(path: Path = BASELINE_JSONL) -> None:
    if not path.exists():
        print(f"[skip] {path} not found - run repeated_runs.py baseline first.")
        return
    d = pd.DataFrame([_baseline_flat(x) for x in _load_jsonl(path)])
    if d.empty:
        print("[skip] baseline jsonl empty")
        return

    w = d[d.condition == "with_rag"].set_index("pair")
    wo = d[d.condition == "without_rag"].set_index("pair")
    pairs = sorted(set(w.index) & set(wo.index))
    n_pairs = len(pairs)

    score_metrics = ["overall_score"] + [f"score_{x}" for x in RUBRIC_DIMS]
    issue_metrics = ["n_issues", "n_missing_citation", "n_unsupported_claim", "n_unsafe_overconfidence",
                     "n_missing_required_section", "n_untrusted_source", "n_source_support_issues"]

    # ---- per-pair paired table (item 3) ----
    pair_rows = []
    for p in pairs:
        rw, rwo = w.loc[p], wo.loc[p]
        row = {"pair": p, "original_run_id": rw["original_run_id"],
               "route_with": rw["case_status"], "route_without": rwo["case_status"],
               "rsc_with": int(rw["retrieved_sources_count"]), "rsc_without": int(rwo["retrieved_sources_count"])}
        for m in score_metrics:
            row[f"{m}_with"] = rw[m]
            row[f"{m}_without"] = rwo[m]
            row[f"{m}_diff"] = (rw[m] - rwo[m]) if pd.notna(rw[m]) and pd.notna(rwo[m]) else math.nan
        pair_rows.append(row)
    pairs_df = pd.DataFrame(pair_rows)

    # ---- paired stats (item 4) ----
    stat_rows = []
    for m in score_metrics + issue_metrics:
        a = np.array([w.loc[p, m] for p in pairs], dtype=float)
        b = np.array([wo.loc[p, m] for p in pairs], dtype=float)
        mask = ~(np.isnan(a) | np.isnan(b))
        a, b = a[mask], b[mask]
        if a.size == 0:
            continue
        diff = a - b
        m_, sd_, lo, hi, n = mean_ci(diff.tolist())
        row = {"metric": m, "n_pairs": int(n),
               "mean_with": round(float(a.mean()), 3), "median_with": round(float(np.median(a)), 3),
               "mean_without": round(float(b.mean()), 3), "median_without": round(float(np.median(b)), 3),
               "mean_diff": round(m_, 3), "median_diff": round(float(np.median(diff)), 3),
               "ci95_low": round(lo, 3), "ci95_high": round(hi, 3),
               "paired_t": math.nan, "p_ttest": math.nan, "wilcoxon_W": math.nan, "p_wilcoxon": math.nan,
               "cohen_dz": math.nan, "note": ""}
        if np.std(diff) == 0:
            row["note"] = "all paired differences identical"
        elif SCIPY:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    t, pt = _sps.ttest_rel(a, b)
                    row["paired_t"], row["p_ttest"] = round(float(t), 4), float(f"{pt:.3g}")
                except Exception as e:  # pragma: no cover
                    row["note"] += f" ttest_err={e!r}"
                try:
                    W, pw = _sps.wilcoxon(a, b)
                    row["wilcoxon_W"], row["p_wilcoxon"] = round(float(W), 2), float(f"{pw:.3g}")
                except Exception as e:  # pragma: no cover
                    row["note"] += f" wilcoxon_err={e!r}"
            row["cohen_dz"] = round(float(np.mean(diff) / np.std(diff, ddof=1)), 4)
        else:
            row["note"] = "scipy missing; descriptive only"
        stat_rows.append(row)
    stats_df = pd.DataFrame(stat_rows)

    # ---- issue analysis: issues vs cases, per condition (item 5) ----
    iss_rows = []
    for label, col in [("missing_citation", "n_missing_citation"), ("unsupported_claim", "n_unsupported_claim"),
                       ("unsupported_claim_high", "n_unsupported_claim_high"),
                       ("unsafe_overconfidence", "n_unsafe_overconfidence"),
                       ("missing_required_section", "n_missing_required_section"),
                       ("untrusted_source", "n_untrusted_source"),
                       ("source_support_related", "n_source_support_issues"),
                       ("any_issue", "n_issues")]:
        r = {"issue_type": label}
        for cond, sub in (("with_rag", w), ("without_rag", wo)):
            r[f"{cond}__n_issues"] = int(sub[col].sum())
            r[f"{cond}__n_cases_with"] = int((sub[col] > 0).sum())
        r["diff_n_cases_with (without - with)"] = r["without_rag__n_cases_with"] - r["with_rag__n_cases_with"]
        iss_rows.append(r)
    issues_df = pd.DataFrame(iss_rows)

    # ---- routing (item 6) ----
    route_dist = pd.DataFrame([
        {"condition": cond,
         "accepted": int((sub.case_status == "accepted").sum()),
         "needs_review": int((sub.case_status == "needs_review").sum()),
         "flagged": int((sub.case_status == "flagged").sum())}
        for cond, sub in (("with_rag", w.loc[pairs]), ("without_rag", wo.loc[pairs]))
    ])
    trans = Counter((w.loc[p, "case_status"], wo.loc[p, "case_status"]) for p in pairs)
    trans_rows = [{"with_rag": a, "without_rag": b, "n_pairs": c} for (a, b), c in sorted(trans.items())]
    trans_df = pd.DataFrame(trans_rows)
    improved = sum(1 for p in pairs if _ROUTE_RANK[w.loc[p, "case_status"]] > _ROUTE_RANK[wo.loc[p, "case_status"]])
    worsened = sum(1 for p in pairs if _ROUTE_RANK[w.loc[p, "case_status"]] < _ROUTE_RANK[wo.loc[p, "case_status"]])
    unchanged = n_pairs - improved - worsened
    route_summary = pd.DataFrame([
        {"routing_vs_rag": "better WITH rag (with_rag rank > without_rag)", "n_pairs": improved},
        {"routing_vs_rag": "unchanged", "n_pairs": unchanged},
        {"routing_vs_rag": "worse WITH rag", "n_pairs": worsened},
    ])

    # ---- artefact confirmation on all without_rag pairs (item 8) ----
    wo_p = wo.loc[pairs]
    artefact = pd.DataFrame([
        {"check": "hallucinated [S#]/[W#] tag in answer text", "without_rag_pairs_affected": int(wo_p["hallucinated_cite_tag_in_answer"].sum()), "expected": 0},
        {"check": "deterministic citations detected (det.citations non-empty)", "without_rag_pairs_affected": int((wo_p["det_citations"].map(len) > 0).sum()), "expected": 0},
        {"check": "unknown/fabricated citation ids (det.unknown_citations)", "without_rag_pairs_affected": int((wo_p["det_unknown_citations"].map(len) > 0).sum()), "expected": 0},
        {"check": "unsupported_claim / high severity", "without_rag_pairs_affected": int((wo_p["n_unsupported_claim_high"] > 0).sum()), "expected": 0},
        {"check": "summary_inline_citation repair issue", "without_rag_pairs_affected": int(wo_p["repair_summary_inline_citation"].sum()), "expected": 0},
        {"check": "sources_used repair issue", "without_rag_pairs_affected": int(wo_p["repair_sources_used"].sum()), "expected": 0},
        {"check": "missing_required_section issue (any)", "without_rag_pairs_affected": int(wo_p["has_missing_required_section"].sum()), "expected": 0},
        {"check": "NO_SOURCES_NOTE parsed as a citation", "without_rag_pairs_affected": int(wo_p["note_flagged_as_citation"].sum()), "expected": 0},
        {"check": "retrieved_sources_count != 0", "without_rag_pairs_affected": int((wo_p["retrieved_sources_count"] != 0).sum()), "expected": 0},
    ])

    write_sheets({
        "Baseline_Pairs": pairs_df,
        "Baseline_Scores": pd.DataFrame([
            {"condition": c, "n": int(len(sub)),
             **{f"{m}__mean": round(float(sub[m].astype(float).mean()), 3) for m in score_metrics},
             **{f"{m}__median": round(float(sub[m].astype(float).median()), 3) for m in score_metrics}}
            for c, sub in (("with_rag", w.loc[pairs]), ("without_rag", wo.loc[pairs]))
        ]),
        "Baseline_Stats": stats_df,
        "Baseline_Issues": issues_df,
        "Baseline_Routing": route_dist,
        "Baseline_Routing_Xtab": trans_df,
        "Baseline_Routing_Summary": route_summary,
        "Baseline_Artifact_Check": artefact,
    })

    def _g(m):
        r = stats_df[stats_df.metric == m]
        return r.iloc[0].to_dict() if not r.empty else {}

    update_summary({"baseline_no_rag_analysis": {
        "n_pairs": n_pairs,
        "direction": "WITH_RAG minus WITHOUT_RAG",
        "scipy_available": SCIPY,
        "retrieved_sources_count_without_rag": 0,
        "overall_score": _g("overall_score"),
        "groundedness": _g("score_groundedness"),
        "arithmetic_consistency": _g("score_arithmetic_consistency"),
        "actionability": _g("score_actionability"),
        "clarity": _g("score_clarity"),
        "safety_ethics": _g("score_safety_ethics"),
        "routing": {"with_rag": route_dist.iloc[0].to_dict(), "without_rag": route_dist.iloc[1].to_dict(),
                    "better_with_rag": improved, "unchanged": unchanged, "worse_with_rag": worsened},
        "issue_cases_diff_without_minus_with": {r["issue_type"]: r["diff_n_cases_with (without - with)"] for r in iss_rows},
        "artifact_check_all_zero": bool((artefact["without_rag_pairs_affected"] == 0).all()),
    }})
    _fig_baseline(pairs_df, trans_df)


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------

def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _fig_no_api(df: pd.DataFrame, sev_sheets: Dict[str, pd.DataFrame], thr: pd.DataFrame) -> None:
    plt = _mpl()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # issue type x severity
    its = sev_sheets["Issue_Type_Severity"].set_index("issue_type")[["low", "medium", "high"]]
    ax = its.plot(kind="bar", stacked=True, figsize=(9, 5), color=["#5B8FF9", "#F6BD16", "#E8684A"])
    ax.set_ylabel("number of issues (500-case dataset)")
    ax.set_title("Evaluator-detected issues by type and severity")
    ax.figure.tight_layout()
    ax.figure.savefig(FIG_DIR / "fig_issue_type_severity.png", dpi=150)
    plt.close(ax.figure)

    # threshold sensitivity
    fig, axx = plt.subplots(figsize=(9, 5))
    x = range(len(thr))
    axx.bar([i - 0.25 for i in x], thr["accepted"], width=0.25, label="accepted", color="#5B8FF9")
    axx.bar([i for i in x], thr["needs_review"], width=0.25, label="needs_review", color="#F6BD16")
    axx.bar([i + 0.25 for i in x], thr["flagged"], width=0.25, label="flagged", color="#E8684A")
    axx.set_xticks(list(x))
    axx.set_xticklabels(thr["scenario"], rotation=20, ha="right", fontsize=8)
    axx.set_ylabel("cases")
    axx.set_title("Routing sensitivity to acceptance / flag thresholds (post-hoc)")
    axx.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_threshold_sensitivity.png", dpi=150)
    plt.close(fig)

    # subgroup mean overall_score by employment
    emp = df.groupby("employment_status")["overall_score"].agg(["mean", "count"])
    fig, axx = plt.subplots(figsize=(8, 5))
    axx.bar(emp.index.astype(str), emp["mean"], color="#5B8FF9")
    axx.set_ylim(7.5, 9.5)
    axx.set_ylabel("mean overall_score")
    axx.set_title("Mean overall_score by employment status (exploratory)")
    for i, (mn, c) in enumerate(zip(emp["mean"], emp["count"])):
        axx.text(i, mn + 0.02, f"n={c}", ha="center", fontsize=8)
    axx.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_subgroup_employment.png", dpi=150)
    plt.close(fig)
    print(f"[fig] wrote 3 no-API figures -> {FIG_DIR.relative_to(REPO_ROOT)}")


def _fig_repeated(prof_df: pd.DataFrame, dim_df: pd.DataFrame) -> None:
    plt = _mpl()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, axx = plt.subplots(figsize=(8, 5))
    axx.hist(prof_df["score_sd"], bins=20, color="#5B8FF9", edgecolor="white")
    axx.set_xlabel("within-profile SD of overall_score (5 repeats)")
    axx.set_ylabel("number of profiles")
    axx.set_title("Repeated-run stability: within-profile overall_score SD")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_repeated_score_sd.png", dpi=150)
    plt.close(fig)

    fig, axx = plt.subplots(figsize=(6, 5))
    ident = float(prof_df["all_repeats_identical_routing"].mean() * 100)
    axx.bar(["identical routing\nin all 5 repeats", "at least one\nrouting change"],
            [ident, 100 - ident], color=["#5AA454", "#E8684A"])
    axx.set_ylabel("% of profiles")
    axx.set_ylim(0, 100)
    axx.set_title("Repeated-run routing stability")
    for i, v in enumerate([ident, 100 - ident]):
        axx.text(i, v + 1, f"{v:.1f}%", ha="center")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_repeated_routing_stability.png", dpi=150)
    plt.close(fig)

    dd = dim_df[dim_df["dimension"] != "overall_score"]
    fig, axx = plt.subplots(figsize=(8, 5))
    axx.bar(dd["dimension"].str.replace("score_", ""), dd["mean_within_profile_sd"], color="#5B8FF9")
    axx.set_ylabel("mean within-profile SD")
    axx.set_title("Repeated-run variability by rubric dimension")
    axx.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_repeated_dimension_var.png", dpi=150)
    plt.close(fig)
    print(f"[fig] wrote 3 repeated-run figures -> {FIG_DIR.relative_to(REPO_ROOT)}")


def _fig_baseline(pairs_df: pd.DataFrame, trans_df: pd.DataFrame) -> None:
    """Item 10: paired overall_score, paired groundedness, diff histogram, routing transitions."""
    if pairs_df.empty:
        return
    plt = _mpl()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    def _paired_plot(metric: str, title: str, fname: str) -> None:
        wv = pairs_df[f"{metric}_with"].astype(float).to_numpy()
        wov = pairs_df[f"{metric}_without"].astype(float).to_numpy()
        fig, axx = plt.subplots(figsize=(6, 5))
        jx = rng.normal(0, 0.03, size=len(wv))
        for i in range(len(wv)):
            axx.plot([1 + jx[i], 2 + jx[i]], [wv[i], wov[i]], color="#B0B0B0", lw=0.6, alpha=0.6, zorder=1)
        axx.scatter(np.full_like(wv, 1.0) + jx, wv, color="#5B8FF9", s=26, zorder=2, label="WITH RAG")
        axx.scatter(np.full_like(wov, 2.0) + jx, wov, color="#E8684A", s=26, zorder=2, label="WITHOUT RAG")
        axx.plot([0.85, 1.15], [wv.mean(), wv.mean()], color="#1C3D8C", lw=2)
        axx.plot([1.85, 2.15], [wov.mean(), wov.mean()], color="#8C2A1C", lw=2)
        axx.set_xticks([1, 2]); axx.set_xticklabels(["WITH RAG", "WITHOUT RAG"])
        axx.set_ylabel(metric.replace("score_", ""))
        axx.set_title(title)
        fig.tight_layout(); fig.savefig(FIG_DIR / fname, dpi=150); plt.close(fig)

    _paired_plot("overall_score", "No-RAG baseline: paired overall_score (n=%d)" % len(pairs_df),
                 "fig_baseline_paired_overall.png")
    _paired_plot("score_groundedness", "No-RAG baseline: paired groundedness (n=%d)" % len(pairs_df),
                 "fig_baseline_paired_groundedness.png")

    diff = pairs_df["overall_score_diff"].astype(float).dropna().to_numpy()
    fig, axx = plt.subplots(figsize=(7, 5))
    bins = np.arange(np.floor(diff.min()) - 0.5, np.ceil(diff.max()) + 1.5, 1)
    axx.hist(diff, bins=bins, color="#5B8FF9", edgecolor="white")
    axx.axvline(0, color="grey", ls="--", lw=1)
    axx.axvline(diff.mean(), color="#E8684A", lw=2, label=f"mean {diff.mean():.2f}")
    axx.set_xlabel("paired overall_score difference  (WITH_RAG - WITHOUT_RAG)")
    axx.set_ylabel("number of pairs")
    axx.set_title("Distribution of paired overall_score differences")
    axx.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_baseline_diff_hist.png", dpi=150); plt.close(fig)

    if not trans_df.empty:
        order = ["accepted", "needs_review", "flagged"]
        mat = np.zeros((3, 3), dtype=int)
        for _, r in trans_df.iterrows():
            mat[order.index(r["with_rag"]), order.index(r["without_rag"])] = r["n_pairs"]
        fig, axx = plt.subplots(figsize=(6, 5))
        im = axx.imshow(mat, cmap="Blues")
        axx.set_xticks(range(3)); axx.set_xticklabels(order, rotation=20, ha="right")
        axx.set_yticks(range(3)); axx.set_yticklabels(order)
        axx.set_xlabel("WITHOUT RAG route"); axx.set_ylabel("WITH RAG route")
        axx.set_title("Routing transitions (n pairs)")
        for i in range(3):
            for j in range(3):
                axx.text(j, i, str(mat[i, j]), ha="center", va="center",
                         color="white" if mat[i, j] > mat.max() / 2 else "black")
        fig.colorbar(im, ax=axx, fraction=0.046)
        fig.tight_layout(); fig.savefig(FIG_DIR / "fig_baseline_routing_transitions.png", dpi=150); plt.close(fig)

    print(f"[fig] wrote 4 baseline figures -> {FIG_DIR.relative_to(REPO_ROOT)}")


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------

def run_all_no_api() -> None:
    df = frame_original()
    summary: Dict[str, Any] = {
        "_meta": {
            "n_original_cases": int(len(df)),
            "slice_rule": "first 500 records of runs/runs.jsonl sorted by ts_unix (record 501 excluded)",
            "scipy_available": SCIPY,
            "routing_counts": df["case_status"].value_counts().to_dict(),
        }
    }
    sheets: Dict[str, pd.DataFrame] = {}

    sheets.update(analysis_overview(df))

    s, sm = analysis_issue_severity(df); sheets.update(s); summary.update(sm)
    sev_sheets = s
    s, sm = analysis_penalty_audit(); sheets.update(s); summary.update(sm)
    s, sm = analysis_threshold_sensitivity(df); sheets.update(s); summary.update(sm)
    thr = s["Threshold_Sensitivity"]
    s, sm = analysis_rag_config(); sheets.update(s); summary.update(sm)
    s, sm = analysis_subgroups(df); sheets.update(s); summary.update(sm)
    s, sm = analysis_budget_generation(df); sheets.update(s); summary.update(sm)
    s, sm = analysis_calibration_provenance(); sheets.update(s); summary.update(sm)
    s, sm = analysis_arithmetic_safeguard(df); sheets.update(s); summary.update(sm)
    s, sm = analysis_sample_interaction(); sheets.update(s); summary.update(sm)

    write_sheets(sheets)
    update_summary(summary)
    _fig_no_api(df, sev_sheets, thr)
    print("\n[done] no-API analyses complete.")
    print(f"  tables : {XLSX_PATH.relative_to(REPO_ROOT)}  ({len(sheets)} sheets)")
    print(f"  summary: {SUMMARY_PATH.relative_to(REPO_ROOT)}")
    print(f"  figures: {FIG_DIR.relative_to(REPO_ROOT)}")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Revision 2026 no-API analytics")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("all", help="run every no-API analysis (TASK E,F,G,H,I,J,K,L,M)")
    pr = sub.add_parser("analyze-repeated", help="summarise outputs/repeated_runs.jsonl (TASK B)")
    pr.add_argument("--input", default=str(REPEATED_JSONL))
    pb = sub.add_parser("analyze-baseline", help="summarise outputs/baseline_no_rag.jsonl (TASK D)")
    pb.add_argument("--input", default=str(BASELINE_JSONL))
    args = p.parse_args(argv)

    if args.cmd in (None, "all"):
        run_all_no_api()
    elif args.cmd == "analyze-repeated":
        analyze_repeated(Path(args.input))
    elif args.cmd == "analyze-baseline":
        analyze_baseline(Path(args.input))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
