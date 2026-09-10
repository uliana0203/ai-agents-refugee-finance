"""
experiments/test_analysis.py
==============================

Lightweight checks for the robustness analytics + the scoring_audit patch.
Runs with either:

    python experiments/test_analysis.py
    pytest experiments/test_analysis.py -q

No API calls. No writes to runs/.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import analysis as ra  # noqa: E402

RUNS = REPO_ROOT / "runs" / "runs.jsonl"
EVALUATOR_PY = REPO_ROOT / "agents" / "evaluator.py"
OUT_DIR = REPO_ROOT / "experiments" / "outputs"


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _recompute_overall(rec: dict) -> int:
    ev = rec["evaluator"]
    det = ev.get("deterministic_checks", {})
    rub = ev.get("llm_rubric", {})
    scores = rub.get("scores", {}) or {}
    vals = [scores.get(k, 0) for k in ra.RUBRIC_DIMS]
    base = int(round(sum(vals) / len(vals))) if vals else 0
    overall = base
    for it in ra._issues_of(rec):
        _, _, p = ra._penalty_for_issue(it, bool(det.get("private_programs_flag")))
        overall -= p
    return max(0, min(10, overall))


# ----------------------------------------------------------------------------
# TASK F: scoring / routing invariance
# ----------------------------------------------------------------------------

def test_scoring_formula_reproduces_stored_overall_score():
    recs = ra.load_original_500()
    bad = [r["run_id"] for r in recs if _recompute_overall(r) != r["evaluator"]["final"]["overall_score"]]
    assert not bad, f"documented penalty formula diverges from stored overall_score for {len(bad)} cases"


def test_routing_formula_reproduces_stored_case_status():
    recs = ra.load_original_500()
    bad = []
    for r in recs:
        f = r["evaluator"]["final"]
        got = ra.recompute_routing(f["overall_score"], f["allow_to_show"], f["grounded_only_pass"])
        if got != r["case_status"]:
            bad.append(r["run_id"])
    assert not bad, f"recomputed routing diverges from stored case_status for {len(bad)} cases"


def test_scoring_audit_patch_is_additive_only():
    src = EVALUATOR_PY.read_text(encoding="utf-8")
    # overall_score must still be assigned from the clipped `overall`, and scoring_audit
    # must be built AFTER that assignment (so it cannot influence it).
    i_overall = src.index('llm_eval["overall_score"] = overall')
    i_audit = src.index('llm_eval["scoring_audit"] =')
    assert i_overall < i_audit, "scoring_audit must be assigned after overall_score"
    # routing inputs untouched
    assert '"overall_score": overall,' in src
    assert '"allow_to_show": allow,' in src
    assert '"grounded_only_pass": grounded_only_pass,' in src
    # no penalty weight literals were changed
    for key, val in {
        "penalty_math_error": 4, "penalty_missing_citations": 1,
        "penalty_missing_required_sections": 2, "penalty_untrusted_source": 1,
        "penalty_private_program_hallucination": 4,
        "penalty_unsupported_claim_high": 2, "penalty_unsupported_claim_medium": 1,
    }.items():
        assert re.search(rf"{key}: int = {val}\b", src), f"penalty weight {key} changed"


def test_scoring_audit_present_and_consistent_if_experiment_ran():
    for name in ("repeated_runs_test.jsonl", "repeated_runs.jsonl",
                 "baseline_no_rag_test.jsonl", "baseline_no_rag_test_v2.jsonl", "baseline_no_rag.jsonl"):
        p = OUT_DIR / name
        if not p.exists():
            continue
        rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        for x in rows:
            audit = x["evaluator"]["llm_rubric"].get("scoring_audit")
            assert audit is not None, f"{name}: scoring_audit missing"
            assert audit["final_score"] == x["evaluator"]["final"]["overall_score"], f"{name}: audit final_score != overall_score"
            assert audit["base_score"] + audit["total_penalty"] == audit["final_score_before_clipping"]
            assert max(0, min(10, audit["final_score_before_clipping"])) == audit["final_score"]


# ----------------------------------------------------------------------------
# original-data immutability
# ----------------------------------------------------------------------------

def test_runs_jsonl_untouched_by_analysis():
    before = (_sha(RUNS), RUNS.stat().st_size)
    ra.run_all_no_api()
    after = (_sha(RUNS), RUNS.stat().st_size)
    assert before == after, "runs/runs.jsonl changed during analysis!"


def test_no_revision_script_writes_to_runs():
    for name in ("analysis.py", "repeated_runs.py"):
        p = REPO_ROOT / "experiments" / name
        if not p.exists():
            continue
        src = p.read_text(encoding="utf-8")
        # no write-mode open() whose target literally names the original dataset
        for m in re.finditer(r'open\(\s*([^,]+),\s*["\'][wa]["\']', src):
            target = m.group(1)
            assert "runs/runs.jsonl" not in target and "runs\\\\runs.jsonl" not in target, \
                f"{name}: write-mode open on {target!r}"
        # revision scripts must not reuse the pipeline's append_jsonl on the runs path
        assert "append_jsonl" not in src, f"{name}: must not import the pipeline append_jsonl"
    rr = (REPO_ROOT / "experiments" / "repeated_runs.py")
    if rr.exists():
        assert "refusing to write runs/runs.jsonl" in rr.read_text(encoding="utf-8"), \
            "repeated_runs.py must keep the runs.jsonl write guard"


def test_output_paths_are_sandboxed():
    for pth in (ra.XLSX_PATH, ra.SUMMARY_PATH, ra.REPEATED_JSONL, ra.BASELINE_JSONL, ra.FIG_DIR):
        assert str(pth).replace("\\", "/").startswith(str((REPO_ROOT / "experiments" / "outputs")).replace("\\", "/"))


# ----------------------------------------------------------------------------
# severity aggregation logic
# ----------------------------------------------------------------------------

def test_case_max_severity():
    assert ra.case_max_severity([]) == "none"
    assert ra.case_max_severity(["low"]) == "low"
    assert ra.case_max_severity(["low", "low", "low"]) == "low"
    assert ra.case_max_severity(["low", "medium"]) == "medium"
    assert ra.case_max_severity(["medium", "high"]) == "high"
    assert ra.case_max_severity(["high", "low", "medium"]) == "high"
    assert ra.case_max_severity([None, "low"]) == "low"          # None -> treated as low
    assert ra.case_max_severity(["weird"]) == "low"              # unknown -> low


def test_severity_counts():
    assert ra.severity_counts([]) == {"low": 0, "medium": 0, "high": 0}
    assert ra.severity_counts(["low", "low", "medium"]) == {"low": 2, "medium": 1, "high": 0}
    assert ra.severity_counts(["high", "medium", "medium", "low"]) == {"low": 1, "medium": 2, "high": 1}


def test_issue_vs_case_counts_are_distinct():
    df = ra.frame_original()
    # number of ISSUES > number of CASES with an issue (a case may carry several)
    n_issue_records = int(df["n_issues"].sum())
    n_cases_with_issue = int((df["n_issues"] > 0).sum())
    assert n_issue_records >= n_cases_with_issue
    assert n_cases_with_issue <= len(df)


def test_no_high_severity_in_original_500():
    df = ra.frame_original()
    highs = sum(1 for sevs in df["issue_sevs"] for s in sevs if ra.normalise_severity(s) == "high")
    assert highs == 0, "unexpected high-severity issue in the 500-case dataset"


# ----------------------------------------------------------------------------
# repeated / baseline fixed-input equality (only if the 2x2 tests were run)
# ----------------------------------------------------------------------------

def _fixed_key(d: dict) -> str:
    fi = d.get("fixed_inputs", {})
    return json.dumps({k: fi.get(k) for k in ("profile_json", "profile_text", "user_query", "clarifying_qa")},
                      sort_keys=True, ensure_ascii=False)


def test_repeated_fixed_inputs_identical_across_repeats():
    p = OUT_DIR / "repeated_runs_test.jsonl"
    if not p.exists():
        print("  [skip] repeated_runs_test.jsonl not present yet")
        return
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    groups: dict = {}
    for r in rows:
        groups.setdefault(r["profile_repeat_group_id"], []).append(r)
    assert groups, "no groups in repeated test output"
    for gid, rs in groups.items():
        keys = {_fixed_key(r) for r in rs}
        assert len(keys) == 1, f"group {gid}: fixed_inputs differ across repeats"
        assert len({r["repeat_id"] for r in rs}) == len(rs), f"group {gid}: repeat_id not unique"
        assert len({r["run_id"] for r in rs}) == len(rs), f"group {gid}: run_id not unique"
        assert len({r["original_run_id"] for r in rs}) == 1


def test_baseline_pair_inputs_identical_across_conditions():
    files = [OUT_DIR / "baseline_no_rag_test.jsonl", OUT_DIR / "baseline_no_rag_test_v2.jsonl"]
    present = [p for p in files if p.exists()]
    if not present:
        print("  [skip] no baseline_no_rag_test*.jsonl present yet")
        return
    for p in present:
        rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        pairs: dict = {}
        for r in rows:
            pairs.setdefault(r["profile_pair_id"], []).append(r)
        assert pairs, f"{p.name}: no pairs in baseline test output"
        for pid, rs in pairs.items():
            keys = {_fixed_key(r) for r in rs}
            assert len(keys) == 1, f"{p.name} pair {pid}: fixed_inputs differ between with_rag / without_rag"
            conds = {r["condition"] for r in rs}
            assert conds == {"with_rag", "without_rag"}, f"{p.name} pair {pid}: conditions = {conds}"


# ----------------------------------------------------------------------------
# runner
# ----------------------------------------------------------------------------

def _run() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:  # pragma: no cover
            failed += 1
            print(f"ERROR {t.__name__}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
