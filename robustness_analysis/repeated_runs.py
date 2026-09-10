"""
experiments/repeated_runs.py
==============================

Two additional experiments that DO call the LLM APIs. Both re-use the FIXED inputs
of original cases (first 500 records of runs/runs.jsonl, chronological). The
RefugeeAgent is never run. runs/runs.jsonl is never written.

    repeated  (TASK A)  reproducibility / stability
        For each selected profile, replay the SAME fixed inputs
        (profile_json, profile_text, user_query, clarifying_qa) N times through
            ConsultantAgent.final()  ->  EvaluatorAgent.evaluate()  ->  routing
        Default 100 profiles x 5 repeats.

    baseline  (TASK C)  paired no-RAG ablation
        For each selected profile, run the SAME fixed inputs under two conditions:
            with_rag     -> normal ConsultantAgent (retrieval on)
            without_rag  -> ConsultantAgent with retrieval disabled (experimental
                            subclass; nothing in agents/ is changed)
        Everything else (model, temperature, structured output, deterministic
        safeguards, Evaluator, penalties, routing) is identical.
        Default 50 profiles x 2 conditions.

Usage
-----
    # small smoke tests (safe to run now)
    python experiments/repeated_runs.py repeated --test
    python experiments/repeated_runs.py baseline --test

    # full experiments (only after explicit go-ahead)
    python experiments/repeated_runs.py repeated --n-profiles 100 --n-repeats 5 --seed 42
    python experiments/repeated_runs.py baseline --n-pairs 50 --seed 42
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import re
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

import analysis as ra  # noqa: E402  (chronological loader + routing recompute)

OUT_DIR = REPO_ROOT / "experiments" / "outputs"
ACCEPT_THRESHOLD = 9   # from run_langgraph.py
FLAG_BELOW = 7         # from run_langgraph.py


# ----------------------------------------------------------------------------
# runtime
# ----------------------------------------------------------------------------

def _build_runtime():
    """Load the consultant FAISS index and instantiate Consultant + Evaluator once."""
    from src.tools import build_or_load_index, RagConfig
    from agents.consultant import ConsultantAgent, ConsultantAgentConfig
    from agents.evaluator import EvaluatorAgent, EvaluatorConfig

    cfg = RagConfig(chunk_size=1500, chunk_overlap=100)  # identical to run_langgraph.make_runtime()
    vs = build_or_load_index(
        books_folder=str(REPO_ROOT / "books_consultant"),
        index_folder=str(REPO_ROOT / "indexes" / "consultant"),
        cfg=cfg,
    )
    consultant = ConsultantAgent(vs, ConsultantAgentConfig())
    evaluator = EvaluatorAgent(EvaluatorConfig())
    return vs, consultant, evaluator, cfg


# ----------------------------------------------------------------------------
# WITHOUT-RAG control condition (Variant 1: citation-contract adjustment)
# ----------------------------------------------------------------------------
#
# Same sections / types as agents.consultant._FinalAnswerStruct. Only the
# citation mandates are removed from the field descriptions, because the no-RAG
# condition has no retrieved sources to cite. Names, types and semantic meaning
# of the six structured sections are unchanged.

class _NoRagQuickBudgetCheck(BaseModel):
    monthly_income: float = Field(description="Client's monthly income in PLN")
    monthly_expenses_total: float = Field(description="Client's total monthly expenses in PLN")
    monthly_surplus: float = Field(description="monthly_income minus monthly_expenses_total")


class _NoRagFinalAnswerStruct(BaseModel):
    summary: str = Field(
        description="2-4 sentence narrative summary of the retirement advice. "
                    "Do not use or invent [S#] or [W#] citation tags."
    )
    quick_budget_check: _NoRagQuickBudgetCheck = Field(
        description="Budget numbers taken directly from the client profile."
    )
    suggested_monthly_retirement_saving_amount: str = Field(
        description="Specific PLN amount or range the client should save monthly for retirement "
                    "(e.g. 'PLN 200-300 per month'). If surplus <= 0 write "
                    "'PLN 0 for now; stabilise budget first'. "
                    "Add '(general rule of thumb, not Poland-specific)' where relevant."
    )
    retirement_related_options_in_poland: List[str] = Field(
        description="2-4 concrete retirement-related options available in Poland for this client "
                    "(e.g. ZUS contributions, voluntary ZUS, checking the ZUS projection). "
                    "Do not add [S#] or [W#] citation tags."
    )
    next_steps: List[str] = Field(
        description="Exactly 3-5 actionable next steps for this specific client, ordered by priority. "
                    "Do not add [S#] or [W#] citation tags."
    )
    sources_used: List[str] = Field(
        description="Leave this empty; no external sources are available in this condition."
    )


_CITATION_TAG_RE = re.compile(r"\s*\[(?:S|W)\d+\]")


class NoRagConsultantAgent:
    """Experimental WITHOUT-RAG control condition (Variant 1).

    Re-implements ConsultantAgent.final() faithfully but:
      * no retrieval at all -> sources = []  (no vectorstore / live web / Tavily /
        source ids / retrieved evidence reach the model);
      * the prompt + structured-output schema drop every requirement that cannot be
        met without sources ("use retrieved sources", "place [S#] tags", "must
        contain inline citations", "each item must end with [S#]", ZUS-with-[S#]);
      * the citation block is replaced with an explicit no-sources instruction;
      * sources_used is forced empty and any stray [S#]/[W#] tag is removed so no
        fabricated citation ids can appear;
      * the base section-repair is reused unchanged EXCEPT the citation-specific
        'summary_inline_citation' mark is dropped.

    Everything else is the base agent, verbatim: same model, same temperature,
    same structured-output engine (base.llm), same _normalize_nested_summary,
    _remove_unsupported_private_programs, _ensure_required_final_sections,
    deterministic arithmetic overwrite, _format_final_answer_text. The Evaluator,
    its rubric, its penalties and the routing thresholds are NOT touched.
    """

    NO_SOURCE_INSTRUCTION = (
        "No external sources are available in this experimental condition. Do not use or invent "
        "[S#] or [W#] citation tags, URLs, source ids or any fabricated reference. Provide the "
        "recommendation using only the information available to the model and the supplied client "
        "profile. Present Poland-specific legal, pension, tax and social-insurance facts cautiously "
        "and advise the client to verify them with ZUS or another relevant official source where "
        "needed. Leave sources_used empty."
    )

    # Plain-text marker written into the sources_used section for the no-RAG condition.
    # It is NOT a source and NOT a citation: no [S#]/[W#] tag, no URL, no id, no reference.
    # Its only purpose is to stop the (unchanged) Evaluator from treating an empty
    # sources_used list as a missing required section (a pure plumbing penalty, not a
    # RAG effect). retrieved_sources_count for this condition is still 0.
    NO_SOURCES_NOTE = "No external sources were retrieved in this experimental condition."

    def __init__(self, base):
        self._base = base
        self._llm_struct = base.llm.with_structured_output(
            _NoRagFinalAnswerStruct, method="function_calling"
        )

    # -- helpers --------------------------------------------------------------

    def _strip_citation_tags(self, obj):
        """Recursively remove any [S#]/[W#] tag from strings (belt-and-braces for item 5/11)."""
        if isinstance(obj, str):
            return _CITATION_TAG_RE.sub("", obj).strip()
        if isinstance(obj, list):
            return [self._strip_citation_tags(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self._strip_citation_tags(v) for k, v in obj.items()}
        return obj

    def _repair_no_rag(self, parsed: Dict[str, Any], profile_json: Dict[str, Any]) -> List[Dict[str, str]]:
        """Base section repair with sources=[], then drop ONLY the citation-specific
        'summary_inline_citation' mark (item 6). All other structural repairs kept."""
        issues = self._base._ensure_required_final_sections(parsed, profile_json, [])
        return [it for it in issues if str(it.get("field")) != "summary_inline_citation"]

    # -- public -------------------------------------------------------------

    def final(self, profile_text, user_query, clarifying_qa, profile_json):
        sources: List[Dict[str, Any]] = []  # items 1 & 8: nothing retrieval-derived

        qa_block = "\n".join(
            f"Q: {item.get('question', '')}\nA: {item.get('answer', '')}"
            for item in clarifying_qa
        ).strip()

        income = float(profile_json.get("income", 0.0))
        expenses = sum(
            float(profile_json.get(k, 0.0))
            for k in ("housing", "utilities", "food", "transport", "healthcare", "other", "remittances")
        )
        surplus = round(income - expenses, 2)

        # identical to WITH-RAG: profile-derived hints (Rodzina 800+/MOPS/ZUS gap), not retrieval-derived
        context_hints = self._base._profile_context_hints(profile_json)
        hints_block = f"\nPROFILE CONTEXT HINTS (act on these):\n{context_hints}\n" if context_hints else ""

        prompt = f"""
You are a cautious financial guidance assistant.

Write a final answer for the client.

Important rules:
- {self.NO_SOURCE_INSTRUCTION}
- If something is a general financial heuristic, explicitly label it "(general rule of thumb, not Poland-specific)".
- Do not mention PPK, IKE, IKZE, OFE, or any other Polish pension vehicle.
- Do NOT confuse "social pension" (invalidity benefit for those unable to work) with the regular "old-age pension". They may share the same minimum amount (PLN 1,780.96) but are different benefits with different eligibility conditions.
- Avoid exact minimum old-age pension amounts. Tell the client to check the current amount on the official ZUS website.
- Use the client clarifications.
- Keep the answer practical and clear.
- Return ONLY valid JSON as an object with EXACTLY these keys:
{{
  "summary": "...",
  "quick_budget_check": {{
    "monthly_income": {income},
    "monthly_expenses_total": {expenses},
    "monthly_surplus": {surplus}
  }},
  "suggested_monthly_retirement_saving_amount": "...",
  "retirement_related_options_in_poland": ["...", "..."],
  "next_steps": ["...", "...", "..."],
  "sources_used": []
}}

Client profile:
{profile_text}

Structured profile:
{json.dumps(profile_json, ensure_ascii=False)}
{hints_block}
User query:
{user_query}

Clarifying Q/A:
{qa_block}

Known arithmetic:
- monthly_income = {income}
- monthly_expenses_total = {expenses}
- monthly_surplus = {surplus}

Retrieved sources:
No external sources are available in this experimental condition.
""".strip()

        try:
            result = self._llm_struct.invoke(prompt)
            parsed = result.model_dump() if hasattr(result, "model_dump") else dict(result)
        except Exception as e:
            print(f"[NoRagConsultantAgent] structured output failed ({e}), falling back to json mode")
            resp = self._base.llm_json.invoke(prompt)
            raw = getattr(resp, "content", str(resp)).strip()
            parsed = self._base._safe_parse_structured_answer(raw)

        parsed = self._base._normalize_nested_summary(parsed)
        parsed = self._base._remove_unsupported_private_programs(parsed, sources)

        # No fabricated citations. sources_used carries a plain-text no-source marker
        # (not a source, not a citation) BEFORE repair, so the base repair does not add
        # a `sources_used` mark and the Evaluator does not raise `missing_required_section`
        # purely because the list is empty. retrieved_sources_count stays 0.
        parsed["sources_used"] = [self.NO_SOURCES_NOTE]
        repair_issues = self._repair_no_rag(parsed, profile_json)

        qb = parsed.get("quick_budget_check") or parsed.get("Quick budget check")
        if not isinstance(qb, dict):
            qb = {}
        qb["monthly_income"] = income
        qb["monthly_expenses_total"] = expenses
        qb["monthly_surplus"] = surplus
        parsed["quick_budget_check"] = qb

        parsed = self._strip_citation_tags(parsed)
        parsed["sources_used"] = [self.NO_SOURCES_NOTE]

        final_answer_text = self._base._format_final_answer_text(parsed)
        return {
            "final_answer": final_answer_text,
            "final_answer_struct": parsed,
            "sources": [],  # retrieved_sources_count == 0; the sentinel is not a source
            "repair_issues": repair_issues,
        }


# ----------------------------------------------------------------------------
# metadata
# ----------------------------------------------------------------------------

def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10
        ).stdout.strip() or "unavailable"
    except Exception:
        return "unavailable"


def _experiment_metadata(experiment: str, version: str, seed: int, rag_cfg) -> Dict[str, Any]:
    from agents.consultant import ConsultantAgentConfig
    from agents.evaluator import EvaluatorConfig
    from agents.refugee import RefugeeAgentConfig

    cc = ConsultantAgentConfig()
    ec = EvaluatorConfig()
    rc = RefugeeAgentConfig()

    def _d(x):
        return asdict(x) if is_dataclass(x) else dict(x)

    return {
        "experiment_name": experiment,
        "experiment_version": version,
        "selection_seed": seed,
        "python_version": platform.python_version(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit_sha": _git_sha(),
        "refugee_agent": "not executed (fixed inputs replayed from runs/runs.jsonl)",
        "refugee_agent_config_ref": {"model": rc.model, "seed_default": rc.seed},
        "consultant": {
            "model_name": cc.model_name,
            "temperature": cc.temperature,
            "max_clarifying_questions": cc.max_clarifying_questions,
        },
        "evaluator": {
            "model": ec.model,
            "temperature": ec.temperature,
            "tol_pln": ec.tol_pln,
            "penalty_math_error": ec.penalty_math_error,
            "penalty_missing_citations": ec.penalty_missing_citations,
            "penalty_missing_required_sections": ec.penalty_missing_required_sections,
            "penalty_untrusted_source": ec.penalty_untrusted_source,
            "penalty_private_program_hallucination": ec.penalty_private_program_hallucination,
            "penalty_unsupported_claim_high": ec.penalty_unsupported_claim_high,
            "penalty_unsupported_claim_medium": ec.penalty_unsupported_claim_medium,
        },
        "rag": {
            "embedding_model": rag_cfg.embeddings_model,
            "semantic_chunking": rag_cfg.semantic_chunking,
            "semantic_breakpoint_threshold_type": rag_cfg.semantic_breakpoint_threshold_type,
            "semantic_breakpoint_threshold_amount": rag_cfg.semantic_breakpoint_threshold_amount,
            "chunk_size": rag_cfg.chunk_size,
            "chunk_overlap": rag_cfg.chunk_overlap,
            "retrieval_top_k_consultant": 5,
        },
        "web": {
            "web_enabled": cc.web_enabled,
            "allowed_domains": list(cc.allowed_domains),
            "web_max_sources": cc.web_max_sources,
            "tavily_enabled": cc.tavily_enabled,
        },
        "routing": {
            "accept_threshold_overall_score": ACCEPT_THRESHOLD,
            "flag_below_overall_score": FLAG_BELOW,
            "accept_requires": "allow_to_show AND grounded_only_pass AND overall_score >= accept_threshold",
            "grounded_only_pass": "False if any high-severity unsupported_claim/math_error or medium+ "
                                  "missing_citation/missing_required_section/untrusted_source",
        },
        "consultant_config_full": _d(cc),
        "evaluator_config_full": _d(ec),
        "rag_config_full": _d(rag_cfg),
    }


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _fixed_inputs(rec: Dict[str, Any]) -> Dict[str, Any]:
    ref = rec["refugee"]
    return {
        "profile_json": copy.deepcopy(ref["profile_json"]),
        "profile_text": ref["profile_text"],
        "user_query": ref["user_query"],
        "clarifying_qa": copy.deepcopy(rec.get("refugee_clarifying_qa") or []),
    }


def _route(eval_out: Dict[str, Any]) -> str:
    f = eval_out.get("final", {})
    return ra.recompute_routing(
        f.get("overall_score"), bool(f.get("allow_to_show")), bool(f.get("grounded_only_pass")),
        ACCEPT_THRESHOLD, FLAG_BELOW,
    )


def _one_pass_once(consultant, evaluator, fi: Dict[str, Any]) -> Dict[str, Any]:
    final_out = consultant.final(fi["profile_text"], fi["user_query"], fi["clarifying_qa"], fi["profile_json"])
    eval_out = evaluator.evaluate(
        profile_json=fi["profile_json"],
        user_query=fi["user_query"],
        clarifying_qa=fi["clarifying_qa"],
        final_answer=final_out["final_answer"],
        final_sources=final_out["sources"],
        final_answer_struct=final_out.get("final_answer_struct"),
        repair_issues=final_out.get("repair_issues", []),
    )
    return {
        "consultant_final": {
            "final_answer": final_out["final_answer"],
            "final_answer_struct": final_out.get("final_answer_struct", {}),
            "sources": final_out["sources"],
            "repair_issues": final_out.get("repair_issues", []),
        },
        "evaluator": eval_out,
        "final": eval_out.get("final", {}),
        "case_status": _route(eval_out),
    }


def _one_pass(consultant, evaluator, fi: Dict[str, Any], max_attempts: int = 5) -> Dict[str, Any]:
    """Retry the full Consultant->Evaluator pass on any exception (transient LLM/JSON
    failures re-sample cleanly at temperature 0.1). Only after max_attempts does the
    case become an error stub so one bad response never aborts a 500-run experiment.
    Nothing in agents/ is modified; this is pure orchestration resilience."""
    last_exc: Exception = RuntimeError("no attempt made")
    for attempt in range(1, max_attempts + 1):
        try:
            return _one_pass_once(consultant, evaluator, copy.deepcopy(fi))
        except Exception as exc:  # noqa: BLE001 - deliberately broad; see docstring
            last_exc = exc
            print(f"      [pass attempt {attempt}/{max_attempts}] failed: {type(exc).__name__}: {str(exc)[:160]}")
            if attempt < max_attempts:
                time.sleep(5 * attempt)
    return {
        "consultant_final": {"final_answer": "", "final_answer_struct": {}, "sources": [], "repair_issues": []},
        "evaluator": {},
        "final": {},
        "case_status": "error",
        "error": f"{type(last_exc).__name__}: {str(last_exc)[:400]}",
    }


def _select(recs: List[Dict[str, Any]], n: int, seed: int) -> List[int]:
    import random
    idx = list(range(len(recs)))
    random.Random(seed).shuffle(idx)
    return sorted(idx[:n])


def _append(path: Path, obj: Dict[str, Any]) -> None:
    assert "runs/runs.jsonl" not in str(path).replace("\\", "/"), "refusing to write runs/runs.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------
# experiment: repeated runs
# ----------------------------------------------------------------------------

def _completed_pairs(out_path: Path) -> set:
    """(profile_repeat_group_id, repeat_id) already present in out_path (successful only)."""
    done = set()
    if not out_path.exists():
        return done
    with open(out_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            x = json.loads(line)
            if x.get("case_status") != "error":
                done.add((x.get("profile_repeat_group_id"), x.get("repeat_id")))
    return done


def run_repeated(n_profiles: int, n_repeats: int, seed: int, out_path: Path,
                 input_path: Path, resume: bool = False) -> None:
    recs = ra.load_original_500(input_path)
    chosen = _select(recs, n_profiles, seed)  # deterministic: gid -> same original profile on resume
    _, consultant, evaluator, rag_cfg = _build_runtime()
    meta = _experiment_metadata("repeated_runs", "1.0", seed, rag_cfg)
    meta["n_profiles"] = n_profiles
    meta["n_repeats"] = n_repeats
    meta["selected_original_run_ids"] = [recs[i]["run_id"] for i in chosen]
    meta["resumed"] = bool(resume)

    already = _completed_pairs(out_path) if resume else set()
    if not resume and out_path.exists():
        out_path.unlink()
    if resume:
        print(f"[resume] {len(already)} (group,repeat) pairs already complete in {out_path.name}")

    total = len(chosen) * n_repeats
    done = len(already)
    n_err = 0
    t0 = time.time()
    for gid, ridx in enumerate(chosen):
        src = recs[ridx]
        fi = _fixed_inputs(src)
        for rep in range(1, n_repeats + 1):
            if (gid, rep) in already:
                continue
            res = _one_pass(consultant, evaluator, fi)
            if res.get("case_status") == "error":
                n_err += 1
            rec = {
                "experiment": "repeated_runs",
                "original_run_id": src["run_id"],
                "profile_repeat_group_id": gid,
                "repeat_id": rep,
                "run_id": str(uuid.uuid4()),
                "ts_unix": int(time.time()),
                "selection_seed": seed,
                "fixed_inputs": fi,
                **res,
                "experiment_metadata": meta,
            }
            _append(out_path, rec)
            done += 1
            print(f"  [{done}/{total}] group={gid} repeat={rep} "
                  f"score={res.get('final', {}).get('overall_score')} route={res['case_status']} "
                  f"({time.time() - t0:.0f}s)")
    print(f"\n[done] repeated: {done}/{total} runs ({n_err} errors this session) -> {os.path.abspath(out_path)}")


# ----------------------------------------------------------------------------
# experiment: no-RAG baseline
# ----------------------------------------------------------------------------

def run_baseline(n_pairs: int, seed: int, out_path: Path, input_path: Path) -> None:
    recs = ra.load_original_500(input_path)
    chosen = _select(recs, n_pairs, seed)
    _, consultant, evaluator, rag_cfg = _build_runtime()
    norag = NoRagConsultantAgent(consultant)
    meta = _experiment_metadata("baseline_no_rag", "1.0", seed, rag_cfg)
    meta["n_pairs"] = n_pairs
    meta["conditions"] = ["with_rag", "without_rag"]
    meta["manipulation"] = "presence/absence of retrieval evidence only; all else identical"
    meta["selected_original_run_ids"] = [recs[i]["run_id"] for i in chosen]

    if out_path.exists():
        out_path.unlink()
    total = len(chosen) * 2
    done = 0
    t0 = time.time()
    for pid, ridx in enumerate(chosen):
        src = recs[ridx]
        fi = _fixed_inputs(src)
        for cond, agent in (("with_rag", consultant), ("without_rag", norag)):
            final_out = agent.final(fi["profile_text"], fi["user_query"],
                                    copy.deepcopy(fi["clarifying_qa"]), copy.deepcopy(fi["profile_json"]))
            eval_out = evaluator.evaluate(
                profile_json=fi["profile_json"], user_query=fi["user_query"],
                clarifying_qa=fi["clarifying_qa"], final_answer=final_out["final_answer"],
                final_sources=final_out["sources"], final_answer_struct=final_out.get("final_answer_struct"),
                repair_issues=final_out.get("repair_issues", []),
            )
            rec = {
                "experiment": "baseline_no_rag",
                "original_run_id": src["run_id"],
                "profile_pair_id": pid,
                "condition": cond,
                "run_id": str(uuid.uuid4()),
                "ts_unix": int(time.time()),
                "selection_seed": seed,
                "fixed_inputs": fi,
                "retrieved_sources_count": len(final_out["sources"]),  # 0 for without_rag; sentinel is NOT a source
                "consultant_final": {
                    "final_answer": final_out["final_answer"],
                    "final_answer_struct": final_out.get("final_answer_struct", {}),
                    "sources": final_out["sources"],
                    "repair_issues": final_out.get("repair_issues", []),
                },
                "evaluator": eval_out,
                "final": eval_out.get("final", {}),
                "case_status": _route(eval_out),
                "experiment_metadata": meta,
            }
            _append(out_path, rec)
            done += 1
            n_src = len(final_out["sources"])
            print(f"  [{done}/{total}] pair={pid} {cond:<11} "
                  f"score={rec['final'].get('overall_score')} route={rec['case_status']} "
                  f"n_sources={n_src} ({time.time() - t0:.0f}s)")
    print(f"\n[done] baseline: {done} runs -> {os.path.abspath(out_path)}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Revision 2026 API experiments (repeated runs / no-RAG baseline)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("repeated", help="TASK A: repeated-run reproducibility experiment")
    pr.add_argument("--input", default=str(REPO_ROOT / "runs" / "runs.jsonl"))
    pr.add_argument("--output", default=str(OUT_DIR / "repeated_runs.jsonl"))
    pr.add_argument("--n-profiles", type=int, default=100)
    pr.add_argument("--n-repeats", type=int, default=5)
    pr.add_argument("--seed", type=int, default=42)
    pr.add_argument("--test", action="store_true", help="2 profiles x 2 repeats -> repeated_runs_test.jsonl")
    pr.add_argument("--resume", action="store_true",
                    help="keep existing --output and only run (group,repeat) pairs still missing")

    pb = sub.add_parser("baseline", help="TASK C: paired no-RAG baseline")
    pb.add_argument("--input", default=str(REPO_ROOT / "runs" / "runs.jsonl"))
    pb.add_argument("--output", default=str(OUT_DIR / "baseline_no_rag.jsonl"))
    pb.add_argument("--n-pairs", type=int, default=50)
    pb.add_argument("--seed", type=int, default=42)
    pb.add_argument("--test", action="store_true", help="2 profiles x 2 conditions -> baseline_no_rag_test.jsonl")

    args = p.parse_args(argv)

    if args.cmd == "repeated":
        if args.test:
            run_repeated(2, 2, args.seed, OUT_DIR / "repeated_runs_test.jsonl", Path(args.input))
        else:
            run_repeated(args.n_profiles, args.n_repeats, args.seed, Path(args.output),
                         Path(args.input), resume=args.resume)
    elif args.cmd == "baseline":
        if args.test:
            # Smoke output history (all kept for the before/after audit trail):
            #   baseline_no_rag_test.jsonl      - v1, pre-fix (citation-contract artifacts)
            #   baseline_no_rag_test_v2.jsonl   - v2, citation-free prompt/schema, sources_used=[]
            #   baseline_no_rag_test_v2b.jsonl  - v2b, + NO_SOURCES_NOTE sentinel (current)
            run_baseline(2, args.seed, OUT_DIR / "baseline_no_rag_test_v2b.jsonl", Path(args.input))
        else:
            run_baseline(args.n_pairs, args.seed, Path(args.output), Path(args.input))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
