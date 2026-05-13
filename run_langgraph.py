"""
LangGraph orchestration pipeline for synthetic refugee pension-advice case generation.

Each graph iteration runs one full case: RefugeeAgent generates a profile ->
ConsultantAgent produces a draft + clarifying questions -> RefugeeAgent answers ->
ConsultantAgent writes the final answer -> EvaluatorAgent scores it.
Cases are routed to save_accepted / save_needs_review / save_flagged nodes
and appended to runs/runs.jsonl for downstream EDA.

Usage:
    python run_langgraph.py              # generates N_CASES (default 5) cases
    N_CASES=50 python run_langgraph.py   # batch run
"""

from __future__ import annotations
import os
import time
import uuid
from typing import TypedDict, List, Dict, Any

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

import re

from agents.refugee import RefugeeAgent, RefugeeAgentConfig
from agents.consultant import ConsultantAgent, ConsultantAgentConfig
from agents.evaluator import EvaluatorAgent, EvaluatorConfig
from src.tools import build_or_load_index, RagConfig, append_jsonl

_PRIVATE_PROGRAMS = ("ppk", "ike", "ikze", "ofe")


def _check_draft_private_programs(draft_answer: str, sources: List[Dict[str, Any]]) -> List[str]:
    """Flag private program names that appear in the draft but lack support in the retrieved sources."""
    hay = " ".join((s.get("snippet") or "") for s in sources).lower()
    answer_lower = (draft_answer or "").lower()
    return [
        p for p in _PRIVATE_PROGRAMS
        if re.search(rf"\b{p}\b", answer_lower) and not re.search(rf"\b{p}\b", hay)
    ]


# -------------------------
# Load .env
# -------------------------
load_dotenv()


# -------------------------
# State
# -------------------------
class GraphState(TypedDict, total=False):
    n_cases: int
    out_jsonl: str
    i: int
    results: List[Dict[str, Any]]
    run_id: str
    ts_unix: int
    case_record: Dict[str, Any]
    case_status: str


# -------------------------
# Runtime (shared objects once)
# -------------------------

def make_runtime() -> Dict[str, Any]:
    """Build shared agent instances and FAISS indexes once per process."""
    cfg_refuge = RagConfig(chunk_size=1500, chunk_overlap=100)
    cfg_consult = RagConfig(chunk_size=1500, chunk_overlap=100)

    vs_refuge = build_or_load_index(
        books_folder="books_refuge",
        index_folder="indexes/refuge",
        cfg=cfg_refuge,
    )
    vs_consult = build_or_load_index(
        books_folder="books_consultant",
        index_folder="indexes/consultant",
        cfg=cfg_consult,
    )

    refugee = RefugeeAgent(vs_refuge, RefugeeAgentConfig())
    consultant = ConsultantAgent(vs_consult, ConsultantAgentConfig())
    evaluator = EvaluatorAgent(EvaluatorConfig())

    return {
        "refugee": refugee,
        "consultant": consultant,
        "evaluator": evaluator,
    }


RUNTIME = make_runtime()


# -------------------------
# Node: one full case
# -------------------------

def node_one_case(state: GraphState) -> GraphState:
    """Execute one full pipeline iteration and assemble the case record."""
    refugee: RefugeeAgent = RUNTIME["refugee"]
    consultant: ConsultantAgent = RUNTIME["consultant"]
    evaluator: EvaluatorAgent = RUNTIME["evaluator"]

    run_id = str(uuid.uuid4())
    ts = int(time.time())

    # 1) Refugee generates a synthetic case
    ref_out = refugee.run()

    # 2) Consultant draft + clarifying questions
    draft_out = consultant.draft(
        ref_out["profile_text"],
        ref_out["user_query"],
        ref_out["profile_json"],
    )
    clarifying_questions = (draft_out.get("clarifying_questions") or [])[:3]

    # 3) Refugee answers clarifying questions
    qa = refugee.answer_clarifying_questions(
        anchors=ref_out["anchors"],
        profile_json=ref_out["profile_json"],
        persona_json=ref_out["persona_json"],
        questions=clarifying_questions,
        max_q=len(clarifying_questions),
    )

    # 4) Consultant final answer
    final_out = consultant.final(
        ref_out["profile_text"],
        ref_out["user_query"],
        qa,
        ref_out["profile_json"],
    )

    # 5) Evaluator
    eval_out = evaluator.evaluate(
        profile_json=ref_out["profile_json"],
        user_query=ref_out["user_query"],
        clarifying_qa=qa,
        final_answer=final_out["final_answer"],
        final_sources=final_out["sources"],
        final_answer_struct=final_out.get("final_answer_struct"),
        repair_issues=final_out.get("repair_issues", []),
    )

    final_eval = eval_out.get("final", {}) if isinstance(eval_out, dict) else {}
    overall_score = int(final_eval.get("overall_score", 0))
    allow_to_show = bool(final_eval.get("allow_to_show", False))
    grounded_only_pass = bool(final_eval.get("grounded_only_pass", False))

    # Quality routing label
    if not allow_to_show or overall_score < 7:
        case_status = "flagged"
    elif allow_to_show and grounded_only_pass and overall_score >= 9:
        case_status = "accepted"
    else:
        case_status = "needs_review"

    record = {
        "run_id": run_id,
        "ts_unix": ts,
        "case_status": case_status,
        "refugee": ref_out,
        "consultant_draft": {
            "draft_answer": draft_out.get("draft_answer"),
            "draft_sources": draft_out.get("sources", []),
            "clarifying_questions": clarifying_questions,
            "draft_private_programs_warning": _check_draft_private_programs(
                draft_out.get("draft_answer", ""),
                draft_out.get("sources", []),
            ),
        },
        "refugee_clarifying_qa": qa,
        "consultant_final": {
            "final_answer": final_out["final_answer"],
            "final_answer_struct": final_out.get("final_answer_struct", {}),
            "final_sources": final_out["sources"],
            "repair_issues": final_out.get("repair_issues", []),
        },
        "evaluator": eval_out,
    }

    return {
        **state,
        "run_id": run_id,
        "ts_unix": ts,
        "case_record": record,
        "case_status": case_status,
    }


def route_case(state: GraphState) -> str:
    """Map case_status to the corresponding save node name."""
    status = state.get("case_status", "needs_review")
    if status == "accepted":
        return "accepted"
    if status == "flagged":
        return "flagged"
    return "needs_review"


def node_save_accepted(state: GraphState) -> GraphState:
    """Append an accepted case to the output JSONL and increment the counter."""
    record = state["case_record"]
    out_path = state.get("out_jsonl", "runs/runs.jsonl")
    append_jsonl(out_path, record)

    results = state.get("results", [])
    results.append(record)

    return {
        **state,
        "results": results,
        "i": int(state.get("i", 0)) + 1,
    }


def node_save_needs_review(state: GraphState) -> GraphState:
    """Append a needs-review case to the output JSONL and increment the counter."""
    record = state["case_record"]
    out_path = state.get("out_jsonl", "runs/runs.jsonl")
    append_jsonl(out_path, record)

    results = state.get("results", [])
    results.append(record)

    return {
        **state,
        "results": results,
        "i": int(state.get("i", 0)) + 1,
    }


def node_save_flagged(state: GraphState) -> GraphState:
    """Append a flagged case to the output JSONL and increment the counter."""
    record = state["case_record"]
    out_path = state.get("out_jsonl", "runs/runs.jsonl")
    append_jsonl(out_path, record)

    results = state.get("results", [])
    results.append(record)

    return {
        **state,
        "results": results,
        "i": int(state.get("i", 0)) + 1,
    }


def should_continue(state: GraphState) -> str:
    """Return 'loop' if the case quota has not been reached, otherwise 'end'."""
    n = int(state.get("n_cases", 1))
    i = int(state.get("i", 0))
    return "loop" if i < n else "end"


# -------------------------
# Build graph
# -------------------------

def build_graph():
    """Wire the LangGraph state machine and return the compiled graph."""
    g = StateGraph(GraphState)

    g.add_node("one_case", node_one_case)
    g.add_node("save_accepted", node_save_accepted)
    g.add_node("save_needs_review", node_save_needs_review)
    g.add_node("save_flagged", node_save_flagged)

    g.set_entry_point("one_case")

    g.add_conditional_edges(
        "one_case",
        route_case,
        {
            "accepted": "save_accepted",
            "needs_review": "save_needs_review",
            "flagged": "save_flagged",
        },
    )

    g.add_conditional_edges("save_accepted", should_continue, {"loop": "one_case", "end": END})
    g.add_conditional_edges("save_needs_review", should_continue, {"loop": "one_case", "end": END})
    g.add_conditional_edges("save_flagged", should_continue, {"loop": "one_case", "end": END})

    return g.compile()


GRAPH = build_graph()


# -------------------------
# Runner
# -------------------------

def run_langgraph(n_cases: int = 3, out_jsonl: str = "runs/runs.jsonl") -> List[Dict[str, Any]]:
    """Run the pipeline for n_cases iterations; returns all case records (including flagged)."""
    out_state = GRAPH.invoke(
        {
            "n_cases": n_cases,
            "out_jsonl": out_jsonl,
            "i": 0,
            "results": [],
        },
        config={"recursion_limit": n_cases * 3 + 10},
    )
    return out_state["results"]


if __name__ == "__main__":
    n = int(os.environ.get("N_CASES", 5))
    results = run_langgraph(n_cases=n, out_jsonl="runs/runs.jsonl")
    print("Generated cases:", len(results))
    last_final = results[-1]["evaluator"]["final"]
    print(
        "Last overall_score:",
        last_final.get("overall_score"),
        "| allow:",
        last_final.get("allow_to_show"),
        "| grounded_only_pass:",
        last_final.get("grounded_only_pass"),
    )
