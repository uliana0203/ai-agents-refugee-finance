"""
Evaluator agent that scores consultant answers on five dimensions using DeepSeek.

Evaluation is two-stage: deterministic checks run first (citation presence, arithmetic
mismatch, private-program hallucination), then an LLM rubric scores groundedness,
arithmetic_consistency, actionability, clarity, and safety_ethics (0-10 each).
Penalties from deterministic checks are subtracted from the LLM overall_score.
Answers with any high-severity issue get allow_to_show=False.
"""

from __future__ import annotations

import ast
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


@dataclass
class EvaluatorConfig:
    """Scoring thresholds and penalty weights; DeepSeek is used for low-cost LLM evaluation."""

    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com/v1"
    api_key_env: str = "DEEPSEEK_API_KEY"

    temperature: float = 0.1

    require_citations_for_poland_claims: bool = True
    min_citation_count: int = 1

    private_program_keywords: Tuple[str, ...] = ("ppk", "ike", "ikze")
    regulatory_triggers: Tuple[str, ...] = (
        "poland", "zus", "social insurance", "retirement age", "pension", "compulsory",
        "apply", "eligibility", "benefits", "scheme", "old-age pension"
    )

    tol_pln: float = 30.0

    penalty_math_error: int = 4
    penalty_missing_citations: int = 1
    penalty_missing_required_sections: int = 2
    penalty_untrusted_source: int = 1
    penalty_private_program_hallucination: int = 4
    allowed_source_domains: Tuple[str, ...] = (
        "gov.pl",
        "zus.pl",
        "podatki.gov.pl",
        "biznes.gov.pl",
        "euraxess.pl",
        "ec.europa.eu",
    )

    max_retries: int = 3
    retry_base_delay: float = 5.0
    penalty_unsupported_claim_high: int = 2
    penalty_unsupported_claim_medium: int = 1


class EvaluatorAgent:
    """Scores a consultant answer; returns deterministic_checks, llm_rubric, flagged_spans, and final verdict."""

    def __init__(self, cfg: EvaluatorConfig = EvaluatorConfig()):
        self.cfg = cfg
        api_key = os.getenv(cfg.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key env var: {cfg.api_key_env}")

        self.llm = ChatOpenAI(
            model=cfg.model,
            temperature=cfg.temperature,
            base_url=cfg.base_url,
            api_key=api_key,
        )

    # -------------------------
    # Helpers
    # -------------------------

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Remove ```json ... ``` wrappers that models sometimes add around JSON output."""
        text = (text or "").strip()
        if not text.startswith("```"):
            return text
        lines = text.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    @staticmethod
    def _extract_citation_tags(text: str) -> List[str]:
        """Find all [S#] and [W#] citation tags in the answer text."""
        return re.findall(r"\[(?:S|W)\d+\]", text or "")

    @staticmethod
    def _text_before_sources_used(text: str) -> str:
        """Return answer text before the final Sources used section for inline-citation checks."""
        if not text:
            return ""
        m = re.search(r"(?:^|\n)\s*(?:\d+\)\s*)?Sources used\s*:", text, flags=re.IGNORECASE)
        return text[:m.start()] if m else text

    @staticmethod
    def _parse_money_token(tok: str) -> Optional[float]:
        """Convert a PLN/EUR money string to float; returns None if unparseable."""
        tok = tok.strip().replace("€", "").replace("PLN", "").replace("pln", "").replace(",", "").strip()
        try:
            return float(tok)
        except ValueError:
            return None

    @staticmethod
    def _sources_text(sources: List[Dict[str, Any]]) -> str:
        """Concatenate all source snippets into a single string for keyword scanning."""
        return "\n".join([(s.get("snippet") or "") for s in sources])

    def _private_programs_in_sources(self, sources: List[Dict[str, Any]]) -> bool:
        """Return True if any of PPK/IKE/IKZE appear in source snippets, meaning the answer may cite them."""
        hay = self._sources_text(sources).lower()
        return any(re.search(rf"\b{re.escape(k)}\b", hay) for k in self.cfg.private_program_keywords)

    def _private_programs_in_answer(self, answer: str) -> List[str]:
        """Return which private program keywords (ppk/ike/ikze) appear in the answer."""
        a = (answer or "").lower()
        return [k for k in self.cfg.private_program_keywords if re.search(rf"\b{re.escape(k)}\b", a)]

    def _estimate_expenses_income_surplus(
        self, profile_json: Dict[str, Any]
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Recompute income, expenses, and surplus from profile_json as ground truth for mismatch detection."""
        try:
            income = float(profile_json["income"])
            expenses = (
                float(profile_json["housing"])
                + float(profile_json["utilities"])
                + float(profile_json["food"])
                + float(profile_json["transport"])
                + float(profile_json["healthcare"])
                + float(profile_json["other"])
                + float(profile_json.get("remittances", 0.0))
            )
            surplus = income - expenses
            return income, expenses, surplus
        except (KeyError, ValueError, TypeError):
            return None, None, None

    def _poland_claims_need_citations(self, answer: str) -> bool:
        """Return True if the answer contains regulatory trigger words requiring [S#] citation support."""
        a = (answer or "").lower()
        return any(t in a for t in self.cfg.regulatory_triggers)

    def _citations_only_in_sources_used(self, answer: str) -> bool:
        """True when citations exist but none appear before the Sources used section."""
        all_citations = self._extract_citation_tags(answer)
        inline_citations = self._extract_citation_tags(self._text_before_sources_used(answer))
        return bool(all_citations) and not bool(inline_citations)

    @staticmethod
    def _valid_source_ids(sources: List[Dict[str, Any]]) -> List[str]:
        """Extract the string IDs of all sources that have an id field."""
        return [str(s.get("id", "")) for s in sources if s.get("id")]

    def _unknown_citations(self, answer: str, sources: List[Dict[str, Any]]) -> List[str]:
        """Return citation tags that do not match any known source ID — indicates hallucinated references."""
        valid = set(self._valid_source_ids(sources))
        cited = self._extract_citation_tags(answer)
        return sorted(set([c for c in cited if c.strip("[]") not in valid]))

    def _untrusted_web_sources(self, sources: List[Dict[str, Any]]) -> List[str]:
        """Return Tavily/live-web final sources outside the curated allowed domains."""
        bad: List[str] = []
        for s in sources:
            source_type = str(s.get("source_type", ""))
            if source_type not in ("tavily_search", "live_web"):
                continue
            url = str(s.get("source", "")).lower()
            if url and not any(domain in url for domain in self.cfg.allowed_source_domains):
                bad.append(str(s.get("source", "")))
        return bad

    # -------------------------
    # Structured answer helpers
    # -------------------------

    @staticmethod
    def _safe_parse_structured_answer(raw: Any) -> Dict[str, Any]:
        """Parse JSON with json.loads -> ast.literal_eval fallback; returns {} on failure, never raises."""
        if isinstance(raw, dict):
            return raw

        if isinstance(raw, str):
            txt = raw.strip()
            if not txt:
                return {}

            # remove fenced code block
            m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", txt, flags=re.DOTALL | re.IGNORECASE)
            if m:
                txt = m.group(1).strip()

            if txt.startswith("{") and txt.endswith("}"):
                try:
                    parsed = json.loads(txt)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    pass

                try:
                    parsed = ast.literal_eval(txt)
                    if isinstance(parsed, dict):
                        return parsed
                except (ValueError, SyntaxError):
                    pass

        return {}

    def _get_structured_budget_values(
        self, final_answer_struct: Optional[Dict[str, Any]]
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Extract monthly_income, monthly_expenses_total, and monthly_surplus from the quick_budget_check block."""
        if not final_answer_struct:
            return None, None, None

        qb = (
            final_answer_struct.get("quick_budget_check")
            or final_answer_struct.get("Quick budget check")
        )

        if not isinstance(qb, dict):
            return None, None, None

        income = qb.get("monthly_income") or qb.get("Monthly income")
        expenses = qb.get("monthly_expenses_total") or qb.get("Monthly expenses total")
        surplus = qb.get("monthly_surplus") or qb.get("Monthly surplus")

        def _to_float(x):
            if x is None:
                return None
            if isinstance(x, (int, float)):
                return float(x)
            if isinstance(x, str):
                return self._parse_money_token(x)
            return None

        return _to_float(income), _to_float(expenses), _to_float(surplus)

    @staticmethod
    def _missing_required_sections(final_answer_struct: Optional[Dict[str, Any]]) -> List[str]:
        """Return required final-answer sections missing from structured output."""
        if not isinstance(final_answer_struct, dict):
            return [
                "summary",
                "quick_budget_check",
                "suggested_monthly_retirement_saving_amount",
                "retirement_related_options_in_poland",
                "next_steps",
                "sources_used",
            ]

        def has_value(*keys: str) -> bool:
            for key in keys:
                value = final_answer_struct.get(key)
                if isinstance(value, str) and value.strip():
                    return True
                if isinstance(value, list) and any(str(x).strip() for x in value):
                    return True
                if isinstance(value, dict) and value:
                    return True
                if value not in (None, "", [], {}):
                    return True
            return False

        missing: List[str] = []
        if not has_value("summary", "Summary"):
            missing.append("summary")

        qb = final_answer_struct.get("quick_budget_check") or final_answer_struct.get("Quick budget check")
        if not isinstance(qb, dict) or not all(
            qb.get(k) is not None or qb.get(k.replace("_", " ").title()) is not None
            for k in ("monthly_income", "monthly_expenses_total", "monthly_surplus")
        ):
            missing.append("quick_budget_check")

        if not has_value("suggested_monthly_retirement_saving_amount", "Suggested monthly retirement saving amount"):
            missing.append("suggested_monthly_retirement_saving_amount")
        if not has_value("retirement_related_options_in_poland", "Retirement-related options in Poland"):
            missing.append("retirement_related_options_in_poland")
        if not has_value("next_steps", "Next steps"):
            missing.append("next_steps")
        if not has_value("sources_used", "Sources used"):
            missing.append("sources_used")
        return missing

    # -------------------------
    # Safer number extraction from text
    # -------------------------

    def _extract_expenses_total_from_answer(self, answer: str) -> Optional[float]:
        """Regex fallback to extract total expenses from answer text when the structured block is absent."""
        a = answer or ""

        patterns = [
            r"expenses_total\s*[:=]\s*(?:PLN|€)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
            r"total monthly expenses(?: amount to| are| =|:)?\s*(?:PLN|€)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
            r"monthly expenses total(?: amount to| are| =|:)?\s*(?:PLN|€)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
            r"expenses of\s*(?:PLN|€)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
        ]
        for p in patterns:
            m = re.search(p, a, flags=re.IGNORECASE)
            if m:
                return self._parse_money_token(m.group(1))
        return None

    def _extract_surplus_from_answer(self, answer: str) -> Optional[float]:
        """Regex fallback to extract monthly surplus from answer text when the structured block is absent."""
        a = answer or ""

        patterns = [
            r"surplus\s*[:=]\s*(?:PLN|€)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
            r"surplus of\s*(?:PLN|€)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
            r"leaving you with a surplus of\s*(?:PLN|€)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
            r"monthly surplus of\s*(?:PLN|€)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
        ]
        for p in patterns:
            m = re.search(p, a, flags=re.IGNORECASE)
            if m:
                return self._parse_money_token(m.group(1))

        return None

    @staticmethod
    def _find_flagged_spans(answer: str) -> List[Dict[str, str]]:
        """Locate specific age claims and private program mentions to populate the flagged_spans audit trail."""
        a = answer or ""
        lower = a.lower()
        spans: List[Dict[str, str]] = []

        patterns = [
            ("specific_age_claim", r"\b60\s+years?\s+for\s+women\b"),
            ("specific_age_claim", r"\b65\s+years?\s+for\s+men\b"),
            ("specific_age_claim", r"\bstatutory retirement age.*?\b60\b"),
            ("specific_age_claim", r"\bstatutory retirement age.*?\b65\b"),
            ("private_program", r"\bppk\b"),
            ("private_program", r"\bike\b"),
            ("private_program", r"\bikze\b"),
        ]

        for kind, pat in patterns:
            for m in re.finditer(pat, lower):
                spans.append({"type": kind, "text": a[m.start():m.end()]})

        return spans

    # -------------------------
    # LLM rubric
    # -------------------------

    def _llm_rubric(
        self,
        profile_json: Dict[str, Any],
        user_query: str,
        qa: List[Dict[str, str]],
        final_answer: str,
        sources: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Call DeepSeek to score 5 dimensions (0-10 each) and collect issues; clamps all scores to [0,10]."""
        src_lines = []
        for s in sources:
            src_lines.append(
                f"{s.get('id')} | {s.get('source')} | page={s.get('page')} | {(s.get('snippet') or '')[:450].replace(chr(10), ' ')}"
            )
        src_text = "\n".join(src_lines)
        qa_text = "\n".join([f"- Q: {x.get('question','')}\n  A: {x.get('answer','')}" for x in qa]) if qa else "(none)"

        income_true, expenses_true, surplus_true = self._estimate_expenses_income_surplus(profile_json)
        has_deficit = (income_true is not None and expenses_true is not None and expenses_true > income_true)

        system = (
            "You are a strict evaluator of retirement-advice answers. "
            "You MUST score each dimension based ONLY on concrete evidence in the provided FINAL_ANSWER_TEXT. "
            "Do NOT give average or default scores - every score must differ from others unless the evidence is identical. "
            "Do not penalize general heuristics explicitly labeled as 'general rule of thumb, not Poland-specific'. "
            "\n\nEXEMPTIONS - do NOT penalize these as unsupported claims:\n"
            "1. Rodzina 800+ benefit (PLN 800/child/month) when the profile shows dependents > 0. "
            "This is a well-known Polish state program. No [S#] citation required.\n"
            "2. MOPS/GOPS social assistance eligibility when the profile shows unemployment + low income. "
            "No [S#] citation required.\n"
            "3. ZUS pension contribution gap when the profile shows unemployed/economically_inactive status. "
            "No [S#] citation required.\n"
            "4. Any claim explicitly labeled 'general rule of thumb, not Poland-specific'.\n"
            "\nDEFICIT PROFILES: If PROFILE_SURPLUS_SIGN is DEFICIT, do NOT penalize actionability "
            "for missing a concrete saving amount. Instead rate actionability by quality of concrete "
            "next steps: applying for Rodzina 800+, MOPS assistance, seeking employment, reducing "
            "specific discretionary expenses. Identifying the deficit correctly and recommending "
            "benefit applications is worth 7+ on actionability."
            "\n\nReturn ONLY valid JSON with no explanation outside the JSON."
        )

        profile_surplus_sign = "DEFICIT" if has_deficit else "SURPLUS_OR_BALANCED"

        user = f"""
You are evaluating this specific answer. Read it carefully before scoring.

USER_QUERY:
{user_query}

PROFILE_JSON (ground truth numbers):
{json.dumps(profile_json, ensure_ascii=False)}

PROFILE_SURPLUS_SIGN: {profile_surplus_sign}

CLARIFYING_QA:
{qa_text}

FINAL_ANSWER_TEXT (what you are evaluating):
{final_answer}

SOURCES_USED (what was available to the advisor):
{src_text}

Score each dimension 0-10 based ONLY on what is actually present or absent in FINAL_ANSWER_TEXT above.
For each score, you MUST quote a specific phrase from the answer as evidence.

Scoring criteria:
- groundedness (0-10): Are Poland-specific claims backed by [S#] citations? Deduct 2 per unsupported factual claim about Polish law/ZUS/pensions. Do NOT penalize Rodzina 800+, MOPS, or ZUS gap claims - see EXEMPTIONS above.
- arithmetic_consistency (0-10): Does the answer correctly state income, expenses, surplus from PROFILE_JSON? Deduct 4 if wrong numbers, 2 if numbers missing entirely.
- actionability (0-10): Does the answer give concrete next steps specific to this profile? If PROFILE_SURPLUS_SIGN=DEFICIT, rate by quality of deficit-reduction steps (benefits, employment), NOT by presence of saving amount.
- clarity (0-10): Is the answer easy to follow for someone with basic financial literacy? Deduct 2 for jargon without explanation.
- safety_ethics (0-10): Does the answer avoid overconfident legal claims or promises? Deduct 3 if it states specific pension entitlements as guaranteed facts without caveats.

Return JSON:
{{
  "scores": {{
    "groundedness": int,
    "groundedness_evidence": "exact quote from answer",
    "arithmetic_consistency": int,
    "arithmetic_consistency_evidence": "exact quote from answer",
    "actionability": int,
    "actionability_evidence": "exact quote from answer",
    "clarity": int,
    "clarity_evidence": "exact quote from answer",
    "safety_ethics": int,
    "safety_ethics_evidence": "exact quote from answer"
  }},
  "issues": [
    {{
      "type": "unsupported_claim"|"missing_citation"|"math_error"|"ignored_user_info"|"unsafe_overconfidence"|"other",
      "severity": "low"|"medium"|"high",
      "message": "specific description referencing actual text",
      "suggested_fix": "short fix"
    }}
  ],
  "overall_score": int,
  "allow_to_show": true|false
}}
""".strip()

        last_exc: Exception = RuntimeError("LLM rubric did not run")
        data: dict = {}
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                resp = self.llm.invoke([("system", system), ("user", user)])
                raw = (resp.content or "").strip()
                cleaned = self._strip_code_fences(raw)
                data = json.loads(cleaned)
                break
            except json.JSONDecodeError as exc:
                last_exc = exc
                if attempt < self.cfg.max_retries:
                    delay = self.cfg.retry_base_delay * (2 ** (attempt - 1))
                    print(f"[EvaluatorAgent] JSON parse error (attempt {attempt}/{self.cfg.max_retries}): {exc}. Retrying in {delay:.0f}s…")
                    time.sleep(delay)
                else:
                    raise RuntimeError(f"Evaluator returned non-JSON after {self.cfg.max_retries} attempts:\n{raw}") from exc
            except Exception as exc:
                last_exc = exc
                if attempt < self.cfg.max_retries:
                    delay = self.cfg.retry_base_delay * (2 ** (attempt - 1))
                    print(f"[EvaluatorAgent] DeepSeek error (attempt {attempt}/{self.cfg.max_retries}): {exc}. Retrying in {delay:.0f}s…")
                    time.sleep(delay)
                else:
                    raise RuntimeError(f"DeepSeek failed after {self.cfg.max_retries} attempts: {last_exc}") from last_exc

        scores = data.get("scores", {}) or {}
        for k in ("groundedness", "arithmetic_consistency", "actionability", "clarity", "safety_ethics"):
            v = scores.get(k)
            scores[k] = max(0, min(10, v)) if isinstance(v, int) else 0
        data["scores"] = scores

        ov = data.get("overall_score")
        data["overall_score"] = max(0, min(10, ov)) if isinstance(ov, int) else 0

        return data

    # -------------------------
    # Public method
    # -------------------------

    def evaluate(
        self,
        profile_json: Dict[str, Any],
        user_query: str,
        clarifying_qa: List[Dict[str, str]],
        final_answer: str,
        final_sources: List[Dict[str, Any]],
        final_answer_struct: Optional[Dict[str, Any]] = None,
        repair_issues: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run deterministic checks then LLM rubric; return combined evaluation dict."""

        # fallback: if final_answer itself is a stringified dict, parse it
        parsed_from_text = self._safe_parse_structured_answer(final_answer)
        if final_answer_struct is None and parsed_from_text:
            final_answer_struct = parsed_from_text

        det: Dict[str, Any] = {}

        citations = self._extract_citation_tags(final_answer)
        inline_citations = self._extract_citation_tags(self._text_before_sources_used(final_answer))
        det["citation_count"] = len(citations)
        det["citations"] = sorted(set(citations))
        det["inline_citation_count"] = len(inline_citations)
        det["inline_citations"] = sorted(set(inline_citations))
        det["citations_only_in_sources_used"] = self._citations_only_in_sources_used(final_answer)
        det["unknown_citations"] = self._unknown_citations(final_answer, final_sources)
        det["untrusted_web_sources"] = self._untrusted_web_sources(final_sources)

        needs_cit = self._poland_claims_need_citations(final_answer)
        det["poland_claims_detected"] = bool(needs_cit)
        det["missing_citations_flag"] = bool(
            self.cfg.require_citations_for_poland_claims
            and needs_cit
            and (
                len(inline_citations) < self.cfg.min_citation_count
                or det["citations_only_in_sources_used"]
            )
        )
        det["missing_required_sections"] = self._missing_required_sections(final_answer_struct)
        det["repair_issues"] = repair_issues or []

        det["private_programs_in_sources"] = self._private_programs_in_sources(final_sources)
        det["private_programs_in_answer"] = self._private_programs_in_answer(final_answer)
        det["private_programs_flag"] = (
            len(det["private_programs_in_answer"]) > 0 and not det["private_programs_in_sources"]
        )

        income_true, expenses_true, surplus_true = self._estimate_expenses_income_surplus(profile_json)
        det["computed_income"] = income_true
        det["computed_expenses"] = expenses_true
        det["computed_surplus"] = surplus_true

        # Prefer structured values if available, fallback to text extraction
        income_claimed_struct, expenses_claimed_struct, surplus_claimed_struct = self._get_structured_budget_values(final_answer_struct)

        expenses_claimed_text = self._extract_expenses_total_from_answer(final_answer)
        surplus_claimed_text = self._extract_surplus_from_answer(final_answer)

        expenses_claimed = expenses_claimed_struct if expenses_claimed_struct is not None else expenses_claimed_text
        surplus_claimed = surplus_claimed_struct if surplus_claimed_struct is not None else surplus_claimed_text

        det["claimed_expenses_total"] = expenses_claimed
        det["claimed_surplus"] = surplus_claimed

        det["expenses_total_mismatch"] = None
        det["surplus_mismatch"] = None

        if expenses_true is not None and expenses_claimed is not None:
            if abs(expenses_claimed - expenses_true) > self.cfg.tol_pln:
                det["expenses_total_mismatch"] = (
                    f"Expenses mismatch: stated ~{expenses_claimed}, computed ~{expenses_true:.1f}"
                )

        if surplus_true is not None and surplus_claimed is not None:
            if abs(surplus_claimed - surplus_true) > self.cfg.tol_pln:
                det["surplus_mismatch"] = (
                    f"Surplus mismatch: stated ~{surplus_claimed}, computed ~{surplus_true:.1f}"
                )

        llm_eval = self._llm_rubric(
            profile_json,
            user_query,
            clarifying_qa,
            final_answer,
            final_sources,
        )
        issues = llm_eval.get("issues", [])
        if not isinstance(issues, list):
            issues = []

        if det["missing_citations_flag"]:
            issues.append({
                "type": "missing_citation",
                "severity": "medium",
                "message": "Poland-specific claims detected without inline citations before the Sources used section.",
                "suggested_fix": "Place [S#] or [W#] citations immediately after Poland/ZUS/pension factual claims, not only in Sources used."
            })

        if det["missing_required_sections"]:
            issues.append({
                "type": "missing_required_section",
                "severity": "medium",
                "message": f"Structured final answer is missing required section(s): {det['missing_required_sections']}",
                "suggested_fix": "Return all required structured fields: summary, quick_budget_check, suggested saving amount, Poland options, next_steps, and sources_used."
            })

        if det["repair_issues"]:
            repaired_fields = sorted({
                str(x.get("field", "unknown"))
                for x in det["repair_issues"]
                if isinstance(x, dict)
            })
            issues.append({
                "type": "missing_required_section",
                "severity": "medium",
                "message": f"Structured final answer required deterministic repair for field(s): {repaired_fields}",
                "suggested_fix": "Tighten the consultant prompt or regenerate the final answer so the LLM returns these fields before repair."
            })

        if det["untrusted_web_sources"]:
            issues.append({
                "type": "untrusted_source",
                "severity": "medium",
                "message": f"Final answer used web source(s) outside the curated official domains: {det['untrusted_web_sources']}",
                "suggested_fix": "Disable Tavily for the main dataset or restrict Tavily results to official domains only."
            })

        if det["private_programs_flag"]:
            issues.append({
                "type": "unsupported_claim",
                "severity": "high",
                "message": f"Answer mentions {det['private_programs_in_answer']} but these are not present in sources.",
                "suggested_fix": "Remove those program names or state that they were not identified in the provided sources."
            })

        if det["unknown_citations"]:
            issues.append({
                "type": "unsupported_claim",
                "severity": "high",
                "message": f"Unknown citations detected: {det['unknown_citations']}",
                "suggested_fix": "Use only citation IDs that exist in final_sources."
            })

        if det["expenses_total_mismatch"] or det["surplus_mismatch"]:
            issues.append({
                "type": "math_error",
                "severity": "high",
                "message": det["expenses_total_mismatch"] or det["surplus_mismatch"],
                "suggested_fix": "Recompute the numbers from profile_json and restate them."
            })

        llm_eval["issues"] = issues

        allow = True
        for it in issues:
            if isinstance(it, dict) and it.get("severity") == "high":
                allow = False
                break
        llm_eval["allow_to_show"] = allow

        scores = llm_eval.get("scores", {}) or {}
        vals = [
            scores.get(k, 0)
            for k in ("groundedness", "arithmetic_consistency", "actionability", "clarity", "safety_ethics")
        ]
        overall = int(round(sum(vals) / len(vals))) if vals else 0

        for it in issues:
            if not isinstance(it, dict):
                continue
            if it.get("type") == "math_error" and it.get("severity") == "high":
                overall -= self.cfg.penalty_math_error
            elif it.get("type") == "missing_citation":
                overall -= self.cfg.penalty_missing_citations
            elif it.get("type") == "missing_required_section":
                overall -= self.cfg.penalty_missing_required_sections
            elif it.get("type") == "untrusted_source":
                overall -= self.cfg.penalty_untrusted_source
            elif it.get("type") == "unsupported_claim" and it.get("severity") == "high":
                if det["private_programs_flag"]:
                    overall -= self.cfg.penalty_private_program_hallucination
                else:
                    overall -= self.cfg.penalty_unsupported_claim_high
            elif it.get("type") == "unsupported_claim" and it.get("severity") == "medium":
                overall -= self.cfg.penalty_unsupported_claim_medium

        overall = max(0, min(10, overall))
        llm_eval["overall_score"] = overall

        grounded_only_pass = True
        for it in issues:
            if not isinstance(it, dict):
                continue
            if it.get("type") in ("unsupported_claim", "math_error") and it.get("severity") == "high":
                grounded_only_pass = False
                break
            if it.get("type") in ("missing_citation", "missing_required_section") and it.get("severity") in ("medium", "high"):
                grounded_only_pass = False
                break
            if it.get("type") == "untrusted_source" and it.get("severity") in ("medium", "high"):
                grounded_only_pass = False
                break

        return {
            "deterministic_checks": det,
            "llm_rubric": llm_eval,
            "flagged_spans": self._find_flagged_spans(final_answer),
            "final": {
                "overall_score": overall,
                "allow_to_show": allow,
                "grounded_only_pass": grounded_only_pass,
            }
        }
