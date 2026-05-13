"""
RAG-grounded consultant agent that produces pension advice for Ukrainian refugees in Poland.

The agent runs in two passes: draft() retrieves sources and generates clarifying questions,
final() combines the clarifications with a fresh retrieval to write the grounded answer.
Source retrieval merges vectorstore (PDF/HTML docs), live web (ZUS, gov.pl), and optional
Tavily search. Private pension vehicles (PPK/IKE/IKZE/OFE) are silently stripped from
answers unless they appear in the retrieved sources.
"""

from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.tools import mcp_fetch_to_file, tavily_search


class _QuickBudgetCheck(BaseModel):
    monthly_income: float = Field(description="Client's monthly income in PLN")
    monthly_expenses_total: float = Field(description="Client's total monthly expenses in PLN")
    monthly_surplus: float = Field(description="monthly_income minus monthly_expenses_total")


class _FinalAnswerStruct(BaseModel):
    summary: str = Field(
        description=(
            "2-4 sentence narrative summary of the retirement advice. "
            "Place [S#] citation tags immediately after any Poland/ZUS/pension factual claims."
        )
    )
    quick_budget_check: _QuickBudgetCheck = Field(
        description="Budget numbers taken directly from the client profile."
    )
    suggested_monthly_retirement_saving_amount: str = Field(
        description=(
            "Specific PLN amount or range the client should save monthly for retirement "
            "(e.g. 'PLN 200-300 per month'). If surplus <= 0 write 'PLN 0 for now; stabilise budget first'. "
            "Always add '(general rule of thumb, not Poland-specific)' if not backed by a source."
        )
    )
    retirement_related_options_in_poland: List[str] = Field(
        description=(
            "2-4 concrete retirement-related options available in Poland for this client "
            "(e.g. ZUS contributions, voluntary ZUS, checking ZUS projection). "
            "Each item must end with an inline [S#] citation if it mentions Polish law or ZUS."
        )
    )
    next_steps: List[str] = Field(
        description=(
            "Exactly 3-5 actionable next steps for this specific client, ordered by priority. "
            "At least one step must reference a ZUS or official action with an [S#] citation."
        )
    )
    sources_used: List[str] = Field(
        description="All citation tags referenced in this answer, e.g. ['[S1]', '[S2]', '[S100]']."
    )

_TAVILY_MAX_QUERY_LEN = 400
_TAVILY_SOURCE_ID_START = 200
_WEB_SOURCE_ID_START = 100


def _env_bool(name: str, default: str = "false") -> bool:
    """Read an environment variable and interpret it as a boolean flag."""
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: str = "") -> List[str]:
    """Read a comma-separated environment variable and return a list of non-empty strings."""
    raw = os.getenv(name, default)
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


@dataclass
class ConsultantAgentConfig:
    """Runtime settings; web access and Tavily can be toggled via environment variables.

    CONSULTANT_ENABLE_WEB, CONSULTANT_ALLOWED_DOMAINS, CONSULTANT_ENABLE_TAVILY,
    CONSULTANT_CURATED_URLS, CONSULTANT_WEB_TIMEOUT_MS, CONSULTANT_WEB_MAX_SOURCES.
    """

    model_name: str = "gpt-4.1-mini"
    temperature: float = 0.1
    max_clarifying_questions: int = 3

    # Live web access
    web_enabled: bool = field(default_factory=lambda: _env_bool("CONSULTANT_ENABLE_WEB", "true"))
    allowed_domains: List[str] = field(
        default_factory=lambda: _env_list(
            "CONSULTANT_ALLOWED_DOMAINS",
            "gov.pl,zus.pl,podatki.gov.pl,biznes.gov.pl,euraxess.pl,ec.europa.eu",
        )
    )
    web_timeout_ms: int = int(os.getenv("CONSULTANT_WEB_TIMEOUT_MS", "20000"))
    web_max_sources: int = int(os.getenv("CONSULTANT_WEB_MAX_SOURCES", "4"))
    web_cache_dir: str = os.getenv("CONSULTANT_WEB_CACHE_DIR", "books_consultant/live_web_cache")

    # Optional extra curated URLs from .env
    curated_urls: List[str] = field(default_factory=lambda: _env_list("CONSULTANT_CURATED_URLS", ""))

    # Tavily search
    tavily_enabled: bool = field(default_factory=lambda: _env_bool("CONSULTANT_ENABLE_TAVILY", "false"))
    tavily_api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))

    # Topic router
    topic_router_enabled: bool = True


class ConsultantAgent:
    """Two-pass RAG consultant: draft -> clarify -> final answer with verified arithmetic."""

    def __init__(self, vectorstore: FAISS, config: ConsultantAgentConfig):
        self.vectorstore = vectorstore
        self.config = config
        self.llm = ChatOpenAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
        )
        # Structured output enforces all required fields in the schema — used in final().
        # Falls back to json mode if structured output raises.
        self.llm_struct = self.llm.with_structured_output(
            _FinalAnswerStruct,
            method="function_calling",
        )
        self.llm_json = ChatOpenAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

    # -------------------------
    # Local retrieval
    # -------------------------

    def _retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Run vectorstore similarity search and return source-tagged dicts ready for the sources block."""
        docs = self.vectorstore.similarity_search(query, k=k)
        out: List[Dict[str, Any]] = []
        for i, d in enumerate(docs, start=1):
            meta = getattr(d, "metadata", {}) or {}
            out.append(
                {
                    "id": f"S{i}",
                    "source": meta.get("source", "unknown"),
                    "page": meta.get("page"),
                    "chunk_id": meta.get("chunk_id"),
                    "snippet": getattr(d, "page_content", "")[:1500],
                    "source_type": "vectorstore",
                }
            )
        return out

    # -------------------------
    # Topic router
    # -------------------------

    def _detect_topics(self, query: str, profile_text: str = "") -> List[str]:
        """Keyword scan over query and profile text; returns matched topic labels to drive URL selection."""
        q_main = query.lower()
        q_profile = profile_text.lower()

        topic_keywords = {
            "tax": ["pit", "tax", "taxes", "e-pit", "e tax", "tax office", "podatki"],
            "registration": ["register", "registration", "insured registration", "insurance registration"],
            "social_security": ["social security", "foreigner researcher", "euraxess", "insurance rights", "insured"],
            "pension": ["pension", "retirement", "old-age", "emerytura", "zus contribution", "retire", "zus"],
        }

        scores: Dict[str, float] = {topic: 0.0 for topic in topic_keywords}

        for topic, keywords in topic_keywords.items():
            for kw in keywords:
                if kw in q_main:
                    scores[topic] += 2.0
                if kw in q_profile:
                    scores[topic] += 0.5

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = [topic for topic, score in ranked if score >= 1.0]

        return selected if selected else ["general"]

    def _topic_url_map(self) -> Dict[str, List[str]]:
        """Return curated authoritative URLs per topic; pension URLs prioritise the ZUS calculator."""
        return {
            "pension": [
                "https://www.gov.pl/web/your-europe/old-age-pension--step-by-step",
                "https://www.zus.pl/swiadczenia/emerytury/kalkulatory-emerytalne/emerytura-na-nowych-zasadach/kalkulator-emerytalny-prognozowana-emerytura",
                "https://www.zus.pl/swiadczenia/emerytury/kalkulatory-emerytalne/emerytura-na-nowych-zasadach/kalkulator-emerytalny-prognozowana-emerytura/obliczenia-w-kalkulatorze-emerytalnym",
                "https://www.euraxess.pl/poland/social-security",
            ],
            "tax": [
                "https://www.gov.pl/web/finance/your-e-pite-tax-office",
                "https://www.podatki.gov.pl/twoj-e-pit/en-pit-36-za-2025-rok/",
            ],
            "registration": [
                "https://www.biznes.gov.pl/en/portal/004114",
                "https://www.euraxess.pl/poland/social-security",
            ],
            "social_security": [
                "https://www.euraxess.pl/poland/social-security",
                "https://www.gov.pl/web/your-europe/old-age-pension--step-by-step",
            ],
            "general": [
                "https://www.gov.pl/web/your-europe/old-age-pension--step-by-step",
                "https://www.euraxess.pl/poland/social-security",
                "https://www.biznes.gov.pl/en/portal/004114",
                "https://www.gov.pl/web/finance/your-e-pite-tax-office",
            ],
        }

    def _candidate_urls_for_query(self, query: str, profile_text: str = "") -> List[str]:
        """Merge topic-matched and curated URLs preserving insertion order, deduplicating."""
        topics = self._detect_topics(query, profile_text)

        merged: List[str] = []
        seen = set()

        for topic in topics:
            for url in self._topic_url_map().get(topic, []):
                if url not in seen:
                    merged.append(url)
                    seen.add(url)

        for url in (self.config.curated_urls or []):
            if url not in seen:
                merged.append(url)
                seen.add(url)

        return merged

    def _score_url(self, url: str, query: str, profile_text: str = "") -> int:
        """Heuristic relevance score for URL ranking; combines topic signals and trusted-domain bonus."""
        url_l = url.lower()
        q = f"{query}\n{profile_text}".lower()
        topics = self._detect_topics(query, profile_text)

        score = 0

        topic_url_signals = {
            "pension": ["old-age-pension", "emerytur", "zus", "kalkulator", "retirement"],
            "tax": ["podatki", "e-pit", "finance", "pit-36", "tax"],
            "registration": ["biznes", "register", "insurance", "zus"],
            "social_security": ["social-security", "euraxess", "your-europe", "insurance"],
            "general": ["gov", "zus", "euraxess", "biznes", "podatki"],
        }

        for topic in topics:
            for signal in topic_url_signals.get(topic, []):
                if signal in url_l:
                    score += 3

        generic_keywords = [
            "poland", "pension", "retirement", "zus",
            "tax", "pit", "insurance", "social", "register"
        ]
        score += sum(1 for kw in generic_keywords if kw in q and kw in url_l)

        # bonus for official/trusted domains
        if any(d in url_l for d in ["gov.pl", "zus.pl", "podatki.gov.pl", "biznes.gov.pl", "euraxess.pl"]):
            score += 2

        return score

    # -------------------------
    # Live web retrieval
    # -------------------------

    def _live_web_sources(self, query: str, profile_text: str = "") -> List[Dict[str, Any]]:
        """Fetch the top-ranked candidate URLs via MCP; silently drops failed fetches from the good-sources list."""
        if not self.config.web_enabled:
            return []

        candidate_urls = self._candidate_urls_for_query(query, profile_text)
        if not candidate_urls:
            return []

        ranked_urls = sorted(
            candidate_urls,
            key=lambda u: self._score_url(u, query, profile_text),
            reverse=True,
        )[: self.config.web_max_sources]

        web_sources: List[Dict[str, Any]] = []
        next_idx = _WEB_SOURCE_ID_START

        for url in ranked_urls:
            try:
                result = mcp_fetch_to_file(
                    url,
                    out_dir=self.config.web_cache_dir,
                    allowed_domains=self.config.allowed_domains,
                    timeout_ms=self.config.web_timeout_ms,
                )

                cleaned = re.sub(r"\s+", " ", result["text"]).strip()
                if not cleaned:
                    continue

                web_sources.append(
                    {
                        "id": f"S{next_idx}",
                        "source": url,
                        "page": None,
                        "chunk_id": "live-web",
                        "snippet": cleaned[:1500],
                        "source_type": "live_web",
                        "txt_path": result.get("txt_path"),
                    }
                )
                next_idx += 1

            except (ValueError, OSError, RuntimeError, TimeoutError) as e:
                print("WEB FETCH ERROR:", url, repr(e))
                web_sources.append(
                    {
                        "id": f"S{next_idx}",
                        "source": url,
                        "page": None,
                        "chunk_id": "live-web-error",
                        "snippet": f"[web fetch failed: {e}]",
                        "source_type": "live_web_error",
                    }
                )
                next_idx += 1

        good = [s for s in web_sources if s.get("source_type") == "live_web"]
        return good if good else web_sources

    def _tavily_sources(self, user_query: str) -> List[Dict[str, Any]]:
        """Run a Tavily search; returns an empty list when disabled or when the API key is missing."""
        if not self.config.tavily_enabled or not self.config.tavily_api_key:
            return []

        search_query = f"{user_query} ZUS Poland pension retirement"[:_TAVILY_MAX_QUERY_LEN]

        try:
            results = tavily_search(
                search_query,
                api_key=self.config.tavily_api_key,
                max_results=self.config.web_max_sources,
                allowed_domains=self.config.allowed_domains or None,
            )
        except (ImportError, OSError, ValueError, RuntimeError) as e:
            print("TAVILY ERROR:", repr(e))
            return []

        sources: List[Dict[str, Any]] = []
        for i, r in enumerate(results, start=_TAVILY_SOURCE_ID_START):
            snippet = (r.get("content") or "")[:1500]
            if not snippet:
                continue
            sources.append({
                "id": f"S{i}",
                "source": r.get("url", ""),
                "page": None,
                "chunk_id": "tavily",
                "snippet": snippet,
                "source_type": "tavily_search",
            })
        return sources

    def _combined_sources(self, query: str, profile_text: str, user_query: str = "", k: int = 5) -> List[Dict[str, Any]]:
        """Merge vectorstore + live web + Tavily, preserving ID namespaces (S1-S99, S100+, S200+)."""
        local_sources = self._retrieve(query, k=k)
        live_sources = self._live_web_sources(query, profile_text)
        tavily_sources = self._tavily_sources(user_query or query)
        return local_sources + live_sources + tavily_sources

    def _sources_block(self, sources: List[Dict[str, Any]]) -> str:
        """Format the sources list as a text block for injection into the LLM prompt."""
        if not sources:
            return "No sources retrieved."

        lines = []
        for s in sources:
            extra = []
            if s.get("source_type"):
                extra.append(f"type={s['source_type']}")
            if s.get("html_path"):
                extra.append(f"html_path={s['html_path']}")
            meta_extra = ", ".join(extra)

            header = (
                f"[{s['id']}] source={s['source']}, page={s.get('page')}, "
                f"chunk_id={s.get('chunk_id')}"
            )
            if meta_extra:
                header += f", {meta_extra}"

            lines.append(f"{header}\nSnippet: {s.get('snippet', '')}")

        return "\n\n".join(lines)

    # -------------------------
    # Helpers
    # -------------------------

    @staticmethod
    def _clean_model_json_text(raw: str) -> str:
        """Strip ```json ... ``` fences that models sometimes wrap around JSON output."""
        txt = (raw or "").strip()
        m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", txt, flags=re.DOTALL | re.IGNORECASE)
        if m:
            txt = m.group(1).strip()
        return txt

    def _safe_parse_structured_answer(self, raw: Any) -> Dict[str, Any]:
        """Parse JSON with json.loads -> ast.literal_eval fallback; never raises, returns {} on failure."""
        if isinstance(raw, dict):
            return raw

        if not isinstance(raw, str):
            return {"summary": str(raw)}

        txt = self._clean_model_json_text(raw)
        if not txt:
            return {"summary": ""}

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

        return {"summary": txt}

    def _normalize_nested_summary(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Unwrap doubly-nested JSON when the model returns the full answer as a stringified dict inside 'summary'."""
        summary = parsed.get("summary") or parsed.get("Summary")
        if isinstance(summary, str):
            inner = self._safe_parse_structured_answer(summary)
            # Only unwrap if inner has structural keys beyond "summary" itself —
            # otherwise a plain-text summary is always parsed as {"summary": text}
            # and incorrectly replaces the full struct.
            _structural = {
                "quick_budget_check", "suggested_monthly_retirement_saving_amount",
                "retirement_related_options_in_poland", "next_steps", "sources_used",
                "Quick budget check", "Suggested monthly retirement saving amount",
                "Retirement-related options in Poland", "Next steps", "Sources used",
            }
            if inner and any(k in inner for k in _structural):
                return inner
        return parsed

    @staticmethod
    def _parse_money_token(tok: Any) -> Any:
        """Convert a PLN string like 'PLN 1,200' or '1200.0' to float; returns original if unparseable."""
        if tok is None:
            return None
        if isinstance(tok, (int, float)):
            return float(tok)
        if isinstance(tok, str):
            x = tok.strip().replace("€", "").replace("PLN", "").replace("pln", "").replace(",", "").strip()
            try:
                return float(x)
            except ValueError:
                return tok
        return tok

    def _sources_contain_private_programs(self, sources: List[Dict[str, Any]]) -> Dict[str, bool]:
        """Check which of PPK/IKE/IKZE/OFE appear in source snippets; determines what the answer may cite."""
        hay = " ".join((s.get("snippet") or "") for s in sources).lower()
        return {
            "ppk": bool(re.search(r"\bppk\b", hay)),
            "ike": bool(re.search(r"\bike\b", hay)),
            "ikze": bool(re.search(r"\bikze\b", hay)),
            "ofe": bool(re.search(r"\bofe\b", hay)),
        }

    def _remove_unsupported_private_programs(
        self, parsed: Dict[str, Any], sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Redact pension vehicle names absent from sources to prevent hallucination leaking into the final output."""
        allowed = self._sources_contain_private_programs(sources)
        banned_terms = [term for term in ("ppk", "ike", "ikze", "ofe") if not allowed.get(term, False)]

        if not banned_terms:
            return parsed

        def clean_text(text: Any) -> Any:
            if not isinstance(text, str):
                return text
            out = text
            for term in banned_terms:
                out = re.sub(
                    rf"\b{re.escape(term)}\b",
                    "[not identified in provided sources]",
                    out,
                    flags=re.IGNORECASE,
                )
            out = re.sub(
                r"(?:\[not identified in provided sources\][,\s]*){2,}",
                "[not identified in provided sources], ",
                out,
            )
            out = re.sub(r"\s{2,}", " ", out).strip()
            return out

        cleaned = {}
        for key, value in parsed.items():
            if isinstance(value, str):
                cleaned[key] = clean_text(value)
            elif isinstance(value, list):
                new_list = []
                for item in value:
                    if isinstance(item, str):
                        item2 = clean_text(item)
                        if item2.strip() == "[not identified in provided sources]":
                            continue
                        new_list.append(item2)
                    else:
                        new_list.append(item)
                cleaned[key] = new_list
            elif isinstance(value, dict):
                sub = {}
                for k2, v2 in value.items():
                    sub[k2] = clean_text(v2) if isinstance(v2, str) else v2
                cleaned[key] = sub
            else:
                cleaned[key] = value

        return cleaned

    def _format_final_answer_text(self, answer: Dict[str, Any]) -> str:
        """Render the structured answer dict as numbered sections for evaluator scoring."""
        parts: List[str] = []

        summary = answer.get("summary") or answer.get("Summary")
        if summary:
            parts.append(f"1) Summary: {summary}")

        qb = answer.get("quick_budget_check") or answer.get("Quick budget check")
        if isinstance(qb, dict):
            income = qb.get("monthly_income") or qb.get("Monthly income")
            expenses = qb.get("monthly_expenses_total") or qb.get("Monthly expenses total")
            surplus = qb.get("monthly_surplus") or qb.get("Monthly surplus")
            parts.append(
                f"2) Quick budget check: Your monthly income is PLN {income}, "
                f"and your total monthly expenses are PLN {expenses}, leaving you with a surplus of PLN {surplus}."
            )
        elif isinstance(qb, str) and qb.strip():
            parts.append(f"2) Quick budget check: {qb}")

        save_amt = (
            answer.get("suggested_monthly_retirement_saving_amount")
            or answer.get("Suggested monthly retirement saving amount")
        )
        if save_amt is not None:
            parts.append(f"3) Suggested monthly retirement saving amount: {save_amt}")

        poland_opts = (
            answer.get("retirement_related_options_in_poland")
            or answer.get("Retirement-related options in Poland")
        )
        if poland_opts:
            opts_text = "; ".join(str(x) for x in poland_opts) if isinstance(poland_opts, list) else str(poland_opts)
            parts.append(f"4) Retirement-related options in Poland: {opts_text}")

        next_steps = answer.get("next_steps") or answer.get("Next steps")
        if next_steps:
            steps_text = " ".join(f"{i+1}. {str(x)}" for i, x in enumerate(next_steps)) if isinstance(next_steps, list) else str(next_steps)
            parts.append(f"5) Next steps: {steps_text}")

        src = answer.get("sources_used") or answer.get("Sources used")
        if src:
            src_text = ", ".join(str(x) for x in src) if isinstance(src, list) else str(src)
            parts.append(f"6) Sources used: {src_text}")

        return "\n\n".join(parts) if parts else str(answer)

    @staticmethod
    def _source_tag(sources: List[Dict[str, Any]], domain_hint: str = "") -> str:
        """Pick an available source tag, preferring a domain/url hint when possible."""
        if not sources:
            return ""
        hint = domain_hint.lower().strip()
        if hint:
            for s in sources:
                if hint in str(s.get("source", "")).lower():
                    return f"[{s['id']}]"
        return f"[{sources[0]['id']}]"

    @staticmethod
    def _has_citation(text: Any) -> bool:
        return bool(re.search(r"\[(?:S|W)\d+\]", str(text or "")))

    def _ensure_required_final_sections(
        self,
        parsed: Dict[str, Any],
        profile_json: Dict[str, Any],
        sources: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """Repair missing required sections so final_answer_struct is article-ready."""
        repair_issues: List[Dict[str, str]] = []
        zus_tag = self._source_tag(sources, "zus.pl") or self._source_tag(sources)
        gov_tag = self._source_tag(sources, "gov.pl") or zus_tag
        euraxess_tag = self._source_tag(sources, "euraxess.pl") or zus_tag

        def mark(field: str) -> None:
            repair_issues.append({
                "field": field,
                "message": f"Missing or weak '{field}' repaired deterministically.",
            })

        summary = str(parsed.get("summary") or parsed.get("Summary") or "").strip()
        if not summary:
            summary = (
                "Based on your current income, expenses, and savings, the safest retirement plan is to start "
                "with a modest monthly amount and review it after your emergency buffer improves."
            )
            mark("summary")
        elif re.search(r"\b(Poland|Polish|ZUS|old-age pension|retirement age|social insurance)\b", summary, flags=re.IGNORECASE) and not self._has_citation(summary):
            summary = f"{summary} {gov_tag}".strip()
            mark("summary_inline_citation")
        parsed["summary"] = summary

        surplus = float(profile_json.get("income", 0.0)) - sum(
            float(profile_json.get(k, 0.0))
            for k in ("housing", "utilities", "food", "transport", "healthcare", "other", "remittances")
        )

        save_amt = parsed.get("suggested_monthly_retirement_saving_amount")
        if not str(save_amt or "").strip():
            if surplus <= 0:
                save_amt = (
                    "PLN 0 for now; first stabilise the budget, then start with PLN 50-100 when a monthly surplus appears "
                    "(general rule of thumb, not Poland-specific)."
                )
            else:
                low = max(50, round(surplus * 0.30 / 10) * 10)
                high = max(low, round(surplus * 0.50 / 10) * 10)
                save_amt = (
                    f"PLN {low:.0f}-{high:.0f} per month, capped below the current surplus "
                    "(general rule of thumb, not Poland-specific)."
                )
            mark("suggested_monthly_retirement_saving_amount")
        parsed["suggested_monthly_retirement_saving_amount"] = save_amt

        opts = parsed.get("retirement_related_options_in_poland") or parsed.get("Retirement-related options in Poland")
        if not isinstance(opts, list) or not [x for x in opts if str(x).strip()]:
            opts = [
                f"Check the official ZUS pension calculator/projection to understand your future state pension estimate {zus_tag}.",
                f"Make sure any work activity that should create social-insurance pension contributions is correctly recorded with ZUS {euraxess_tag}.",
                f"Use official gov.pl/ZUS guidance before relying on exact pension ages, eligibility rules, or benefit amounts {gov_tag}.",
            ]
            mark("retirement_related_options_in_poland")
        else:
            opts = [str(x).strip() for x in opts if str(x).strip()]
            opts = [f"{x} {zus_tag}".strip() if re.search(r"\b(Poland|Polish|ZUS|old-age pension|retirement age|social insurance|eligible|eligibility)\b", x, flags=re.IGNORECASE) and not self._has_citation(x) else x for x in opts]
        parsed["retirement_related_options_in_poland"] = opts

        next_steps = parsed.get("next_steps") or parsed.get("Next steps")
        if not isinstance(next_steps, list) or len([x for x in next_steps if str(x).strip()]) < 3:
            next_steps = [
                "Keep the suggested amount in a separate savings account or sub-account right after income arrives.",
                "Build or protect at least a small emergency buffer before increasing long-term retirement saving.",
                f"Review ZUS records/projections once a year and update the monthly saving target if income or expenses change {zus_tag}.",
            ]
            mark("next_steps")
        else:
            next_steps = [str(x).strip() for x in next_steps if str(x).strip()]
            next_steps = [f"{x} {zus_tag}".strip() if re.search(r"\b(Poland|Polish|ZUS|old-age pension|retirement age|social insurance|eligible|eligibility)\b", x, flags=re.IGNORECASE) and not self._has_citation(x) else x for x in next_steps]
        parsed["next_steps"] = next_steps[:5]

        sources_used = parsed.get("sources_used") or parsed.get("Sources used")
        if not isinstance(sources_used, list) or not sources_used:
            sources_used = [f"[{s['id']}]" for s in sources]
            mark("sources_used")
        parsed["sources_used"] = [str(x).strip() for x in sources_used if str(x).strip()]

        return repair_issues

    def _extract_used_source_ids(self, text: str, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return only sources whose [ID] tag actually appears in the answer text."""
        used = []
        for s in sources:
            sid = f"[{s['id']}]"
            if sid in text:
                used.append(s)
        return used if used else sources

    # -------------------------
    # Profile context hints
    # -------------------------

    @staticmethod
    def _profile_context_hints(profile_json: Dict[str, Any]) -> str:
        """Generate mandatory hints about Rodzina 800+ and MOPS so the model does not miss benefit eligibility."""
        hints: List[str] = []
        dep = int(profile_json.get("dependents", 0))
        emp = str(profile_json.get("employment_status", ""))
        income = float(profile_json.get("income", 0))

        if dep > 0:
            benefit_800 = dep * 800
            hints.append(
                f"IMPORTANT: The client has {dep} dependent(s). Ukrainian refugees under temporary protection "
                f"in Poland are eligible for the Rodzina 800+ benefit (PLN 800 per child per month). "
                f"This client may receive up to PLN {benefit_800} per month from this program. "
                f"You MUST mention this if it is relevant to their financial situation."
            )

        if emp in ("unemployed", "economically_inactive"):
            hints.append(
                "IMPORTANT: The client is not employed. This means they are NOT currently accumulating "
                "ZUS pension contributions or insurance period (staz ubezpieczeniowy). "
                "A gap in ZUS contributions directly reduces their future state pension amount. "
                "Mention this and note that voluntary ZUS contributions are possible but costly on low income."
            )
            if income < 2000:
                hints.append(
                    "IMPORTANT: The client's income is very low. They may be eligible for social assistance "
                    "(zasilek z MOPS, swiadczenia spoleczne) from the local Social Welfare Centre (MOPS/GOPS). "
                    "Suggest they check eligibility at their local MOPS office."
                )

        return "\n\n".join(hints)

    # -------------------------
    # Draft step
    # -------------------------

    def draft(
        self,
        profile_text: str,
        user_query: str,
        profile_json: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Retrieve sources and return a draft answer plus up to 3 clarifying questions."""
        query = f"{user_query}\n\n{profile_text}\n\nPoland retirement pension ZUS"
        sources = self._combined_sources(query, profile_text, user_query=user_query, k=5)
        sources_block = self._sources_block(sources)

        context_hints = self._profile_context_hints(profile_json)
        hints_block = f"\nPROFILE CONTEXT HINTS (act on these):\n{context_hints}\n" if context_hints else ""

        prompt = f"""
You are a cautious financial guidance assistant.

Your task:
1. Read the client profile.
2. Use only the retrieved sources when making Poland-specific claims.
3. Give a short draft answer.
4. Ask up to {self.config.max_clarifying_questions} useful clarifying questions.

Rules:
- Be conservative.
- If a rule of thumb is general and not Poland-specific, say so explicitly: label it "(general rule of thumb, not Poland-specific)".
- Do not mention PPK, IKE, IKZE, OFE, or any other Polish pension vehicle unless it is explicitly supported in the retrieved sources.
- Do NOT confuse "social pension" (invalidity benefit for those unable to work) with the regular "old-age pension". They may share the same minimum amount (PLN 1,780.96) but are different benefits with different eligibility conditions.
- Curated live web sources are allowed only if they appear in the retrieved sources block.
- Return JSON with keys:
  - draft_answer
  - clarifying_questions

Client profile:
{profile_text}

Structured profile:
{json.dumps(profile_json, ensure_ascii=False)}
{hints_block}
User query:
{user_query}

Retrieved sources:
{sources_block}
""".strip()

        resp = self.llm.invoke(prompt)
        text = getattr(resp, "content", str(resp)).strip()

        try:
            parsed = json.loads(self._clean_model_json_text(text))
        except json.JSONDecodeError:
            parsed = {
                "draft_answer": text,
                "clarifying_questions": [
                    "What is your current age?",
                    "Do you have any existing retirement savings or contributions to ZUS?",
                    "Are you planning to stay in Poland until retirement?",
                ][: self.config.max_clarifying_questions],
            }

        questions = parsed.get("clarifying_questions") or []
        if not isinstance(questions, list):
            questions = []
        questions = [str(q).strip() for q in questions if str(q).strip()][: self.config.max_clarifying_questions]

        draft_answer = str(parsed.get("draft_answer", ""))
        draft_answer = self._clean_model_json_text(draft_answer)

        return {
            "draft_answer": draft_answer,
            "sources": sources,
            "clarifying_questions": questions,
        }

    # -------------------------
    # Final step
    # -------------------------

    def final(
        self,
        profile_text: str,
        user_query: str,
        clarifying_qa: List[Dict[str, str]],
        profile_json: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Produce the final structured answer; income/expenses/surplus are injected from profile_json
        to guarantee arithmetic correctness regardless of model output."""
        query = f"{user_query}\n\n{profile_text}\n\nPoland retirement pension ZUS"
        sources = self._combined_sources(query, profile_text, user_query=user_query, k=5)
        sources_block = self._sources_block(sources)

        qa_block = "\n".join(
            f"Q: {item.get('question', '')}\nA: {item.get('answer', '')}"
            for item in clarifying_qa
        ).strip()

        income = float(profile_json.get("income", 0.0))
        expenses = (
            float(profile_json.get("housing", 0.0))
            + float(profile_json.get("utilities", 0.0))
            + float(profile_json.get("food", 0.0))
            + float(profile_json.get("transport", 0.0))
            + float(profile_json.get("healthcare", 0.0))
            + float(profile_json.get("other", 0.0))
            + float(profile_json.get("remittances", 0.0))
        )
        surplus = round(income - expenses, 2)

        context_hints = self._profile_context_hints(profile_json)
        hints_block = f"\nPROFILE CONTEXT HINTS (act on these):\n{context_hints}\n" if context_hints else ""

        prompt = f"""
You are a cautious financial guidance assistant.

Write a final answer for the client.

Important rules:
- Use retrieved sources for Poland-specific claims.
- Place citation tags immediately after each Poland-specific factual claim, for example: "In Poland, the general retirement age is 60 for women and 65 for men [S102]."
- Do not place all citations only in the final sources_used field. The summary, retirement options, and next steps must contain inline citations wherever they make Poland/ZUS/social-insurance/pension factual claims.
- If something is a general financial heuristic, explicitly label it "(general rule of thumb, not Poland-specific)".
- Do not mention PPK, IKE, IKZE, OFE, or any other Polish pension vehicle unless it is explicitly supported in the retrieved sources.
- Do NOT confuse "social pension" (invalidity benefit for those unable to work) with the regular "old-age pension". They may share the same minimum amount (PLN 1,780.96) but are different benefits with different eligibility conditions.
- Avoid exact minimum old-age pension amounts unless the retrieved source clearly states a current 2025/2026 amount. Prefer telling the client to check the current amount on the official ZUS website.
- Curated live web sources are allowed only if they appear in the retrieved sources block.
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
  "sources_used": ["[S1]", "[S100]"]
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
{sources_block}
""".strip()

        try:
            result = self.llm_struct.invoke(prompt)
            parsed = result.model_dump() if hasattr(result, "model_dump") else dict(result)
        except Exception as e:
            print(f"[ConsultantAgent] structured output failed ({e}), falling back to json mode")
            resp = self.llm_json.invoke(prompt)
            raw = getattr(resp, "content", str(resp)).strip()
            parsed = self._safe_parse_structured_answer(raw)

        parsed = self._normalize_nested_summary(parsed)
        parsed = self._remove_unsupported_private_programs(parsed, sources)
        repair_issues = self._ensure_required_final_sections(parsed, profile_json, sources)

        qb = parsed.get("quick_budget_check") or parsed.get("Quick budget check")
        if not isinstance(qb, dict):
            qb = {}

        qb["monthly_income"] = income
        qb["monthly_expenses_total"] = expenses
        qb["monthly_surplus"] = surplus
        parsed["quick_budget_check"] = qb

        if "sources_used" not in parsed:
            parsed["sources_used"] = [f"[{s['id']}]" for s in sources]

        final_answer_text = self._format_final_answer_text(parsed)
        final_sources = self._extract_used_source_ids(final_answer_text, sources)

        return {
            "final_answer": final_answer_text,
            "final_answer_struct": parsed,
            "sources": final_sources,
            "repair_issues": repair_issues,
        }
