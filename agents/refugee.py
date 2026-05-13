"""
Synthetic refugee agent for generating demographically calibrated client profiles.

Numeric budget figures are produced algorithmically from NBP 2025 survey weights;
LLM is used only for qualitative constraints, persona flavour, and clarifying answers.
Profile diversity across employment status, gender, income, and language level is
controlled via the anchor sampling distribution in RefugeeAgentConfig.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS

from src.tools import retrieve_chunks


@dataclass
class RefugeeAgentConfig:
    """Sampling parameters and LLM settings for the refugee agent.

    seed=None uses system entropy so consecutive script runs never repeat
    the same profile sequence. Pass an int for reproducible experiments.
    """

    model: str = "gpt-4o-mini"

    # LLM usage:
    # - constraints: short phrases (style)
    # - user_question: retirement-only question (style)
    # - persona: one-time latent persona (creativity but bounded)
    # - clarify_answers: answers from persona+profile only (stable)
    temperature_question: float = 0.45
    temperature_constraints: float = 0.55
    temperature_persona: float = 0.65
    temperature_clarify: float = 0.25

    k: int = 4  # RAG for constraints only

    seed: Optional[int] = None  # None -> different sequence every script run; set int for reproducibility
    female_prior: float = 0.66
    min_months_in_poland: int = 6

    round_to: int = 10
    min_retirement_surplus_pln: int = 100


class RefugeeAgent:
    """
    Persona-mode refugee agent:

    - Numeric budget profile generated algorithmically (PLN).
    - RAG used only for qualitative constraint phrases.
    - persona_json generated ONCE per case and stored.
    - Clarifying answers:
        * first: rule-based for common questions (employment, goals, age/retire horizon, plans/enrollment)
        * then: LLM fallback that MUST use ONLY anchors+profile+persona (no new facts)
      Answers are 2-4 sentences and should not say "I don't know my goals".
    """

    def __init__(self, vs_refuge: FAISS, cfg: RefugeeAgentConfig = RefugeeAgentConfig()):
        self.vs = vs_refuge
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)  # seed=None -> different sequence every script run

        self.llm_q = ChatOpenAI(model=cfg.model, temperature=cfg.temperature_question)
        self.llm_c = ChatOpenAI(model=cfg.model, temperature=cfg.temperature_constraints)
        self.llm_p = ChatOpenAI(model=cfg.model, temperature=cfg.temperature_persona)
        self.llm_a = ChatOpenAI(model=cfg.model, temperature=cfg.temperature_clarify)

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

    def _round_money(self, x: float) -> float:
        """Round to the nearest 10 PLN for realistic-looking budget figures."""
        r = self.cfg.round_to
        return float(round(x / r) * r)

    def _triangular(self, low: float, mode: float, high: float) -> float:
        """Draw one sample from a triangular distribution using the instance RNG."""
        return self.rng.triangular(low, high, mode)

    def _sample_persona_controls(self, profile_json: Dict[str, Any]) -> Dict[str, Any]:
        """Algorithmically sample persona fields that otherwise collapse under LLM defaults."""
        age = self.rng.randint(25, 58)
        city_tier = self.rng.choices(["large", "medium", "small"], weights=[0.42, 0.38, 0.20], k=1)[0]
        retirement_priority = self.rng.choices(["low", "medium", "high"], weights=[0.28, 0.46, 0.26], k=1)[0]

        if age < 35:
            horizon = self.rng.choice(["early 60s", "mid-60s", "after 65", "not sure yet"])
        elif age < 48:
            horizon = self.rng.choice(["around 60", "mid-60s", "after 65"])
        else:
            horizon = self.rng.choice(["around 60", "early 60s", "mid-60s"])

        savings = float(profile_json.get("savings", 0.0))
        surplus = (
            float(profile_json.get("income", 0.0))
            - float(profile_json.get("housing", 0.0))
            - float(profile_json.get("utilities", 0.0))
            - float(profile_json.get("food", 0.0))
            - float(profile_json.get("transport", 0.0))
            - float(profile_json.get("healthcare", 0.0))
            - float(profile_json.get("other", 0.0))
            - float(profile_json.get("remittances", 0.0))
        )
        has_plan = bool(savings >= 1500 and surplus >= 250 and retirement_priority in ("medium", "high") and self.rng.random() < 0.45)

        return {
            "age": age,
            "city_tier": city_tier,
            "retirement_horizon": horizon,
            "retirement_priority": retirement_priority,
            "has_dedicated_retirement_plan": has_plan,
        }

    # -------------------------
    # Anchors
    # -------------------------

    def sample_anchors(self) -> Dict[str, Any]:
        """Draw demographic anchors from NBP 2025 survey distributions."""
        gender = "female" if self.rng.random() < self.cfg.female_prior else "male"

        # NBP 2025 survey: Ukrainian refugees in Poland (n=3,965)
        # permanent_job 54%, self_employed 4%, other_work 17%, unemployed 14%, economically_inactive 11%
        employment_status = self.rng.choices(
            ["permanent_job", "self_employed", "other_work", "unemployed", "economically_inactive"],
            weights=[0.54, 0.04, 0.17, 0.14, 0.11],
            k=1,
        )[0]

        # Income bands in PLN calibrated to NBP 2025 medians
        # (median female refugee: PLN 3,872; median male: PLN 4,665; total: PLN 4,456)
        if employment_status in ("unemployed", "economically_inactive"):
            income_band = self.rng.choices(
                [(800, 1500), (1500, 2500), (2500, 3500)],
                weights=[0.50, 0.35, 0.15],
                k=1,
            )[0]
        elif gender == "female":
            income_band = self.rng.choices(
                [(2000, 3200), (3200, 4500), (4500, 7000)],
                weights=[0.35, 0.45, 0.20],
                k=1,
            )[0]
        else:  # male
            income_band = self.rng.choices(
                [(2500, 4000), (4000, 5500), (5500, 8500)],
                weights=[0.30, 0.45, 0.25],
                k=1,
            )[0]

        months = self.rng.randint(self.cfg.min_months_in_poland, 48)

        # Polish language proficiency; calibrated to NBP 2025 Fig 11 refugees: none=5%, basic=39%, well=42%, fluent=14%
        if months <= 12:
            lang_weights = [0.15, 0.50, 0.28, 0.07]
        elif months <= 24:
            lang_weights = [0.06, 0.40, 0.42, 0.12]
        else:
            lang_weights = [0.02, 0.32, 0.46, 0.20]
        if employment_status in ("permanent_job", "self_employed"):
            lang_weights = [max(0, w - 0.03) if i < 2 else w + 0.03 for i, w in enumerate(lang_weights)]
        polish_language_level = self.rng.choices(
            ["none", "basic", "intermediate", "advanced"],
            weights=lang_weights,
            k=1,
        )[0]

        # NBP 2025: 36% of refugees send remittances to Ukraine
        if employment_status in ("unemployed", "economically_inactive"):
            rem_flag = self.rng.choices([0, 1], weights=[0.80, 0.20], k=1)[0]
        else:
            dependents_proxy = self.rng.choices([0, 1], weights=[0.55, 0.45], k=1)[0]
            rem_flag = self.rng.choices(
                [0, 1],
                weights=[0.64, 0.36] if dependents_proxy == 0 else [0.50, 0.50],
                k=1,
            )[0]

        dependents = self.rng.choices([0, 1, 2, 3], weights=[0.35, 0.40, 0.20, 0.05], k=1)[0]

        return {
            "gender": gender,
            "dependents": int(dependents),
            "employment_status": employment_status,
            "months": int(months),
            "income_min": income_band[0],
            "income_max": income_band[1],
            "remittances_flag": int(rem_flag),
            "polish_language_level": polish_language_level,
        }

    # -------------------------
    # Deterministic budget generator (PLN)
    # -------------------------

    def generate_profile_numbers(self, anchors: Dict[str, Any]) -> Dict[str, Any]:
        """Build a realistic monthly budget (PLN) from anchors; ensures min surplus for retirement saving."""
        income = self._round_money(self.rng.uniform(anchors["income_min"], anchors["income_max"]))
        dep = anchors["dependents"]
        emp = anchors["employment_status"]

        housing_share = self._triangular(0.30, 0.42, 0.55)
        utilities_share = self._triangular(0.06, 0.09, 0.12)

        food_low, food_mode, food_high = 0.14, 0.20, 0.28
        if dep >= 1:
            food_low += 0.02
            food_mode += 0.03
            food_high += 0.03
        food_share = self._triangular(food_low, food_mode, food_high)

        transport_low, transport_mode, transport_high = 0.04, 0.06, 0.10
        if emp in ("permanent_job", "self_employed"):
            transport_low += 0.01
            transport_mode += 0.02
        transport_share = self._triangular(transport_low, transport_mode, transport_high)

        healthcare_share = self._triangular(0.02, 0.04, 0.07)

        other_low, other_mode, other_high = 0.04, 0.07, 0.12
        if dep >= 1:
            other_high += 0.02
        other_share = self._triangular(other_low, other_mode, other_high)

        # NBP 2025: 36% send remittances, mostly under PLN 1,000/month
        if anchors["remittances_flag"] == 0:
            rem_share = 0.0
        else:
            rem_share = self._triangular(0.05, 0.10, 0.18)
            if dep > 1:
                rem_share = min(rem_share + 0.02, 0.22)

        housing = self._round_money(income * housing_share)
        utilities = self._round_money(income * utilities_share)
        food = self._round_money(income * food_share)
        transport = self._round_money(income * transport_share)
        healthcare = self._round_money(income * healthcare_share)
        other = self._round_money(income * other_share)
        remittances = self._round_money(income * rem_share)

        # retirement surplus target
        if emp in ("permanent_job", "self_employed"):
            surplus_target = max(self.cfg.min_retirement_surplus_pln, self._round_money(income * self._triangular(0.05, 0.07, 0.10)))
        elif emp == "other_work":
            surplus_target = max(self.cfg.min_retirement_surplus_pln, self._round_money(income * self._triangular(0.03, 0.05, 0.07)))
        else:  # unemployed / economically_inactive
            surplus_target = max(self.cfg.min_retirement_surplus_pln, self._round_money(income * self._triangular(0.02, 0.03, 0.05)))

        # Hard floors: minimum realistic PLN values per category
        housing_min = self._round_money(500 + 100 * dep)  # PLN 500/600/700/800
        food_min = self._round_money(400 + 200 * dep)     # PLN 400/600/800/1000
        other_min = self._round_money(120 + 80 * dep)     # PLN 120/200/280/360
        utilities_min = 50.0   # shared flat minimum: electricity+water split
        transport_min = 30.0   # public transport minimum

        housing = max(housing, housing_min)
        food = max(food, food_min)
        other = max(other, other_min)

        expenses = housing + utilities + food + transport + healthcare + other + remittances
        max_expenses = max(0.0, income - surplus_target)

        # Stage 1: cut other but not below other_min
        if expenses > max_expenses and expenses > 0:
            excess = expenses - max_expenses
            reducible_other = max(0.0, other - other_min)
            cut = min(reducible_other, excess)
            other = self._round_money(other - cut)
            other = max(other, other_min)

        expenses = housing + utilities + food + transport + healthcare + other + remittances

        # Stage 2: scale only flexible items; housing/food/other/utilities_min/transport_min stay fixed
        if expenses > max_expenses and expenses > 0:
            fixed_sum = housing + food + other + utilities_min + transport_min
            budget_for_scalable = max(0.0, max_expenses - fixed_sum)
            # scalable portion above the minimums
            scalable_above_min = (
                max(0.0, utilities - utilities_min)
                + max(0.0, transport - transport_min)
                + healthcare
                + remittances
            )
            if scalable_above_min > 0:
                ratio = max(0.0, min(1.0, budget_for_scalable / scalable_above_min))
                utilities = self._round_money(utilities_min + max(0.0, utilities - utilities_min) * ratio)
                transport = self._round_money(transport_min + max(0.0, transport - transport_min) * ratio)
                healthcare = self._round_money(healthcare * ratio)
                remittances = self._round_money(remittances * ratio)
            else:
                utilities = max(utilities, utilities_min)
                transport = max(transport, transport_min)

        expenses = housing + utilities + food + transport + healthcare + other + remittances
        surplus = max(0.0, income - expenses)

        months = int(anchors["months"])
        savings = self._round_money(
            max(0.0, self.rng.uniform(0.5, 2.0) * surplus + self.rng.uniform(0, 0.5) * (months / 12.0) * 200.0)
        )
        if emp in ("unemployed", "economically_inactive") and surplus < 200:
            savings = self._round_money(self.rng.uniform(0, 350))

        return {
            "income": float(income),
            "savings": float(savings),
            "housing": float(housing),
            "utilities": float(utilities),
            "food": float(food),
            "transport": float(transport),
            "healthcare": float(healthcare),
            "other": float(other),
            "remittances": float(remittances),
            "dependents": dep,
            "employment_status": emp,
            "gender": anchors["gender"],
            "months": months,
            "polish_language_level": anchors.get("polish_language_level", "basic"),
        }

    # -------------------------
    # RAG qualitative constraints only
    # -------------------------

    def _make_context(self, query: str) -> str:
        """Retrieve RAG chunks to ground constraint phrase generation."""
        hits = retrieve_chunks(self.vs, query, k=self.cfg.k)
        parts: List[str] = []
        for h in hits:
            src = h.metadata.get("source", "unknown")
            page = h.metadata.get("page", None)
            snippet = (h.page_content or "")[:700].strip()
            parts.append(f"[source={src} page={page}]\n{snippet}")
        return "\n\n---\n\n".join(parts)

    def generate_constraints_text(self, profile_json: Dict[str, Any], anchors: Dict[str, Any]) -> List[str]:
        """Ask the LLM for 2-4 qualitative barrier phrases grounded in RAG context."""
        context = self._make_context(
            "Common constraints/barriers for Ukrainian refugees in Poland related to work, housing, finances, remittances"
        )
        system = (
            "You propose short qualitative constraints for a synthetic refugee persona. "
            "Do NOT change numbers. Return ONLY JSON list of 2 to 4 short strings."
        )
        user = f"""
Return 2-4 realistic constraint phrases as JSON list of strings.

ANCHORS:
{json.dumps(anchors, ensure_ascii=False)}

PROFILE:
{json.dumps(profile_json, ensure_ascii=False)}

CONTEXT:
{context}
""".strip()
        resp = self.llm_c.invoke([("system", system), ("user", user)])
        cleaned = self._strip_code_fences((resp.content or "").strip())
        try:
            arr = json.loads(cleaned)
        except json.JSONDecodeError:
            return ["language barrier", "unstable job"]
        if not isinstance(arr, list):
            return ["language barrier", "unstable job"]

        out: List[str] = []
        for x in arr:
            if isinstance(x, str) and x.strip():
                out.append(x.strip()[:90])
        return out[:4] if out else ["language barrier"]

    # -------------------------
    # Persona (one-time, persistent)
    # -------------------------

    def generate_persona_json(
        self,
        anchors: Dict[str, Any],
        profile_json: Dict[str, Any],
        constraints: List[str],
    ) -> Dict[str, Any]:
        """Generate a latent persona consistent with anchors; applies sanity clamps and savings-based plan flag."""
        controls = self._sample_persona_controls(profile_json)
        system = (
            "You create a synthetic persona for research. "
            "You MUST stay consistent with provided anchors/profile. "
            "Do NOT introduce facts that contradict the profile (income, savings, months, dependents, employment). "
            "Return ONLY valid JSON. No triple backticks."
        )

        user = f"""
Create ONE latent persona consistent with this case.

ANCHORS:
{json.dumps(anchors, ensure_ascii=False)}

PROFILE_JSON (PLN):
{json.dumps(profile_json, ensure_ascii=False)}

QUALITATIVE CONSTRAINTS (phrases):
{json.dumps(constraints, ensure_ascii=False)}

FIXED PERSONA CONTROLS (must use exactly):
{json.dumps(controls, ensure_ascii=False)}

Return JSON with EXACT keys:
- age (int, 25..58)
- education (one of: "high_school", "vocational", "bachelor", "master", "phd")
- occupation (short string, consistent with employment_status; no employer names)
- city_tier (one of: "large", "medium", "small")
- housing_situation (short string, consistent with housing cost)
- financial_literacy (one of: "low", "medium", "high")
- risk_tolerance (one of: "low", "medium", "high")
- retirement_horizon (string like "mid-60s" or "after 65")
- retirement_priority (one of: "low", "medium", "high")
- retirement_goals (array of 2-4 short strings)
- has_dedicated_retirement_plan (boolean)
- knows_if_enrolled_private_plan (one of: "yes", "no", "not_sure")
- polish_language_level (must match anchors: one of "none", "basic", "intermediate", "advanced")
- notes (1 short sentence, no PII)

Rules:
- retirement_goals must always have 2-4 items (even if general), e.g. "cover essentials", "healthcare buffer", "avoid debt", "independent living".
- If savings are low (<1500 PLN), prefer has_dedicated_retirement_plan=false.
- If employment_status is "unemployed" or "economically_inactive", occupation should reflect it (e.g., "currently unemployed, seeking work").
- If polish_language_level is "none" or "basic", financial_literacy should lean toward "low" or "medium".
- Keep it realistic and concise.
""".strip()

        resp = self.llm_p.invoke([("system", system), ("user", user)])
        cleaned = self._strip_code_fences((resp.content or "").strip())
        try:
            persona = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Persona returned non-JSON:\n{cleaned}") from e

        required = [
            "age", "education", "occupation", "city_tier", "housing_situation",
            "financial_literacy", "risk_tolerance", "retirement_horizon",
            "retirement_priority", "retirement_goals",
            "has_dedicated_retirement_plan", "knows_if_enrolled_private_plan",
            "polish_language_level", "notes"
        ]
        for k in required:
            if k not in persona:
                raise RuntimeError(f"Persona missing key '{k}': {persona}")

        # sanity / defaults
        persona["age"] = int(controls["age"])

        if persona.get("city_tier") not in ("large", "medium", "small"):
            persona["city_tier"] = controls["city_tier"]
        persona["city_tier"] = controls["city_tier"]

        if persona.get("retirement_priority") not in ("low", "medium", "high"):
            persona["retirement_priority"] = controls["retirement_priority"]
        persona["retirement_priority"] = controls["retirement_priority"]
        persona["retirement_horizon"] = controls["retirement_horizon"]

        if not isinstance(persona.get("retirement_goals"), list) or len(persona["retirement_goals"]) < 2:
            persona["retirement_goals"] = [
                "cover basic living costs",
                "have a buffer for healthcare",
                "avoid financial stress in old age",
            ]
        persona["retirement_goals"] = [str(x).strip()[:80] for x in persona["retirement_goals"] if str(x).strip()][:4]

        persona["has_dedicated_retirement_plan"] = bool(controls["has_dedicated_retirement_plan"])

        if persona.get("knows_if_enrolled_private_plan") not in ("yes", "no", "not_sure"):
            persona["knows_if_enrolled_private_plan"] = "not_sure"

        # enforce consistency with anchors
        if persona.get("polish_language_level") not in ("none", "basic", "intermediate", "advanced"):
            persona["polish_language_level"] = profile_json.get("polish_language_level", "basic")

        return persona

    # -------------------------
    # User question
    # -------------------------

    def profile_to_text(self, p: Dict[str, Any]) -> str:
        """Format profile_json as human-readable text for injection into the LLM prompt."""
        return f"""
Monthly income: PLN {p['income']:.0f}
Current savings: PLN {p['savings']:.0f}
Monthly expenses:
- Housing: PLN {p['housing']:.0f}
- Utilities: PLN {p['utilities']:.0f}
- Food: PLN {p['food']:.0f}
- Transportation: PLN {p['transport']:.0f}
- Healthcare: PLN {p['healthcare']:.0f}
- Other: PLN {p['other']:.0f}
- Remittances: PLN {p['remittances']:.0f}
Has {p['dependents']} dependent(s)
Employment status: {p.get('employment_status', 'unknown')}
Gender: {p['gender']}
Lives in Poland for {p['months']} month(s)
Polish language level: {p.get('polish_language_level', 'basic')}
""".strip()

    def generate_retirement_question(self, profile_text: str, persona: Optional[Dict[str, Any]] = None) -> str:
        """Ask the model to write exactly one retirement savings question in the persona's voice."""
        if persona:
            priority = str(persona.get("retirement_priority", "medium"))
            templates = [
                "Given my financial profile, how much should I save monthly for retirement in Poland and what retirement-related options should I consider?",
                "Can I afford to start saving for retirement now in Poland, and what monthly amount would be realistic for me?",
                "Should I focus on emergency savings first, or can I safely put part of my surplus toward retirement in Poland?",
                "How much of my monthly surplus can I put aside for retirement without making my budget too tight?",
                "What should I do for retirement planning in Poland if I am not sure my ZUS contributions are being recorded correctly?",
                "How can I save for retirement in Poland while still covering dependents and remittances?",
                "What is a realistic first retirement-saving target for me in Poland, and which official options should I check?",
            ]
            if priority == "high":
                templates.extend([
                    "I want to make retirement a high priority; how much should I save monthly in Poland without risking my current budget?",
                    "What practical retirement steps should I take this year in Poland, and what monthly saving amount fits my profile?",
                ])
            return self.rng.choice(templates)

        system = "You are a user asking ONE question to a financial advisor. Return only the question."
        persona_hint = ""
        if persona:
            persona_hint = (
                f"\nPersona hint: age={persona.get('age')}, horizon={persona.get('retirement_horizon')}, "
                f"priority={persona.get('retirement_priority')}, literacy={persona.get('financial_literacy')}.\n"
            )

        user = f"""
Here is my finance profile:
{profile_text}
{persona_hint}

Write exactly ONE question ONLY about saving for retirement in Poland.
Ask:
- how much to save monthly given this profile
- which retirement-related options to consider
One sentence, end with '?'.
""".strip()

        resp = self.llm_q.invoke([("system", system), ("user", user)])
        q = " ".join((resp.content or "").strip().splitlines()).strip()
        if not q.endswith("?"):
            q += "?"
        return q

    # -------------------------
    # Clarifying answers: rule-based first, LLM fallback second
    # -------------------------

    def _preanswer_common(
        self,
        anchors: Dict[str, Any],
        profile_json: Dict[str, Any],
        persona_json: Dict[str, Any],
        question: str,
    ) -> Optional[str]:
        """Rule-based answers for predictable question types; avoids LLM drift on goals, employment, and language."""
        ql = question.lower().strip()

        emp = anchors.get("employment_status", "unemployed")
        months = int(profile_json.get("months", anchors.get("months", self.cfg.min_months_in_poland)))
        savings = float(profile_json.get("savings", 0.0))

        age = int(persona_json.get("age", 35))
        horizon = str(persona_json.get("retirement_horizon", "after 65")).strip()
        priority = str(persona_json.get("retirement_priority", "medium")).strip()
        goals = persona_json.get("retirement_goals", [])
        goals_txt = "; ".join([str(x) for x in goals[:4]]) if isinstance(goals, list) and goals else "cover essentials; healthcare buffer; avoid financial stress"

        has_plan = bool(persona_json.get("has_dedicated_retirement_plan", False))
        enrolled = str(persona_json.get("knows_if_enrolled_private_plan", "not_sure"))

        lang_level = str(anchors.get("polish_language_level", profile_json.get("polish_language_level", "basic")))
        lang_descriptions = {
            "none": "I don't speak Polish yet. I communicate mostly in Ukrainian or Russian, and I rely on colleagues or apps to translate.",
            "basic": "I speak only basic Polish - enough for shopping and simple conversations, but I struggle with official documents and complex discussions.",
            "intermediate": "My Polish is at an intermediate level. I manage daily life and work tasks, but financial or legal language is still challenging.",
            "advanced": "I speak Polish at an advanced level and can handle most situations independently, though some financial and legal terms still require clarification.",
        }

        # Polish language level
        if "language" in ql or "polish" in ql or "speak" in ql or "communication" in ql:
            return f"{lang_descriptions.get(lang_level, lang_descriptions['basic'])} This sometimes makes it harder to navigate banking, pension offices, or official procedures in Poland."

        # employment
        if "employment status" in ql or "self-employed" in ql or ("currently employed" in ql):
            emp_label = {
                "permanent_job": "permanently employed",
                "self_employed": "self-employed",
                "other_work": "employed part-time or on a temporary contract",
                "unemployed": "currently unemployed",
                "economically_inactive": "currently not in the labour market",
            }.get(emp, emp)
            return f"I am {emp_label}. My situation has been stable enough to cover essentials, but I'm still cautious about taking on new fixed costs."

        # age / retirement horizon
        if ("current age" in ql) or ("how old" in ql) or (ql.startswith("what is your age")) or ("age" in ql and "retire" not in ql):
            return f"I'm {age} years old. I'm trying to make decisions that are realistic for my current income level and still move me toward long-term stability."

        if ("planned retirement age" in ql) or ("when do you plan to retire" in ql) or ("at what age" in ql and "retire" in ql):
            return f"I'm thinking about retiring {horizon}. I don't have an exact plan yet, so I'd prefer a simple approach that I can maintain consistently."

        # time in Poland
        if "how long" in ql or "months" in ql or "time in poland" in ql:
            return f"I have lived in Poland for about {months} months. I'm still adapting and trying to stabilize my finances while learning how the pension system works here."

        # existing retirement savings / pension plans
        if "retirement savings" in ql or "pension plan" in ql or "pension plans" in ql:
            if has_plan:
                return "I have started thinking about retirement more seriously and I try to set something aside, but it's still quite basic. I'm looking for a clearer structure and guidance on what options make sense in Poland."
            return f"I don't have a dedicated retirement plan yet - my savings are mostly general (PLN {savings:.0f}). I rely on standard work-related pension contributions, and I want to build a separate habit for retirement savings."

        # enrollment private plans (PPK/IKE/IKZE)
        if "ppk" in ql or "ike" in ql or "ikze" in ql or "private plan" in ql:
            if enrolled == "yes":
                return "As far as I know, I am enrolled in a private plan through my work, but I'm not confident about the details. I'd like help understanding what I'm contributing and whether I should adjust it."
            if enrolled == "no":
                return "I'm not enrolled in any private retirement plan at the moment. If there are safe and suitable options in Poland, I'd like to understand how they work and how to check eligibility."
            return "I'm not sure whether I'm enrolled in any private retirement plan. I haven't actively signed up myself, so I would need to check with my employer or official records."

        # willingness to adjust expenses
        if "open to adjusting" in ql or ("adjust" in ql and "expenses" in ql) or "cut back" in ql:
            return "Yes, I'm open to adjusting my expenses in small steps, as long as it doesn't reduce basic quality of life. I'd prefer realistic changes - like trimming 'other' spending or optimizing utilities - rather than major cuts."

        # long-term goals
        if "long-term" in ql and "goal" in ql and "retirement" in ql:
            return f"My retirement goals are: {goals_txt}. My priority is {priority}, and I'm thinking about retirement {horizon}. I want a plan that is simple, low-stress, and realistic with my current income."

        if "goals" in ql and "retirement" in ql:
            return f"My main goals are: {goals_txt}. I'm aiming for {horizon} and I want to avoid financial stress later in life."

        return None

    def _llm_answer_remaining(
        self,
        anchors: Dict[str, Any],
        profile_json: Dict[str, Any],
        persona_json: Dict[str, Any],
        questions: List[str],
        max_q: int = 3,
    ) -> List[Dict[str, str]]:
        """LLM fallback strictly constrained to anchors + profile + persona — no new facts allowed."""
        system = (
            "You are simulating a refugee user answering clarifying questions. "
            "CRITICAL: You MUST answer using ONLY the provided ANCHORS, PROFILE_JSON, and PERSONA_JSON fields. "
            "You are NOT allowed to invent new facts (no new pension enrollments, no new investments, no new income sources). "
            "If information is missing, say 'I'm not sure' AND provide the most likely preference consistent with the persona. "
            "Answers should be practical and human (2-4 sentences). No PII. "
            "Return ONLY valid JSON list."
        )

        user = f"""
ANCHORS:
{json.dumps(anchors, ensure_ascii=False)}

PROFILE_JSON:
{json.dumps(profile_json, ensure_ascii=False)}

PERSONA_JSON:
{json.dumps(persona_json, ensure_ascii=False)}

QUESTIONS:
{json.dumps(questions[:max_q], ensure_ascii=False)}

Return ONLY JSON list of objects:
[
  {{"question":"...","answer":"..."}},
  ...
]
""".strip()

        resp = self.llm_a.invoke([("system", system), ("user", user)])
        cleaned = self._strip_code_fences((resp.content or "").strip())

        try:
            qa = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Clarify answers returned non-JSON:\n{cleaned}") from e

        if not isinstance(qa, list):
            raise RuntimeError(f"Clarify answers not a list: {qa}")

        out: List[Dict[str, str]] = []
        for item in qa[:max_q]:
            if not isinstance(item, dict):
                continue
            q = str(item.get("question", "")).strip()
            a = str(item.get("answer", "")).strip()
            if q and a:
                out.append({"question": q, "answer": a})
        return out

    def answer_clarifying_questions(
        self,
        anchors: Dict[str, Any],
        profile_json: Dict[str, Any],
        persona_json: Dict[str, Any],
        questions: List[str],
        max_q: int = 3,
    ) -> List[Dict[str, str]]:
        """Route common questions to rule-based answers; delegate the rest to the LLM fallback."""
        qs = [q.strip() for q in questions if q and q.strip()][:max_q]
        answered: List[Dict[str, str]] = []
        remaining: List[str] = []
        seen_answers: set = set()

        for q in qs:
            ans = self._preanswer_common(anchors, profile_json, persona_json, q)
            if ans is None or ans in seen_answers:
                remaining.append(q)
            else:
                seen_answers.add(ans)
                answered.append({"question": q, "answer": ans})

        if remaining:
            llm_answers = self._llm_answer_remaining(anchors, profile_json, persona_json, remaining, max_q=len(remaining))
            # merge in original order
            merged: List[Dict[str, str]] = []
            a_map = {x["question"]: x["answer"] for x in answered}
            l_map = {x["question"]: x["answer"] for x in llm_answers}
            for q in qs:
                if q in a_map:
                    merged.append({"question": q, "answer": a_map[q]})
                elif q in l_map:
                    merged.append({"question": q, "answer": l_map[q]})
                else:
                    merged.append({"question": q, "answer": "I'm not sure, but I can clarify if you tell me what information you need."})
            return merged

        return answered

    # -------------------------
    # Run
    # -------------------------

    def run(self) -> Dict[str, Any]:
        """Generate one complete synthetic case: anchors -> profile -> persona -> question."""
        anchors = self.sample_anchors()
        profile_json = self.generate_profile_numbers(anchors)
        constraints = self.generate_constraints_text(profile_json, anchors)
        persona_json = self.generate_persona_json(anchors, profile_json, constraints)

        profile_text = self.profile_to_text(profile_json)
        user_query = self.generate_retirement_question(profile_text, persona_json)

        return {
            "anchors": anchors,
            "constraints": constraints,
            "persona_json": persona_json,
            "profile_json": profile_json,
            "profile_text": profile_text,
            "user_query": user_query,
        }
