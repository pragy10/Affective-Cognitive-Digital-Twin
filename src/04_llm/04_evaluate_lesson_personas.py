"""
04_evaluate_lesson_personas.py

Phase 6.4 — Multi-Persona Lesson Plan Evaluation

Tests the LLM lesson planner against 4 distinct student affective personas
using the same MCTS-recommended problem (Ordering Fractions, Problem 606850).
This demonstrates that the system adapts its pedagogical strategy and tone
based on the student's emotional state — not just their cognitive mastery.

Output:
  results/llm/persona_comparison.json      <- all 4 lesson plans (machine-readable)
  results/llm/persona_comparison_report.md <- readable comparison for Review 3 report
"""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
SKILL_LOOKUP = ROOT / "data"    / "processed" / "skill_id_to_name.json"
OUT_JSON    = ROOT / "results"  / "llm" / "persona_comparison.json"
OUT_MD      = ROOT / "results"  / "llm" / "persona_comparison_report.md"
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

FALLBACK_MODELS = [
    "google/gemini-2.5-flash",
    "meta-llama/llama-3.1-8b-instruct",
    "anthropic/claude-3.5-haiku",
    "mistralai/mistral-7b-instruct",
]

# ── 4 Distinct Affective Personas ────────────────────────────────────────────
# All share the same problem/skill so the ONLY variable is the affect state.
# Correctness is kept at 0.55 for Frustrated/Confused (struggling) and
# 0.97 for Bored/Concentrated (high mastery) to make the scenarios realistic.

PERSONAS = [
    {
        "name":          "Frustrated Learner",
        "description":   "High frustration, high confusion. Student is struggling and emotionally dysregulated.",
        "problem_id":    606850,
        "skill_id":      50,
        "skill_name":    "Ordering Fractions",
        "correctness":   0.48,
        "frustration":   0.87,
        "confusion":     0.61,
        "concentration": 0.12,
        "boredom":       0.05,
    },
    {
        "name":          "Confused Learner",
        "description":   "High confusion, moderate frustration. Student wants to understand but is stuck.",
        "problem_id":    606850,
        "skill_id":      50,
        "skill_name":    "Ordering Fractions",
        "correctness":   0.52,
        "frustration":   0.38,
        "confusion":     0.79,
        "concentration": 0.25,
        "boredom":       0.08,
    },
    {
        "name":          "Bored Learner",
        "description":   "High boredom, high mastery. Student has already mastered this and is disengaged.",
        "problem_id":    606850,
        "skill_id":      50,
        "skill_name":    "Ordering Fractions",
        "correctness":   0.96,
        "frustration":   0.07,
        "confusion":     0.03,
        "concentration": 0.22,
        "boredom":       0.88,
    },
    {
        "name":          "Engaged / Concentrated Learner",
        "description":   "High concentration, high mastery. Student is in a flow state.",
        "problem_id":    606850,
        "skill_id":      50,
        "skill_name":    "Ordering Fractions",
        "correctness":   0.97,
        "frustration":   0.40,
        "confusion":     0.04,
        "concentration": 0.65,
        "boredom":       0.19,
    },
]

# ── Helpers ──────────────────────────────────────────────────────────────────

def build_user_prompt(p):
    return f"""
A student is about to practise the topic: "{p['skill_name']}".

Predicted Student State:
- Expected Correctness (Cognitive Mastery): {p['correctness']:.0%}
- Frustration   : {p['frustration']:.2f} / 1.0
- Confusion     : {p['confusion']:.2f} / 1.0
- Concentration : {p['concentration']:.2f} / 1.0
- Boredom       : {p['boredom']:.2f} / 1.0

Generate a structured lesson plan as a JSON object with EXACTLY these keys:

{{
  "topic": "<skill name>",
  "student_message": "<2-3 sentence warm, personalised message spoken directly to the student. Adapt tone to their affect. Do NOT mention raw numbers.>",
  "pedagogical_strategy": "<1-2 sentences: what instructional approach is being used and why.>",
  "micro_lesson": "<2-3 sentence conceptual refresher on the topic for the student.>",
  "hints": [
    "<Tier 1 hint: a gentle guiding question>",
    "<Tier 2 hint: a concrete strategy or worked step>"
  ],
  "teacher_note": "<1-2 sentences for the teacher dashboard: summarise the student's current state and the rationale for this recommendation.>"
}}
"""

SYSTEM_PROMPT = (
    "You are an empathetic, expert AI pedagogical tutor for middle-school math students. "
    "You produce structured, personalised lesson plans in valid JSON. "
    "Always respond with ONLY a JSON object — no markdown fences, no extra text."
)


def call_llm(client, user_prompt):
    for model in FALLBACK_MODELS:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                extra_headers={
                    "HTTP-Referer": "http://localhost:8000",
                    "X-Title":      "ACDT Capstone Project",
                },
                max_tokens=512,
                timeout=20.0,
            )
            raw = resp.choices[0].message.content
            cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
            plan = json.loads(cleaned)
            return plan, model
        except json.JSONDecodeError:
            print(f"    JSON parse error for {model}, skipping.")
        except Exception as e:
            print(f"    {model} failed: {e}")
    return None, None


def render_markdown(results):
    lines = [
        "# ACDT Phase 6.4 — Multi-Persona Lesson Plan Evaluation",
        "",
        "**Topic under test:** Ordering Fractions (Problem 606850, Skill ID 50)",
        "",
        "The same MCTS-recommended problem is presented to four distinct affective",
        "personas. Only the student's emotional/cognitive state changes. This table",
        "demonstrates that the ACDT system adapts its pedagogical strategy, tone,",
        "and hint depth based on the detected affect — not just raw correctness.",
        "",
        "---",
        "",
    ]

    for r in results:
        p    = r["persona"]
        plan = r["lesson_plan"]
        lines += [
            f"## Persona: {p['name']}",
            "",
            f"> **Scenario:** {p['description']}",
            "",
            "### Affective State",
            "",
            "| Dimension     | Value |",
            "|---------------|-------|",
            f"| Correctness   | {p['correctness']:.0%} |",
            f"| Frustration   | {p['frustration']:.2f} |",
            f"| Confusion     | {p['confusion']:.2f} |",
            f"| Concentration | {p['concentration']:.2f} |",
            f"| Boredom       | {p['boredom']:.2f} |",
            "",
            "### Generated Lesson Plan",
            "",
            f"**Student Message:**",
            f"> {plan.get('student_message', 'N/A')}",
            "",
            f"**Pedagogical Strategy:**",
            f"> {plan.get('pedagogical_strategy', 'N/A')}",
            "",
            f"**Micro-Lesson (Conceptual Refresher):**",
            f"> {plan.get('micro_lesson', 'N/A')}",
            "",
            "**Hints:**",
        ]
        hints = plan.get("hints", [])
        for i, h in enumerate(hints, 1):
            lines.append(f"- Tier {i}: {h}")
        lines += [
            "",
            f"**Teacher Dashboard Note:**",
            f"> {plan.get('teacher_note', 'N/A')}",
            "",
            f"*(Model used: `{r['model_used']}`)*",
            "",
            "---",
            "",
        ]

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not found in .env file.")
        return

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    all_results = []

    for i, persona in enumerate(PERSONAS, 1):
        print(f"\n{'='*56}")
        print(f"Persona {i}/{len(PERSONAS)}: {persona['name']}")
        print(f"  Frustration={persona['frustration']:.2f} | "
              f"Confusion={persona['confusion']:.2f} | "
              f"Concentration={persona['concentration']:.2f} | "
              f"Boredom={persona['boredom']:.2f}")
        print(f"  Correctness={persona['correctness']:.0%}")
        print(f"Generating lesson plan ...")

        user_prompt = build_user_prompt(persona)
        plan, model_used = call_llm(client, user_prompt)

        if plan is None:
            print(f"  FAILED: Could not generate plan for {persona['name']}")
            continue

        result = {
            "persona":     persona,
            "lesson_plan": plan,
            "model_used":  model_used,
        }
        all_results.append(result)

        # Print a quick preview
        print(f"  Student Message  : {plan.get('student_message','')[:90]}...")
        print(f"  Strategy         : {plan.get('pedagogical_strategy','')[:80]}...")
        print(f"  Teacher Note     : {plan.get('teacher_note','')[:80]}...")

    # ── Save JSON ────────────────────────────────────────────────────────────
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved JSON  : {OUT_JSON}")

    # ── Save Markdown report ─────────────────────────────────────────────────
    md_content = render_markdown(all_results)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved Report: {OUT_MD}")

    print(f"\nDone! Generated {len(all_results)}/{len(PERSONAS)} lesson plans.")


if __name__ == "__main__":
    main()
