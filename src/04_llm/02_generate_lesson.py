"""
02_generate_lesson.py

Reads the MCTS recommended plan, enriches each step with the real skill name
from our lookup files, then calls OpenRouter to generate a full structured
JSON lesson plan for the first recommended activity.
"""

import json
import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT             = Path(__file__).resolve().parents[2]
PLAN_FILE        = ROOT / "results" / "mcts" / "mcts_fair_plan.csv"
SKILL_LOOKUP     = ROOT / "data"    / "processed" / "skill_id_to_name.json"
PROBLEM_LOOKUP   = ROOT / "data"    / "processed" / "problem_to_skill.json"
OUTPUT_FILE      = ROOT / "results" / "llm" / "lesson_plan.json"

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

FALLBACK_MODELS = [
    "google/gemini-2.5-flash",
    "meta-llama/llama-3.1-8b-instruct",
    "anthropic/claude-3.5-haiku",
    "mistralai/mistral-7b-instruct",
]

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_lookups():
    print("Loading skill lookup files ...")
    with open(SKILL_LOOKUP, "r", encoding="utf-8") as f:
        skill_id_to_name = {int(k): v for k, v in json.load(f).items()}
    with open(PROBLEM_LOOKUP, "r", encoding="utf-8") as f:
        problem_to_skill = {int(k): v for k, v in json.load(f).items()}
    return skill_id_to_name, problem_to_skill


def resolve_skill_name(problem_id, skill_id, problem_to_skill, skill_id_to_name):
    """Return the best human-readable skill name we can find."""
    # First try: direct problem_id lookup (most specific)
    info = problem_to_skill.get(int(problem_id))
    if info and info.get("skill_name"):
        return info["skill_name"]
    # Second try: skill_id lookup
    name = skill_id_to_name.get(int(skill_id))
    if name:
        return name
    return "Unknown Topic"


def build_affect_description(frustration, confusion, concentration, boredom):
    """Convert numeric affect scores into a plain-English description."""
    dominant = max(
        ("frustration",   frustration),
        ("confusion",     confusion),
        ("concentration", concentration),
        ("boredom",       boredom),
        key=lambda x: x[1]
    )[0]

    templates = {
        "frustration":   "The student appears frustrated and may need emotional support before diving into the problem.",
        "confusion":     "The student seems confused and would benefit from a clear conceptual reminder before attempting.",
        "concentration": "The student is focused and in the zone — keep momentum high with a direct, confident transition.",
        "boredom":       "The student appears bored and disengaged. A motivating challenge or novel framing may help.",
    }
    return templates[dominant]


def call_llm_with_fallbacks(client, system_prompt, user_prompt):
    """Try each model in order and return the first successful response."""
    for model_name in FALLBACK_MODELS:
        try:
            print(f"  Trying model: {model_name} ...")
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                extra_headers={
                    "HTTP-Referer": "http://localhost:8000",
                    "X-Title":      "ACDT Capstone Project",
                },
                max_tokens=512,
                timeout=20.0,
            )
            content = response.choices[0].message.content
            print(f"  Success with: {model_name}")
            return content, model_name
        except Exception as e:
            print(f"  Failed ({model_name}): {e}")

    return None, None


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # 1. Load API key
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not found in .env file.")
        return

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    # 2. Load lookup tables
    skill_id_to_name, problem_to_skill = load_lookups()

    # 3. Load MCTS plan
    if not PLAN_FILE.exists():
        print(f"ERROR: MCTS plan not found at {PLAN_FILE}")
        return

    print(f"Loading MCTS plan from {PLAN_FILE.name} ...")
    plan_df = pd.read_csv(PLAN_FILE)

    # We generate a lesson for the FIRST recommended step
    step_data    = plan_df.iloc[0]
    problem_id   = int(step_data["original_problem_id"])
    skill_id     = int(step_data["original_skill_id"])
    correctness  = float(step_data["predicted_correctness"])
    frustration  = float(step_data["predicted_frustration"])
    confusion    = float(step_data["predicted_confusion"])
    concentration = float(step_data["predicted_concentration"])
    boredom      = float(step_data["predicted_boredom"])

    # 4. Resolve human-readable skill name
    skill_name      = resolve_skill_name(problem_id, skill_id, problem_to_skill, skill_id_to_name)
    affect_desc     = build_affect_description(frustration, confusion, concentration, boredom)

    print("\n--- Enriched Student Context ---")
    print(f"  Next Activity  : Problem {problem_id}")
    print(f"  Topic (Skill)  : {skill_name}  (Skill ID {skill_id})")
    print(f"  Predicted Mastery (Correctness): {correctness:.0%}")
    print(f"  Affective State: Frustration={frustration:.2f} | Confusion={confusion:.2f} | Concentration={concentration:.2f} | Boredom={boredom:.2f}")
    print(f"  Affect Summary : {affect_desc}")
    print("-" * 50)

    # 5. Build prompts
    system_prompt = (
        "You are an empathetic, expert AI pedagogical tutor for middle-school math students. "
        "You produce structured, personalised lesson plans in valid JSON. "
        "Always respond with ONLY a JSON object — no markdown fences, no extra text."
    )

    user_prompt = f"""
A student is about to practise the topic: "{skill_name}".

Predicted Student State:
- Expected Correctness (Cognitive Mastery): {correctness:.0%}
- Frustration   : {frustration:.2f} / 1.0
- Confusion     : {confusion:.2f} / 1.0
- Concentration : {concentration:.2f} / 1.0
- Boredom       : {boredom:.2f} / 1.0
- Affect Summary: {affect_desc}

Generate a structured lesson plan as a JSON object with EXACTLY these keys:

{{
  "topic": "<skill name>",
  "student_message": "<2-3 sentence warm, personalised message spoken directly to the student. Adapt tone to their affect. Do NOT mention raw numbers.>",
  "pedagogical_strategy": "<1-2 sentences: what instructional approach is being used and why, e.g. scaffolded practice, spaced retrieval, etc.>",
  "micro_lesson": "<2-3 sentence conceptual refresher on the topic that the student can read before attempting the problem.>",
  "hints": [
    "<Tier 1 hint: a gentle guiding question>",
    "<Tier 2 hint: a concrete strategy or worked step>"
  ],
  "teacher_note": "<1-2 sentences for the teacher dashboard: summarise the student's current state and the rationale for this activity recommendation.>"
}}
"""

    # 6. Call LLM
    print("\nCalling OpenRouter LLM ...")
    raw_response, model_used = call_llm_with_fallbacks(client, system_prompt, user_prompt)

    if raw_response is None:
        print("\nCRITICAL: All fallback models failed. Cannot generate lesson plan.")
        return

    # 7. Parse JSON from LLM response
    # Strip any accidental markdown fences just in case
    cleaned = re.sub(r"```(?:json)?|```", "", raw_response).strip()
    try:
        lesson_plan = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"\nWARNING: Could not parse LLM response as JSON: {e}")
        print("Raw response:\n", raw_response)
        return

    # 8. Attach metadata
    lesson_plan["_meta"] = {
        "problem_id":    problem_id,
        "skill_id":      skill_id,
        "model_used":    model_used,
        "mcts_step":     1,
    }

    # 9. Display
    print("\n" + "=" * 56)
    print("FULL STRUCTURED LESSON PLAN (JSON)")
    print("=" * 56)
    print(json.dumps(lesson_plan, indent=2, ensure_ascii=False))
    print("=" * 56)

    # 10. Save to results/llm/lesson_plan.json
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(lesson_plan, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
