"""
05_live_lesson_generation.py

Called by the Express server during the live demo pipeline.
Reads a temp input JSON (path given as first CLI arg), generates
a personalised lesson plan via OpenRouter, and prints ONLY valid
JSON to stdout so Express can parse it.

Usage:
    python src/04_llm/05_live_lesson_generation.py <path_to_temp_input.json>
"""

import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT         = Path(__file__).resolve().parents[2]
SKILL_LOOKUP = ROOT / "data" / "processed" / "skill_id_to_name.json"

FALLBACK_MODELS = [
    "google/gemini-2.5-flash",
    "meta-llama/llama-3.1-8b-instruct",
    "anthropic/claude-3.5-haiku",
    "mistralai/mistral-7b-instruct",
]

def load_skill_name(skill_id):
    if not SKILL_LOOKUP.exists():
        return f"Skill {skill_id}"
    with open(SKILL_LOOKUP, "r", encoding="utf-8") as f:
        lookup = json.load(f)
    return lookup.get(str(skill_id), f"Skill {skill_id}")

def call_llm(client, system_prompt, user_prompt):
    for model in FALLBACK_MODELS:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                extra_headers={
                    "HTTP-Referer": "http://localhost:8000",
                    "X-Title":      "ACDT Capstone Project",
                },
                max_tokens=512,
                timeout=25.0,
            )
            raw     = resp.choices[0].message.content
            cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
            plan    = json.loads(cleaned)
            plan["_meta"] = {"model_used": model}
            return plan
        except Exception:
            continue
    return None

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No input file path provided."}))
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(json.dumps({"error": f"Input file not found: {input_path}"}))
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        inp = json.load(f)

    frustration   = float(inp.get("frustration",   0.4))
    confusion     = float(inp.get("confusion",     0.1))
    concentration = float(inp.get("concentration", 0.6))
    boredom       = float(inp.get("boredom",       0.2))
    correctness   = float(inp.get("correctness",   0.8))
    problem_id    = int(inp.get("problem_id",   606850))
    skill_id      = int(inp.get("skill_id",         50))

    skill_name = load_skill_name(skill_id)

    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print(json.dumps({"error": "OPENROUTER_API_KEY not set."}))
        sys.exit(1)

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    system_prompt = (
        "You are an empathetic, expert AI pedagogical tutor for middle-school math students. "
        "You produce structured, personalised lesson plans in valid JSON. "
        "Always respond with ONLY a JSON object — no markdown fences, no extra text."
    )

    user_prompt = f"""
A student is about to practise the topic: "{skill_name}".

Predicted Student State (Dual-State Digital Twin):
- Cognitive Mastery (Expected Correctness): {correctness:.0%}
- Frustration   : {frustration:.2f} / 1.0
- Confusion     : {confusion:.2f} / 1.0
- Concentration : {concentration:.2f} / 1.0
- Boredom       : {boredom:.2f} / 1.0

Generate a structured lesson plan as a JSON object with EXACTLY these keys:

{{
  "topic": "{skill_name}",
  "student_message": "<2-3 sentence warm, personalised message spoken directly to the student. Adapt tone to their affect. Do NOT mention raw numbers.>",
  "pedagogical_strategy": "<1-2 sentences: what instructional approach is being used and why.>",
  "micro_lesson": "<2-3 sentence conceptual refresher on the topic for the student.>",
  "hints": [
    "<Tier 1 hint: a gentle guiding question>",
    "<Tier 2 hint: a concrete strategy or worked step>"
  ],
  "teacher_note": "<1-2 sentences for the teacher dashboard: summarise the student's current dual state (cognitive + affective) and the rationale for this recommendation.>"
}}
"""

    plan = call_llm(client, system_prompt, user_prompt)
    if plan is None:
        print(json.dumps({"error": "All LLM fallbacks failed."}))
        sys.exit(1)

    # Output ONLY the JSON — Express parses stdout
    print(json.dumps(plan, ensure_ascii=False))

if __name__ == "__main__":
    main()
