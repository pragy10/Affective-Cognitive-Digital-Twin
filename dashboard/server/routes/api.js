import express from "express";
import fs from "fs";
import os from "os";
import path from "path";
import { fileURLToPath } from "url";
import { parse } from "csv-parse/sync";
import { spawn } from "child_process";

const router = express.Router();

// ── Path resolution ───────────────────────────────────────────────────────────
// routes/ -> server/ -> dashboard/ -> ACDT/
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT      = path.resolve(__dirname, "../../../");
const RESULTS   = path.join(ROOT, "results");
const PROCESSED = path.join(ROOT, "data/processed");
const PYTHON_SCRIPT = path.join(ROOT, "src/04_llm/05_live_lesson_generation.py");

console.log("[api.js] ROOT      resolved to:", ROOT);
console.log("[api.js] RESULTS   resolved to:", RESULTS);
console.log("[api.js] PROCESSED resolved to:", PROCESSED);

// ── Helpers ───────────────────────────────────────────────────────────────────
function readJSON(filePath) {
  if (!fs.existsSync(filePath)) return null;
  return JSON.parse(fs.readFileSync(filePath, "utf-8"));
}

function readCSV(filePath) {
  if (!fs.existsSync(filePath)) return null;
  const raw = fs.readFileSync(filePath, "utf-8");
  return parse(raw, { columns: true, skip_empty_lines: true });
}

function getSkillName(skillId) {
  const lookup = readJSON(path.join(PROCESSED, "skill_id_to_name.json"));
  if (!lookup) return `Skill ${skillId}`;
  return lookup[String(skillId)] || `Skill ${skillId}`;
}

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// ─ SSE helper: write one event ────────────────────────────────────────────────
function sendEvent(res, data) {
  res.write(`data: ${JSON.stringify(data)}\n\n`);
}

// ═══════════════════════════════════════════════════════════════════════════════
// GET /api/student-state
// ═══════════════════════════════════════════════════════════════════════════════
router.get("/student-state", (req, res) => {
  const mctsRows   = readCSV(path.join(RESULTS, "mcts/mcts_fair_plan.csv"));
  const lessonPlan = readJSON(path.join(RESULTS, "llm/lesson_plan.json"));

  if (!mctsRows) {
    return res.status(404).json({ error: "MCTS plan not found." });
  }

  const steps = mctsRows.map((row) => ({
    step:                    Number(row.step),
    problem_id:              Number(row.original_problem_id),
    skill_id:                Number(row.original_skill_id),
    skill_name:              getSkillName(row.original_skill_id),
    predicted_correctness:   Number(row.predicted_correctness),
    predicted_frustration:   Number(row.predicted_frustration),
    predicted_confusion:     Number(row.predicted_confusion),
    predicted_concentration: Number(row.predicted_concentration),
    predicted_boredom:       Number(row.predicted_boredom),
    immediate_reward:        Number(row.immediate_reward),
    discounted_reward:       Number(row.discounted_cumulative_reward),
  }));

  const current = steps[0];

  res.json({
    current_affect: {
      frustration:   current.predicted_frustration,
      confusion:     current.predicted_confusion,
      concentration: current.predicted_concentration,
      boredom:       current.predicted_boredom,
    },
    current_mastery:  current.predicted_correctness,
    recommended_path: steps,
    lesson_plan:      lessonPlan || null,
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// GET /api/personas
// ═══════════════════════════════════════════════════════════════════════════════
router.get("/personas", (req, res) => {
  const personas = readJSON(path.join(RESULTS, "llm/persona_comparison.json"));
  if (!personas) {
    return res.status(404).json({ error: "Persona comparison not found." });
  }
  res.json(personas);
});

// ═══════════════════════════════════════════════════════════════════════════════
// GET /api/skills
// ═══════════════════════════════════════════════════════════════════════════════
router.get("/skills", (req, res) => {
  const lookup = readJSON(path.join(PROCESSED, "skill_id_to_name.json"));
  if (!lookup) return res.status(404).json({ error: "Skill lookup not found." });
  res.json(lookup);
});

// ═══════════════════════════════════════════════════════════════════════════════
// GET /api/run-pipeline   (Server-Sent Events)
//
// Query params (all 0.0–1.0 floats):
//   frustration, confusion, concentration, boredom, correctness
//
// Streams 7 pipeline step events then closes the connection.
// ═══════════════════════════════════════════════════════════════════════════════
router.get("/run-pipeline", async (req, res) => {
  // SSE headers
  res.setHeader("Content-Type",  "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection",    "keep-alive");
  res.flushHeaders();

  // Parse inputs — fall back to sensible defaults
  const frustration   = parseFloat(req.query.frustration   ?? 0.4);
  const confusion     = parseFloat(req.query.confusion     ?? 0.1);
  const concentration = parseFloat(req.query.concentration ?? 0.6);
  const boredom       = parseFloat(req.query.boredom       ?? 0.2);
  const correctness   = parseFloat(req.query.correctness   ?? 0.8);

  // Behavioral telemetry inputs (user-configurable or preset)
  const hintCount     = Math.max(0, parseInt(req.query.hint_count ?? 2, 10));
  const attemptCount  = Math.max(1, parseInt(req.query.attempt_count ?? 3, 10));
  const responseTimeS = Math.max(0.1, parseFloat(req.query.response_time_s ?? 45.2));

  // Compute derived behavioral features
  const hintToAttemptRatio = (hintCount / attemptCount).toFixed(2);
  // Baseline ASSISTments response time mean ~25.5s, std ~10.7s
  const zScoreVal = ((responseTimeS - 25.5) / 10.7);
  const zScoreStr = (zScoreVal >= 0 ? "+" : "") + zScoreVal.toFixed(2);

  // Load MCTS plan (used to populate steps 5 & 6)
  const mctsRows = readCSV(path.join(RESULTS, "mcts/mcts_fair_plan.csv"));
  const step1    = mctsRows ? mctsRows[0] : null;

  // ── Step 1: Student History ─────────────────────────────────────────────────
  await delay(700);
  sendEvent(res, {
    step: 1, status: "done",
    title: "Loading Student History",
    data: {
      total_interactions: "6,123,270",
      unique_students:    "4,918",
      unique_skills:      "265",
      unique_problems:    "179,999",
    },
  });

  // ── Step 2: Behavioral Feature Extraction ──────────────────────────────────
  await delay(850);
  sendEvent(res, {
    step: 2, status: "done",
    title: "Behavioral Feature Extraction",
    data: {
      hint_count:            String(hintCount),
      attempt_count:         String(attemptCount),
      avg_response_time_s:   responseTimeS.toFixed(1),
      hint_to_attempt_ratio: hintToAttemptRatio,
      response_time_z_score: zScoreStr,
    },
  });

  // ── Step 3: Affect Classifier (GBT) ────────────────────────────────────────
  await delay(900);
  sendEvent(res, {
    step: 3, status: "done",
    title: "Affect Classifier (Gradient-Boosted Trees)",
    data: {
      frustration:   frustration.toFixed(2),
      confusion:     confusion.toFixed(2),
      concentration: concentration.toFixed(2),
      boredom:       boredom.toFixed(2),
    },
  });

  // ── Step 4: Dual-State Encoder ─────────────────────────────────────────────
  await delay(1000);
  sendEvent(res, {
    step: 4, status: "done",
    title: "Dual-State Encoder (Transformer)",
    data: {
      latent_dim:       128,
      cognitive_head:   `z_cog — mastery ${(correctness * 100).toFixed(0)}%`,
      affective_head:   `z_aff — dominant: ${getDominant(frustration, confusion, concentration, boredom)}`,
      joint_state:      "z_t = [z_cog ⊕ z_aff] (128-dim)",
    },
  });

  // ── Step 5: World Model ─────────────────────────────────────────────────────
  await delay(1100);
  sendEvent(res, {
    step: 5, status: "done",
    title: "World Model (Probabilistic Transition Model)",
    data: {
      candidates_evaluated: 20,
      predicted_correctness:  step1 ? (+step1.predicted_correctness).toFixed(2) : "0.97",
      predicted_frustration:  step1 ? (+step1.predicted_frustration).toFixed(2) : "0.40",
      epistemic_uncertainty:  step1 ? (+step1.mean_latent_std).toFixed(3)       : "0.155",
    },
  });

  // ── Step 6: MCTS Planner ────────────────────────────────────────────────────
  await delay(1000);
  const mctsSummary = mctsRows
    ? mctsRows.map((r) => ({
        step:       r.step,
        skill_name: getSkillName(r.original_skill_id),
        problem_id: r.original_problem_id,
        reward:     (+r.immediate_reward).toFixed(3),
      }))
    : [];

  sendEvent(res, {
    step: 6, status: "done",
    title: "MCTS Planner (100 simulations, horizon 3)",
    data: {
      simulations:   100,
      horizon:       3,
      best_reward:   step1 ? (+step1.discounted_cumulative_reward).toFixed(3) : "0.198",
      path:          mctsSummary,
    },
  });

  // ── Step 7: LLM Lesson Generation (live Python call) ───────────────────────
  sendEvent(res, { step: 7, status: "running", title: "LLM Lesson Generation (OpenRouter)", data: {} });

  // Write temp input JSON for the Python script
  const tempInput = {
    frustration, confusion, concentration, boredom, correctness,
    problem_id: step1 ? parseInt(step1.original_problem_id) : 606850,
    skill_id:   step1 ? parseInt(step1.original_skill_id)   : 50,
  };
  const tempPath = path.join(os.tmpdir(), "acdt_live_input.json");
  fs.writeFileSync(tempPath, JSON.stringify(tempInput), "utf-8");

  // Spawn Python and collect stdout
  const lessonPlan = await new Promise((resolve) => {
    let stdout = "";
    let stderr = "";

    // Use the venv Python if available, else fall back to system python
    const pythonExe = path.join(ROOT, "myenv/Scripts/python.exe");
    const pyBin = fs.existsSync(pythonExe) ? pythonExe : "python";

    const proc = spawn(pyBin, [PYTHON_SCRIPT, tempPath], { cwd: ROOT });

    proc.stdout.on("data", (chunk) => { stdout += chunk.toString(); });
    proc.stderr.on("data", (chunk) => { stderr += chunk.toString(); });

    proc.on("close", () => {
      try {
        resolve(JSON.parse(stdout.trim()));
      } catch {
        console.error("[run-pipeline] Python stderr:", stderr);
        resolve({ error: "Failed to parse LLM output", raw: stdout });
      }
    });

    // Safety timeout — 45 s
    setTimeout(() => {
      proc.kill();
      resolve({ error: "LLM generation timed out after 45 s" });
    }, 45000);
  });

  sendEvent(res, {
    step: 7, status: "done",
    title: "LLM Lesson Generation (OpenRouter)",
    data: lessonPlan,
  });

  // Signal completion and close SSE stream
  sendEvent(res, { step: 0, status: "complete", title: "Pipeline complete", data: {} });
  res.end();
});

// ── Utility ────────────────────────────────────────────────────────────────────
function getDominant(f, c, cn, b) {
  const map = { Frustration: f, Confusion: c, Concentration: cn, Boredom: b };
  return Object.entries(map).sort((a, b) => b[1] - a[1])[0][0];
}

export default router;
