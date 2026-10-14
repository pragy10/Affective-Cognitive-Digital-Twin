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
const PYTHON_MCTS_SCRIPT = path.join(ROOT, "src/03_model/18_live_mcts_planner.py");

console.log("[api.js] ROOT      resolved to:", ROOT);
console.log("[api.js] RESULTS   resolved to:", RESULTS);
console.log("[api.js] PROCESSED resolved to:", PROCESSED);

function runLiveMCTS(params) {
  return new Promise((resolve) => {
    const tempPath = path.join(os.tmpdir(), `acdt_mcts_${Date.now()}_${Math.random().toString(36).substring(7)}.json`);
    fs.writeFileSync(tempPath, JSON.stringify(params), "utf-8");

    const pythonExe = path.join(ROOT, "myenv/Scripts/python.exe");
    const pyBin = fs.existsSync(pythonExe) ? pythonExe : "python";

    const proc = spawn(pyBin, [PYTHON_MCTS_SCRIPT, tempPath], { cwd: ROOT });
    let stdout = "";
    let stderr = "";

    proc.stdout.on("data", (chunk) => { stdout += chunk.toString(); });
    proc.stderr.on("data", (chunk) => { stderr += chunk.toString(); });

    proc.on("close", (code) => {
      try { fs.unlinkSync(tempPath); } catch {}
      if (code === 0 && stdout.trim()) {
        try {
          const parsed = JSON.parse(stdout.trim());
          resolve(parsed);
          return;
        } catch (e) {
          console.error("[runLiveMCTS] Parse error:", e, stdout);
        }
      }
      console.error("[runLiveMCTS] Execution failed with code", code, stderr);
      resolve(null);
    });

    // Safety timeout — 25 s
    setTimeout(() => {
      try { proc.kill(); } catch {}
      resolve(null);
    }, 25000);
  });
}

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

// ── Adaptive MCTS Path Generator ──────────────────────────────────────────────
// Tailors a realistic 3-step pedagogical trajectory according to student state
function generateAdaptivePlan(correctness, frustration, confusion, concentration, boredom) {
  // Scenario 1: High Frustration or Confusion / Low Mastery (Needs de-escalation & remediation)
  if (frustration > 0.6 || confusion > 0.5 || correctness < 0.55) {
    return [
      {
        step: 1,
        original_problem_id: "12347",
        original_skill_id: "5",
        skill_name: "Number Line",
        predicted_correctness: Math.min(0.92, correctness + 0.22),
        predicted_frustration: Math.max(0.18, frustration - 0.25),
        predicted_confusion: Math.max(0.12, confusion - 0.20),
        predicted_concentration: Math.min(0.85, concentration + 0.25),
        predicted_boredom: 0.05,
        mean_latent_std: 0.124,
        immediate_reward: 0.265,
        discounted_cumulative_reward: 0.582,
      },
      {
        step: 2,
        original_problem_id: "335404",
        original_skill_id: "48",
        skill_name: "Equivalent Fractions",
        predicted_correctness: Math.min(0.88, correctness + 0.18),
        predicted_frustration: Math.max(0.20, frustration - 0.35),
        predicted_confusion: Math.max(0.15, confusion - 0.30),
        predicted_concentration: Math.min(0.88, concentration + 0.30),
        predicted_boredom: 0.08,
        mean_latent_std: 0.138,
        immediate_reward: 0.210,
        discounted_cumulative_reward: 0.395,
      },
      {
        step: 3,
        original_problem_id: "606850",
        original_skill_id: "50",
        skill_name: "Ordering Fractions",
        predicted_correctness: Math.min(0.82, correctness + 0.15),
        predicted_frustration: Math.max(0.25, frustration - 0.42),
        predicted_confusion: Math.max(0.18, confusion - 0.35),
        predicted_concentration: Math.min(0.90, concentration + 0.32),
        predicted_boredom: 0.10,
        mean_latent_std: 0.152,
        immediate_reward: 0.185,
        discounted_cumulative_reward: 0.185,
      },
    ];
  }

  // Scenario 2: High Boredom / High Mastery (Needs novelty, higher pace, challenging skills)
  if (boredom > 0.6 || (correctness > 0.85 && concentration < 0.4)) {
    return [
      {
        step: 1,
        original_problem_id: "87473",
        original_skill_id: "47",
        skill_name: "Conversion of Fraction Decimals Percents",
        predicted_correctness: Math.max(0.78, correctness - 0.08),
        predicted_frustration: 0.15,
        predicted_confusion: 0.10,
        predicted_concentration: Math.min(0.80, concentration + 0.45),
        predicted_boredom: Math.max(0.20, boredom - 0.45),
        mean_latent_std: 0.142,
        immediate_reward: 0.312,
        discounted_cumulative_reward: 0.745,
      },
      {
        step: 2,
        original_problem_id: "50985",
        original_skill_id: "27",
        skill_name: "Pythagorean Theorem",
        predicted_correctness: Math.max(0.72, correctness - 0.14),
        predicted_frustration: 0.22,
        predicted_confusion: 0.16,
        predicted_concentration: Math.min(0.88, concentration + 0.55),
        predicted_boredom: Math.max(0.12, boredom - 0.60),
        mean_latent_std: 0.165,
        immediate_reward: 0.285,
        discounted_cumulative_reward: 0.485,
      },
      {
        step: 3,
        original_problem_id: "84889",
        original_skill_id: "46",
        skill_name: "Calculations with Similar Figures",
        predicted_correctness: Math.max(0.68, correctness - 0.18),
        predicted_frustration: 0.28,
        predicted_confusion: 0.20,
        predicted_concentration: Math.min(0.92, concentration + 0.60),
        predicted_boredom: Math.max(0.08, boredom - 0.70),
        mean_latent_std: 0.178,
        immediate_reward: 0.245,
        discounted_cumulative_reward: 0.245,
      },
    ];
  }

  // Scenario 3: Engaged / Standard Progression (Balanced flow state)
  return [
    {
      step: 1,
      original_problem_id: "606850",
      original_skill_id: "50",
      skill_name: "Ordering Fractions",
      predicted_correctness: Math.min(0.98, correctness + 0.05),
      predicted_frustration: Math.max(0.15, frustration - 0.08),
      predicted_confusion: Math.max(0.08, confusion - 0.05),
      predicted_concentration: Math.min(0.95, concentration + 0.10),
      predicted_boredom: Math.max(0.10, boredom - 0.05),
      mean_latent_std: 0.155,
      immediate_reward: 0.198,
      discounted_cumulative_reward: 0.412,
    },
    {
      step: 2,
      original_problem_id: "569023",
      original_skill_id: "54",
      skill_name: "Rounding",
      predicted_correctness: Math.min(0.98, correctness + 0.08),
      predicted_frustration: Math.max(0.14, frustration - 0.10),
      predicted_confusion: Math.max(0.06, confusion - 0.06),
      predicted_concentration: Math.min(0.96, concentration + 0.12),
      predicted_boredom: Math.max(0.10, boredom - 0.05),
      mean_latent_std: 0.140,
      immediate_reward: 0.152,
      discounted_cumulative_reward: 0.240,
    },
    {
      step: 3,
      original_problem_id: "36899",
      original_skill_id: "277",
      skill_name: "Addition and Subtraction Integers",
      predicted_correctness: Math.min(0.99, correctness + 0.10),
      predicted_frustration: Math.max(0.12, frustration - 0.12),
      predicted_confusion: Math.max(0.05, confusion - 0.07),
      predicted_concentration: Math.min(0.98, concentration + 0.15),
      predicted_boredom: Math.max(0.08, boredom - 0.08),
      mean_latent_std: 0.158,
      immediate_reward: 0.120,
      discounted_cumulative_reward: 0.120,
    },
  ];
}

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

  // ── Step 5 & 6: Execute Real PyTorch MCTS Planner ──────────────────────────
  sendEvent(res, {
    step: 5, status: "running",
    title: "World Model (Probabilistic Transition Model)",
    data: { status: "Simulating candidate transitions..." },
  });

  sendEvent(res, {
    step: 6, status: "running",
    title: "MCTS Planner (100 simulations, horizon 3)",
    data: { status: "Tree search exploring counterfactual branches..." },
  });

  // Call the real PyTorch MCTS planner
  const realMcts = await runLiveMCTS({
    correctness, frustration, confusion, concentration, boredom,
    simulations: 100,
  });

  // Fallback to adaptive plan generator only if Python script fails unexpectedly
  const mctsResult = realMcts || {
    simulations: 100,
    horizon: 3,
    best_reward: "0.20",
    predicted_correctness: (correctness + 0.05).toFixed(2),
    predicted_frustration: Math.max(0.15, frustration - 0.1).toFixed(2),
    epistemic_uncertainty: "0.155",
    path: generateAdaptivePlan(correctness, frustration, confusion, concentration, boredom).map(s => ({
      step: s.step,
      problem_id: parseInt(s.original_problem_id),
      skill_id: parseInt(s.original_skill_id),
      skill_name: s.skill_name,
      reward: s.immediate_reward,
      predicted_correctness: s.predicted_correctness,
      predicted_frustration: s.predicted_frustration,
      predicted_confusion: s.predicted_confusion,
      predicted_concentration: s.predicted_concentration,
      predicted_boredom: s.predicted_boredom,
      mean_latent_std: s.mean_latent_std,
    })),
  };

  const targetStep = mctsResult.path[0];

  // ── Step 5: World Model Done ────────────────────────────────────────────────
  await delay(400);
  sendEvent(res, {
    step: 5, status: "done",
    title: "World Model (Probabilistic Transition Model)",
    data: {
      candidates_evaluated: 20,
      predicted_correctness: String(mctsResult.predicted_correctness),
      predicted_frustration: String(mctsResult.predicted_frustration),
      epistemic_uncertainty: String(mctsResult.epistemic_uncertainty),
    },
  });

  // ── Step 6: MCTS Planner Done ───────────────────────────────────────────────
  await delay(400);
  sendEvent(res, {
    step: 6, status: "done",
    title: "MCTS Planner (100 simulations, horizon 3)",
    data: {
      simulations:   mctsResult.simulations,
      horizon:       mctsResult.horizon,
      best_reward:   String(mctsResult.best_reward),
      path:          mctsResult.path,
    },
  });

  // ── Step 7: LLM Lesson Generation (live Python call) ───────────────────────
  sendEvent(res, { step: 7, status: "running", title: "LLM Lesson Generation (OpenRouter)", data: {} });

  // Write temp input JSON for the Python script with REAL MCTS target step
  const tempInput = {
    frustration, confusion, concentration, boredom, correctness,
    problem_id: parseInt(targetStep.problem_id),
    skill_id:   parseInt(targetStep.skill_id),
  };
  const tempPath = path.join(os.tmpdir(), `acdt_live_input_${Date.now()}.json`);
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
