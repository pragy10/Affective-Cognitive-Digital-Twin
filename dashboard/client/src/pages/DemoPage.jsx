import { useState, useRef } from 'react'
import PersonaSelector, { PRESETS } from '../components/PersonaSelector'
import PipelineStep from '../components/PipelineStep'
import AffectRadar from '../components/AffectRadar'
import MasteryBar from '../components/MasteryBar'
import MCTSPathCard from '../components/MCTSPathCard'
import LessonPlanCard from '../components/LessonPlanCard'

const STEP_TITLES = [
  "",                                                    // 0 – unused
  "Loading Student History",                             // 1
  "Behavioral Feature Extraction",                       // 2
  "Affect Classifier (Gradient-Boosted Trees)",          // 3
  "Dual-State Encoder (Transformer)",                    // 4
  "World Model (Probabilistic Transition Model)",        // 5
  "MCTS Planner (100 simulations, horizon 3)",           // 6
  "LLM Lesson Generation (OpenRouter)",                  // 7
];

const DEFAULT_CUSTOM = {
  correctness: 0.75, frustration: 0.35, confusion: 0.20,
  concentration: 0.55, boredom: 0.15,
};

export default function DemoPage() {
  const [selectedPreset, setSelectedPreset]   = useState(PRESETS[0]);
  const [isCustomMode,   setIsCustomMode]      = useState(false);
  const [customState,    setCustomState]       = useState(DEFAULT_CUSTOM);

  // Pipeline state: map of stepNumber -> { status, data }
  const [steps,     setSteps]     = useState({});
  const [running,   setRunning]   = useState(false);
  const [completed, setCompleted] = useState(false);
  const [finalResult, setFinalResult] = useState(null);

  const esRef = useRef(null);

  // ── Active state (preset or custom) ────────────────────────────────────────
  const activeState = isCustomMode ? customState : selectedPreset;

  // ── Run Pipeline ────────────────────────────────────────────────────────────
  function handleRun() {
    if (running) return;

    // Reset
    setSteps({});
    setCompleted(false);
    setFinalResult(null);
    setRunning(true);

    // Initialise all 7 steps as "waiting"
    const initial = {};
    for (let i = 1; i <= 7; i++) {
      initial[i] = { status: "waiting", data: {} };
    }
    setSteps(initial);

    // Build query string from active state
    const { frustration, confusion, concentration, boredom, correctness } = activeState;
    const params = new URLSearchParams({
      frustration:   frustration.toFixed(2),
      confusion:     confusion.toFixed(2),
      concentration: concentration.toFixed(2),
      boredom:       boredom.toFixed(2),
      correctness:   correctness.toFixed(2),
    });

    // Open SSE connection
    const es = new EventSource(`/api/run-pipeline?${params}`);
    esRef.current = es;

    es.onmessage = (event) => {
      const payload = JSON.parse(event.data);

      if (payload.status === "complete") {
        es.close();
        setRunning(false);
        setCompleted(true);
        return;
      }

      setSteps((prev) => ({
        ...prev,
        [payload.step]: { status: payload.status, data: payload.data },
      }));

      // Capture final lesson plan for the dashboard reveal
      if (payload.step === 7 && payload.status === "done") {
        setFinalResult(payload.data);
      }
    };

    es.onerror = () => {
      es.close();
      setRunning(false);
    };
  }

  // ── Helper: build dashboard-compatible state from active persona ────────────
  const dashboardAffect = {
    frustration:   activeState.frustration,
    confusion:     activeState.confusion,
    concentration: activeState.concentration,
    boredom:       activeState.boredom,
  };

  // Build MCTS path from step 6 data when available
  const step6Data = steps[6]?.data;
  const mctPath   = step6Data?.path?.map((s, i) => ({
    step:                    i + 1,
    problem_id:              s.problem_id,
    skill_id:                null,
    skill_name:              s.skill_name,
    predicted_correctness:   activeState.correctness,
    predicted_frustration:   parseFloat(step6Data.predicted_frustration ?? activeState.frustration),
    predicted_confusion:     activeState.confusion,
    predicted_concentration: activeState.concentration,
    predicted_boredom:       activeState.boredom,
    immediate_reward:        parseFloat(s.reward),
  })) ?? null;

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      {/* ── Header ───────────────────────────────────────────────────────── */}
      <div>
        <h1 className="text-2xl font-bold">Live Pipeline Demo</h1>
        <p className="text-gray-400 text-sm mt-1">
          Select a student profile (cognitive + affective state), run the full ACDT pipeline
          live, and watch each stage produce real output.
        </p>
      </div>

      {/* ── Persona Selector ─────────────────────────────────────────────── */}
      {!running && !completed && (
        <section className="bg-gray-900 border border-gray-800 rounded-2xl p-6 space-y-5">
          <h2 className="font-semibold text-white">
            1 — Choose a Student State
            <span className="ml-2 text-xs font-normal text-gray-500">
              (Cognitive + Affective Dual-State)
            </span>
          </h2>
          <PersonaSelector
            selected={selectedPreset}
            custom={customState}
            onSelectPreset={(p) => { setSelectedPreset(p); setIsCustomMode(false); }}
            onCustomChange={(key, val) => setCustomState((prev) => ({ ...prev, [key]: val }))}
            onCustomMode={() => setIsCustomMode((v) => !v)}
            isCustomMode={isCustomMode}
          />
        </section>
      )}

      {/* ── Run button ───────────────────────────────────────────────────── */}
      {!running && !completed && (
        <div className="flex justify-center">
          <button
            onClick={handleRun}
            className="px-8 py-3 rounded-xl bg-indigo-600 hover:bg-indigo-500 active:scale-95 transition-all font-semibold text-white text-sm shadow-lg shadow-indigo-900/40"
          >
            Run Digital Twin Pipeline →
          </button>
        </div>
      )}

      {/* ── Pipeline steps ───────────────────────────────────────────────── */}
      {(running || completed) && (
        <section className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="font-semibold text-white">
              2 — Pipeline Execution
              <span className="ml-2 text-xs font-normal text-gray-500">
                ({isCustomMode ? "Custom State" : selectedPreset.name})
              </span>
            </h2>
            {completed && (
              <button
                onClick={() => { setCompleted(false); setRunning(false); setSteps({}); setFinalResult(null); }}
                className="text-xs text-gray-500 hover:text-gray-300 border border-gray-700 px-3 py-1 rounded-lg transition"
              >
                ← Run Again
              </button>
            )}
          </div>

          <div className="space-y-3">
            {[1, 2, 3, 4, 5, 6, 7].map((n) => (
              <PipelineStep
                key={n}
                stepNumber={n}
                title={STEP_TITLES[n]}
                status={steps[n]?.status ?? "waiting"}
                data={steps[n]?.data ?? {}}
              />
            ))}
          </div>
        </section>
      )}

      {/* ── Dashboard reveal ─────────────────────────────────────────────── */}
      {completed && (
        <section className="space-y-6 border-t border-gray-800 pt-8">
          <h2 className="font-semibold text-white text-lg">
            3 — Student Dashboard (Live Results)
          </h2>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <DashCard title="Current Affective State" subtitle="Detected by GBT Affect Classifier">
              <AffectRadar affect={dashboardAffect} />
            </DashCard>

            <DashCard title="Cognitive Mastery" subtitle="Predicted by Dual-State Encoder">
              <MasteryBar mastery={activeState.correctness} />
            </DashCard>
          </div>

          {mctPath && (
            <DashCard title="MCTS Recommended Learning Path" subtitle="Optimal 3-step plan from Monte Carlo Tree Search">
              <MCTSPathCard steps={mctPath} />
            </DashCard>
          )}

          {finalResult && !finalResult.error && (
            <DashCard title="Personalised Lesson Plan" subtitle="Generated live by LLM via OpenRouter">
              <LessonPlanCard plan={finalResult} />
            </DashCard>
          )}

          {finalResult?.error && (
            <div className="bg-red-900/20 border border-red-700 rounded-xl p-4 text-red-400 text-sm">
              LLM step error: {finalResult.error}
            </div>
          )}
        </section>
      )}
    </div>
  );
}

function DashCard({ title, subtitle, children }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5 space-y-3">
      <div>
        <h3 className="font-semibold text-white">{title}</h3>
        {subtitle && <p className="text-xs text-gray-500 mt-0.5">{subtitle}</p>}
      </div>
      {children}
    </div>
  );
}
