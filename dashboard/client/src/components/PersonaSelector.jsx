const DIMENSION_META = {
  correctness:   { label: "Cognitive Mastery",  color: "indigo",   description: "Predicted probability of answering correctly" },
  frustration:   { label: "Frustration",        color: "red",      description: "Emotional stress / blocked progress" },
  confusion:     { label: "Confusion",          color: "amber",    description: "Conceptual uncertainty / lost understanding" },
  concentration: { label: "Concentration",      color: "emerald",  description: "Focused engagement / flow state" },
  boredom:       { label: "Boredom",            color: "gray",     description: "Under-stimulation / disengagement" },
};

const COLOR_CLASSES = {
  indigo:  { bar: "bg-indigo-500",  text: "text-indigo-400",  badge: "bg-indigo-900/40 border-indigo-700 text-indigo-300" },
  red:     { bar: "bg-red-500",     text: "text-red-400",     badge: "bg-red-900/40    border-red-700    text-red-300" },
  amber:   { bar: "bg-amber-500",   text: "text-amber-400",   badge: "bg-amber-900/40  border-amber-700  text-amber-300" },
  emerald: { bar: "bg-emerald-500", text: "text-emerald-400", badge: "bg-emerald-900/40 border-emerald-700 text-emerald-300" },
  gray:    { bar: "bg-gray-500",    text: "text-gray-400",    badge: "bg-gray-800      border-gray-600   text-gray-300" },
};

export const PRESETS = [
  {
    id: "frustrated",
    icon: "😤",
    name: "Frustrated Learner",
    description: "Struggling cognitively and emotionally dysregulated. Needs de-escalation and heavy scaffolding.",
    correctness: 0.48, frustration: 0.87, confusion: 0.61, concentration: 0.12, boredom: 0.05,
    hintCount: 5, attemptCount: 4, responseTimeS: 78.5,
  },
  {
    id: "confused",
    icon: "🤔",
    name: "Confused Learner",
    description: "Wants to understand but stuck conceptually. Needs clear step-by-step guidance.",
    correctness: 0.52, frustration: 0.38, confusion: 0.79, concentration: 0.25, boredom: 0.08,
    hintCount: 3, attemptCount: 2, responseTimeS: 62.0,
  },
  {
    id: "bored",
    icon: "😴",
    name: "Bored Learner",
    description: "High mastery, under-challenged and disengaged. Needs novelty and a harder challenge.",
    correctness: 0.96, frustration: 0.07, confusion: 0.03, concentration: 0.22, boredom: 0.88,
    hintCount: 0, attemptCount: 1, responseTimeS: 12.3,
  },
  {
    id: "engaged",
    icon: "🎯",
    name: "Engaged Learner",
    description: "In a flow state — high mastery and high concentration. Keep momentum going.",
    correctness: 0.97, frustration: 0.40, confusion: 0.04, concentration: 0.65, boredom: 0.19,
    hintCount: 1, attemptCount: 1, responseTimeS: 24.0,
  },
];

export default function PersonaSelector({ selected, custom, onSelectPreset, onCustomChange, onCustomMode, isCustomMode }) {
  return (
    <div className="space-y-6">
      {/* Preset cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {PRESETS.map((p) => (
          <button
            key={p.id}
            onClick={() => onSelectPreset(p)}
            className={`text-left p-4 rounded-2xl border transition-all hover:border-indigo-600 ${
              selected?.id === p.id && !isCustomMode
                ? "border-indigo-500 bg-indigo-900/20 ring-1 ring-indigo-500"
                : "border-gray-700 bg-gray-900"
            }`}
          >
            <div className="text-2xl mb-2">{p.icon}</div>
            <p className="font-semibold text-white text-sm">{p.name}</p>
            <p className="text-xs text-gray-500 mt-1 leading-relaxed">{p.description}</p>

            {/* Mini dimension bars */}
            <div className="mt-3 space-y-1.5">
              {Object.entries(DIMENSION_META).map(([key, meta]) => {
                const val = p[key];
                const c   = COLOR_CLASSES[meta.color];
                return (
                  <div key={key} className="flex items-center gap-2">
                    <span className={`text-xs w-20 shrink-0 ${c.text}`}>{meta.label}</span>
                    <div className="flex-1 bg-gray-800 rounded-full h-1.5">
                      <div className={`h-1.5 rounded-full ${c.bar}`} style={{ width: `${val * 100}%` }} />
                    </div>
                    <span className="text-xs text-gray-500 w-8 text-right">{(val * 100).toFixed(0)}%</span>
                  </div>
                );
              })}
            </div>
          </button>
        ))}
      </div>

      {/* Custom toggle */}
      <div className="flex items-center gap-3">
        <div className="flex-1 h-px bg-gray-800" />
        <button
          onClick={onCustomMode}
          className={`text-xs px-3 py-1 rounded-full border transition-all ${
            isCustomMode
              ? "border-indigo-600 bg-indigo-900/30 text-indigo-400"
              : "border-gray-700 text-gray-500 hover:text-gray-300"
          }`}
        >
          ✏️ Custom State
        </button>
        <div className="flex-1 h-px bg-gray-800" />
      </div>

      {/* Custom sliders & telemetry — shown only in custom mode */}
      {isCustomMode && (
        <div className="bg-gray-900 border border-gray-700 rounded-2xl p-5 space-y-6">
          {/* Dual-State Latent Sliders */}
          <div className="space-y-4">
            <p className="text-sm font-semibold text-white">1. Define Student Dual-State (Cognitive & Affective):</p>
            {Object.entries(DIMENSION_META).map(([key, meta]) => {
              const c = COLOR_CLASSES[meta.color];
              return (
                <div key={key} className="space-y-1">
                  <div className="flex justify-between items-center">
                    <div>
                      <span className={`text-sm font-medium ${c.text}`}>{meta.label}</span>
                      <span className="text-xs text-gray-600 ml-2">{meta.description}</span>
                    </div>
                    <span className={`text-sm font-bold px-2 py-0.5 rounded border ${c.badge}`}>
                      {(custom[key] * 100).toFixed(0)}%
                    </span>
                  </div>
                  <input
                    type="range"
                    min={0} max={1} step={0.01}
                    value={custom[key]}
                    onChange={(e) => onCustomChange(key, parseFloat(e.target.value))}
                    className="w-full accent-indigo-500"
                  />
                </div>
              );
            })}
          </div>

          {/* Behavioral Telemetry Inputs */}
          <div className="pt-4 border-t border-gray-800 space-y-3">
            <div>
              <p className="text-sm font-semibold text-white">2. Behavioral Telemetry (Step 2 Extraction):</p>
              <p className="text-xs text-gray-500">Configure raw telemetry signals to test dynamic feature extraction.</p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              {/* Hint Count */}
              <div className="bg-gray-800/60 p-3 rounded-xl border border-gray-700/60 space-y-1">
                <label className="text-xs font-medium text-gray-400 block">Hint Count</label>
                <input
                  type="number"
                  min="0"
                  max="20"
                  value={custom.hintCount ?? 2}
                  onChange={(e) => onCustomChange("hintCount", Math.max(0, parseInt(e.target.value) || 0))}
                  className="w-full bg-gray-900 border border-gray-700 rounded-lg px-2.5 py-1.5 text-sm text-white font-mono focus:border-indigo-500 focus:outline-none"
                />
                <span className="text-[10px] text-gray-500 block">Hints requested</span>
              </div>

              {/* Attempt Count */}
              <div className="bg-gray-800/60 p-3 rounded-xl border border-gray-700/60 space-y-1">
                <label className="text-xs font-medium text-gray-400 block">Attempt Count</label>
                <input
                  type="number"
                  min="1"
                  max="20"
                  value={custom.attemptCount ?? 3}
                  onChange={(e) => onCustomChange("attemptCount", Math.max(1, parseInt(e.target.value) || 1))}
                  className="w-full bg-gray-900 border border-gray-700 rounded-lg px-2.5 py-1.5 text-sm text-white font-mono focus:border-indigo-500 focus:outline-none"
                />
                <span className="text-[10px] text-gray-500 block">Total attempts</span>
              </div>

              {/* Response Time (seconds) */}
              <div className="bg-gray-800/60 p-3 rounded-xl border border-gray-700/60 space-y-1">
                <label className="text-xs font-medium text-gray-400 block">Time Taken (s)</label>
                <input
                  type="number"
                  min="1"
                  max="300"
                  step="0.5"
                  value={custom.responseTimeS ?? 45.2}
                  onChange={(e) => onCustomChange("responseTimeS", Math.max(0.5, parseFloat(e.target.value) || 1))}
                  className="w-full bg-gray-900 border border-gray-700 rounded-lg px-2.5 py-1.5 text-sm text-white font-mono focus:border-indigo-500 focus:outline-none"
                />
                <span className="text-[10px] text-gray-500 block">Response latency (s)</span>
              </div>
            </div>

            {/* Live Preview of computed metrics */}
            <div className="flex gap-4 text-xs text-gray-400 bg-gray-950/40 p-2.5 rounded-lg border border-gray-800/80">
              <div>
                <span>Hint/Attempt Ratio: </span>
                <span className="font-mono text-indigo-400">
                  {((custom.hintCount ?? 2) / (custom.attemptCount || 1)).toFixed(2)}
                </span>
              </div>
              <span className="text-gray-700">|</span>
              <div>
                <span>Response Time Z-Score: </span>
                <span className="font-mono text-emerald-400">
                  {(((custom.responseTimeS ?? 45.2) - 25.5) / 10.7 >= 0 ? "+" : "") +
                   (((custom.responseTimeS ?? 45.2) - 25.5) / 10.7).toFixed(2)}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
