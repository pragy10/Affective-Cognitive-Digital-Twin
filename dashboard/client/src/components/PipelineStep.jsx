// Animated pipeline step card
// Props:
//   stepNumber  – 1-7
//   title       – step label
//   status      – "waiting" | "running" | "done" | "error"
//   data        – object of key/value output to display when done

const STATUS_ICON = {
  waiting: (
    <span className="w-7 h-7 rounded-full bg-gray-800 border border-gray-600 flex items-center justify-center text-gray-500 text-xs font-bold shrink-0">
      –
    </span>
  ),
  running: (
    <span className="w-7 h-7 rounded-full bg-indigo-900 border border-indigo-500 flex items-center justify-center shrink-0">
      <span className="w-3 h-3 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin" />
    </span>
  ),
  done: (
    <span className="w-7 h-7 rounded-full bg-emerald-900 border border-emerald-500 flex items-center justify-center text-emerald-400 text-sm shrink-0">
      ✓
    </span>
  ),
  error: (
    <span className="w-7 h-7 rounded-full bg-red-900 border border-red-500 flex items-center justify-center text-red-400 text-sm shrink-0">
      ✗
    </span>
  ),
};

export default function PipelineStep({ stepNumber, title, status, data }) {
  const isDone    = status === "done";
  const isRunning = status === "running";
  const isWaiting = status === "waiting";

  return (
    <div className={`flex gap-4 p-4 rounded-xl border transition-all duration-300 ${
      isDone    ? "border-emerald-800 bg-emerald-950/20" :
      isRunning ? "border-indigo-700 bg-indigo-950/30 animate-pulse" :
                  "border-gray-800 bg-gray-900/40 opacity-50"
    }`}>
      {/* Status icon */}
      {STATUS_ICON[status] ?? STATUS_ICON.waiting}

      {/* Content */}
      <div className="flex-1 min-w-0 space-y-2">
        {/* Header */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500 font-mono shrink-0">Step {stepNumber}</span>
          <p className={`text-sm font-semibold ${
            isDone ? "text-emerald-300" : isRunning ? "text-indigo-300" : "text-gray-500"
          }`}>
            {title}
          </p>
          {isRunning && (
            <span className="text-xs text-indigo-400 animate-pulse">Running...</span>
          )}
        </div>

        {/* Output data — only shown when done */}
        {isDone && data && Object.keys(data).length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-1 pt-1">
            {renderData(data)}
          </div>
        )}
      </div>
    </div>
  );
}

function renderData(data) {
  // Special case: MCTS path array
  if (data.path && Array.isArray(data.path)) {
    return (
      <>
        {Object.entries(data)
          .filter(([k]) => k !== "path")
          .map(([k, v]) => (
            <DataRow key={k} label={formatLabel(k)} value={String(v)} />
          ))}
        <div className="col-span-2 mt-1 space-y-1">
          <p className="text-xs text-gray-500 font-semibold uppercase tracking-wider">Recommended Path</p>
          {data.path.map((s, i) => (
            <div key={i} className="flex items-center gap-2 text-xs">
              <span className="w-5 h-5 rounded-full bg-indigo-900 text-indigo-300 flex items-center justify-center font-bold text-xs shrink-0">
                {s.step}
              </span>
              <span className="text-emerald-300 font-medium">{s.skill_name}</span>
              <span className="text-gray-600">·</span>
              <span className="text-gray-500">Problem #{s.problem_id}</span>
              <span className="text-gray-600">·</span>
              <span className="text-indigo-400">Reward {s.reward}</span>
            </div>
          ))}
        </div>
      </>
    );
  }

  // Special case: LLM lesson plan
  if (data.topic || data.student_message) {
    return (
      <div className="col-span-2 space-y-2">
        {data.topic && (
          <p className="text-xs">
            <span className="text-gray-500">Topic: </span>
            <span className="text-indigo-300 font-semibold">{data.topic}</span>
          </p>
        )}
        {data.student_message && (
          <p className="text-xs text-gray-300 italic leading-relaxed border-l-2 border-indigo-700 pl-2">
            "{data.student_message}"
          </p>
        )}
        {data.pedagogical_strategy && (
          <p className="text-xs text-gray-500">
            <span className="text-amber-400 font-medium">Strategy: </span>
            {data.pedagogical_strategy}
          </p>
        )}
        {data.error && (
          <p className="text-xs text-red-400">{data.error}</p>
        )}
      </div>
    );
  }

  // Default: key-value pairs
  return Object.entries(data).map(([k, v]) => (
    <DataRow key={k} label={formatLabel(k)} value={String(v)} />
  ));
}

function DataRow({ label, value }) {
  return (
    <div className="flex items-baseline gap-1.5 text-xs">
      <span className="text-gray-500 shrink-0">{label}:</span>
      <span className="text-emerald-300 font-mono truncate">{value}</span>
    </div>
  );
}

function formatLabel(key) {
  return key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}
