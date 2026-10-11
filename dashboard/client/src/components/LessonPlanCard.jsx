export default function LessonPlanCard({ plan }) {
  if (!plan) return null

  return (
    <div className="space-y-5">
      {/* Topic pill */}
      <div className="flex items-center gap-2">
        <span className="text-xs uppercase tracking-widest text-indigo-400 font-semibold">
          Topic
        </span>
        <span className="bg-indigo-900/40 border border-indigo-700 text-indigo-300 text-sm font-semibold px-3 py-1 rounded-full">
          {plan.topic}
        </span>
      </div>

      {/* Tutor message */}
      <div className="bg-indigo-950/60 border border-indigo-800 rounded-xl p-4">
        <p className="text-xs text-indigo-400 font-semibold uppercase tracking-widest mb-2">
          Tutor's Message
        </p>
        <p className="text-white leading-relaxed">{plan.student_message}</p>
      </div>

      {/* Two-col: strategy + micro-lesson */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <InfoBlock
          label="Pedagogical Strategy"
          text={plan.pedagogical_strategy}
          accent="amber"
        />
        <InfoBlock
          label="Conceptual Refresher"
          text={plan.micro_lesson}
          accent="sky"
        />
      </div>

      {/* Hints */}
      {plan.hints && plan.hints.length > 0 && (
        <div className="bg-gray-800/60 border border-gray-700 rounded-xl p-4 space-y-2">
          <p className="text-xs text-gray-400 font-semibold uppercase tracking-widest mb-1">
            Scaffolded Hints (unlock if stuck)
          </p>
          {plan.hints.map((hint, i) => (
            <details key={i} className="group">
              <summary className="cursor-pointer text-sm text-gray-300 hover:text-white py-1 list-none flex items-center gap-2">
                <span className="w-5 h-5 rounded-full bg-gray-700 text-gray-400 group-open:bg-indigo-700 group-open:text-white flex items-center justify-center text-xs font-bold shrink-0 transition-all">
                  {i + 1}
                </span>
                Tier {i + 1} Hint
              </summary>
              <p className="text-sm text-gray-300 mt-2 ml-7 leading-relaxed">{hint}</p>
            </details>
          ))}
        </div>
      )}

      {/* Teacher note — subtle */}
      {plan.teacher_note && (
        <div className="border-t border-gray-800 pt-4">
          <p className="text-xs text-gray-600 font-semibold uppercase tracking-widest mb-1">
            For the Teacher
          </p>
          <p className="text-xs text-gray-500 leading-relaxed italic">{plan.teacher_note}</p>
        </div>
      )}

      {plan._meta && (
        <p className="text-xs text-gray-700">
          Model: {plan._meta.model_used} &nbsp;|&nbsp; Problem #{plan._meta.problem_id}
        </p>
      )}
    </div>
  )
}

function InfoBlock({ label, text, accent }) {
  const styles = {
    amber: 'border-amber-800 bg-amber-950/30 text-amber-400',
    sky:   'border-sky-800   bg-sky-950/30   text-sky-400',
  }
  return (
    <div className={`border rounded-xl p-4 ${styles[accent]}`}>
      <p className="text-xs font-semibold uppercase tracking-widest mb-2">{label}</p>
      <p className="text-sm text-gray-300 leading-relaxed">{text}</p>
    </div>
  )
}
