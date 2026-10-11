export default function MasteryBar({ mastery }) {
  const pct = (mastery * 100).toFixed(1)

  const color =
    mastery >= 0.8 ? 'bg-emerald-500' :
    mastery >= 0.6 ? 'bg-yellow-500'  :
                     'bg-red-500'

  const label =
    mastery >= 0.8 ? 'Strong Mastery'    :
    mastery >= 0.6 ? 'Developing Mastery' :
                     'Needs Support'

  return (
    <div className="space-y-3">
      <div className="flex justify-between items-end">
        <span className={`text-xs font-semibold px-2 py-0.5 rounded ${
          mastery >= 0.8 ? 'bg-emerald-900/40 text-emerald-400' :
          mastery >= 0.6 ? 'bg-yellow-900/40 text-yellow-400'   :
                           'bg-red-900/40 text-red-400'
        }`}>
          {label}
        </span>
        <span className="text-3xl font-bold text-white">{pct}%</span>
      </div>

      {/* Bar */}
      <div className="w-full bg-gray-800 rounded-full h-4 overflow-hidden">
        <div
          className={`h-4 rounded-full transition-all duration-700 ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>

      <p className="text-xs text-gray-500">
        Predicted probability of answering the next activity correctly based on your learning history.
      </p>
    </div>
  )
}
