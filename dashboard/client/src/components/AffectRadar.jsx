import {
  Radar, RadarChart, PolarGrid, PolarAngleAxis, ResponsiveContainer, Tooltip
} from 'recharts'

const AFFECT_COLORS = {
  Frustration:   '#ef4444',
  Confusion:     '#f59e0b',
  Concentration: '#22c55e',
  Boredom:       '#6b7280',
}

export default function AffectRadar({ affect }) {
  const data = [
    { label: 'Frustration',   value: +(affect.frustration   * 100).toFixed(1) },
    { label: 'Confusion',     value: +(affect.confusion     * 100).toFixed(1) },
    { label: 'Concentration', value: +(affect.concentration * 100).toFixed(1) },
    { label: 'Boredom',       value: +(affect.boredom       * 100).toFixed(1) },
  ]

  // Dominant affect badge
  const dominant = data.reduce((a, b) => a.value > b.value ? a : b)

  return (
    <div className="space-y-3">
      {/* Dominant badge */}
      <div className="flex gap-2 flex-wrap">
        {data.map(d => (
          <span
            key={d.label}
            className={`text-xs font-medium px-2.5 py-1 rounded-full border ${
              d.label === dominant.label
                ? 'border-indigo-500 bg-indigo-900/40 text-indigo-300'
                : 'border-gray-700 bg-gray-800 text-gray-400'
            }`}
          >
            {d.label}: {d.value}%
          </span>
        ))}
      </div>

      {/* Radar chart */}
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart data={data} margin={{ top: 10, right: 20, bottom: 10, left: 20 }}>
            <PolarGrid stroke="#374151" />
            <PolarAngleAxis
              dataKey="label"
              tick={{ fill: '#9ca3af', fontSize: 12 }}
            />
            <Radar
              dataKey="value"
              stroke="#6366f1"
              fill="#6366f1"
              fillOpacity={0.25}
              strokeWidth={2}
            />
            <Tooltip
              contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 8 }}
              labelStyle={{ color: '#e5e7eb' }}
              formatter={v => [`${v}%`]}
            />
          </RadarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
