import { useEffect, useState } from 'react'
import axios from 'axios'

const AFFECT_BAR_COLORS = {
  frustration:   { bar: 'bg-red-500',     text: 'text-red-400',     label: 'Frustration' },
  confusion:     { bar: 'bg-amber-500',   text: 'text-amber-400',   label: 'Confusion' },
  concentration: { bar: 'bg-emerald-500', text: 'text-emerald-400', label: 'Concentration' },
  boredom:       { bar: 'bg-gray-500',    text: 'text-gray-400',    label: 'Boredom' },
}

export default function TeacherDashboard() {
  const [personas, setPersonas] = useState(null)
  const [loading, setLoading]   = useState(true)
  const [error, setError]       = useState(null)

  useEffect(() => {
    axios.get('/api/personas')
      .then(res => { setPersonas(res.data); setLoading(false) })
      .catch(err => { setError(err.message); setLoading(false) })
  }, [])

  if (loading) return (
    <div className="flex items-center justify-center h-64 text-gray-400">
      <div className="animate-spin w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full mr-3" />
      Loading class data...
    </div>
  )

  if (error) return (
    <div className="bg-red-900/30 border border-red-700 rounded-xl p-6 text-red-300">
      <p className="font-semibold">Could not load persona data</p>
      <p className="text-sm mt-1">{error}</p>
    </div>
  )

  // Identify at-risk students (frustration > 0.6)
  const atRisk = personas.filter(p => p.persona.frustration > 0.6)

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold">Teacher Dashboard</h1>
        <p className="text-gray-400 text-sm mt-0.5">
          Class-level overview — affective states and personalised interventions
        </p>
      </div>

      {/* Early warning banner */}
      {atRisk.length > 0 && (
        <div className="bg-red-900/30 border border-red-700 rounded-xl p-4 flex items-start gap-3">
          <span className="text-red-400 text-xl mt-0.5">!</span>
          <div>
            <p className="font-semibold text-red-300">
              Early Warning: {atRisk.length} student{atRisk.length > 1 ? 's' : ''} at high frustration risk
            </p>
            <p className="text-sm text-red-400 mt-0.5">
              {atRisk.map(p => p.persona.name).join(', ')} — consider immediate intervention.
            </p>
          </div>
        </div>
      )}

      {/* Student cards grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {personas.map((item, idx) => {
          const p    = item.persona
          const plan = item.lesson_plan
          const isAtRisk = p.frustration > 0.6

          return (
            <div
              key={idx}
              className={`bg-gray-900 border rounded-2xl p-5 space-y-4 ${
                isAtRisk ? 'border-red-700' : 'border-gray-800'
              }`}
            >
              {/* Persona header */}
              <div className="flex items-start justify-between gap-2">
                <div>
                  <h3 className="font-semibold text-white">{p.name}</h3>
                  <p className="text-xs text-gray-500 mt-0.5">{p.description}</p>
                </div>
                {isAtRisk && (
                  <span className="shrink-0 text-xs font-semibold bg-red-900/50 text-red-400 border border-red-700 px-2 py-0.5 rounded-full">
                    At Risk
                  </span>
                )}
              </div>

              {/* Mastery row */}
              <div className="flex items-center gap-3">
                <span className="text-xs text-gray-400 w-20 shrink-0">Mastery</span>
                <div className="flex-1 bg-gray-800 rounded-full h-2.5">
                  <div
                    className={`h-2.5 rounded-full ${
                      p.correctness >= 0.8 ? 'bg-emerald-500' :
                      p.correctness >= 0.6 ? 'bg-yellow-500'  : 'bg-red-500'
                    }`}
                    style={{ width: `${(p.correctness * 100).toFixed(0)}%` }}
                  />
                </div>
                <span className="text-xs text-white w-10 text-right">
                  {(p.correctness * 100).toFixed(0)}%
                </span>
              </div>

              {/* Affect bars */}
              <div className="space-y-1.5">
                {Object.entries(AFFECT_BAR_COLORS).map(([key, cfg]) => (
                  <div key={key} className="flex items-center gap-3">
                    <span className={`text-xs w-24 shrink-0 ${cfg.text}`}>{cfg.label}</span>
                    <div className="flex-1 bg-gray-800 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full ${cfg.bar}`}
                        style={{ width: `${(p[key] * 100).toFixed(0)}%` }}
                      />
                    </div>
                    <span className="text-xs text-gray-400 w-10 text-right">
                      {(p[key] * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
              </div>

              {/* Divider */}
              <div className="border-t border-gray-800" />

              {/* AI recommended strategy */}
              {plan && (
                <div className="space-y-1">
                  <p className="text-xs text-gray-500 uppercase tracking-widest font-semibold">
                    Recommended Intervention
                  </p>
                  <p className="text-sm text-indigo-300 leading-relaxed">
                    {plan.pedagogical_strategy}
                  </p>
                  <p className="text-xs text-gray-600 italic mt-1">{plan.teacher_note}</p>
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
