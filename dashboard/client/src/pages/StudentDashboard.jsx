import { useEffect, useState } from 'react'
import axios from 'axios'
import AffectRadar from '../components/AffectRadar'
import MasteryBar from '../components/MasteryBar'
import MCTSPathCard from '../components/MCTSPathCard'
import LessonPlanCard from '../components/LessonPlanCard'

export default function StudentDashboard() {
  const [data, setData]     = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError]   = useState(null)

  useEffect(() => {
    axios.get('/api/student-state')
      .then(res => { setData(res.data); setLoading(false) })
      .catch(err => { setError(err.message); setLoading(false) })
  }, [])

  if (loading) return (
    <div className="flex items-center justify-center h-64 text-gray-400">
      <div className="animate-spin w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full mr-3" />
      Loading student state...
    </div>
  )

  if (error) return (
    <div className="bg-red-900/30 border border-red-700 rounded-xl p-6 text-red-300">
      <p className="font-semibold">Could not load student data</p>
      <p className="text-sm mt-1">{error}</p>
      <p className="text-sm mt-2 text-gray-400">Make sure the Express server is running: <code className="bg-gray-800 px-1 rounded">npm run dev</code> inside <code className="bg-gray-800 px-1 rounded">dashboard/server</code></p>
    </div>
  )

  const { current_affect, current_mastery, recommended_path, lesson_plan } = data

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Header row */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Your Learning Dashboard</h1>
          <p className="text-gray-400 text-sm mt-0.5">
            Personalised recommendations powered by your Digital Twin
          </p>
        </div>
        <div className="bg-indigo-900/40 border border-indigo-700 rounded-xl px-4 py-2 text-center">
          <p className="text-xs text-indigo-300 uppercase tracking-widest">Predicted Mastery</p>
          <p className="text-3xl font-bold text-indigo-300 mt-0.5">
            {(current_mastery * 100).toFixed(0)}%
          </p>
        </div>
      </div>

      {/* Two-col layout */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left col */}
        <div className="space-y-6">
          <Card title="Current Affective State" subtitle="How you are feeling right now">
            <AffectRadar affect={current_affect} />
          </Card>

          <Card title="Cognitive Mastery" subtitle="Predicted correctness on next activity">
            <MasteryBar mastery={current_mastery} />
          </Card>
        </div>

        {/* Right col */}
        <div className="space-y-6">
          <Card title="Recommended Learning Path" subtitle="Next steps chosen by your Digital Twin">
            <MCTSPathCard steps={recommended_path} />
          </Card>
        </div>
      </div>

      {/* Lesson plan — full width */}
      {lesson_plan && (
        <Card title="Your Personalised Lesson Plan" subtitle="Prepared by your AI Tutor">
          <LessonPlanCard plan={lesson_plan} />
        </Card>
      )}
    </div>
  )
}

function Card({ title, subtitle, children }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5 space-y-3">
      <div>
        <h2 className="font-semibold text-white">{title}</h2>
        {subtitle && <p className="text-xs text-gray-500 mt-0.5">{subtitle}</p>}
      </div>
      {children}
    </div>
  )
}
