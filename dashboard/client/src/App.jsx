import { useState } from 'react'
import StudentDashboard from './pages/StudentDashboard'
import TeacherDashboard from './pages/TeacherDashboard'
import DemoPage from './pages/DemoPage'
import './index.css'

const TABS = [
  { id: 'demo',    label: '▶  Live Demo' },
  { id: 'student', label: 'Student View' },
  { id: 'teacher', label: 'Teacher View' },
];

export default function App() {
  const [view, setView] = useState('demo')

  return (
    <div className="min-h-screen bg-gray-950 text-white font-sans">
      {/* Top nav */}
      <nav className="bg-gray-900 border-b border-gray-800 px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center text-sm font-bold">
            AC
          </div>
          <span className="text-lg font-semibold tracking-tight">
            Affective-Cognitive Digital Twin
          </span>
        </div>

        <div className="flex gap-1 bg-gray-800 rounded-lg p-1">
          {TABS.map(tab => (
            <button
              key={tab.id}
              onClick={() => setView(tab.id)}
              className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all ${
                view === tab.id
                  ? 'bg-indigo-600 text-white shadow'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </nav>

      {/* Page content */}
      <main className="p-6">
        {view === 'demo'    && <DemoPage />}
        {view === 'student' && <StudentDashboard />}
        {view === 'teacher' && <TeacherDashboard />}
      </main>
    </div>
  )
}
