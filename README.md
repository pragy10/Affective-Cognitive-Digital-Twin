# Affective-Cognitive Digital Twin (ACDT) for Predictive Personalized Learning

ACDT is an end-to-end framework that models student learning by jointly tracking **cognitive mastery** and **affective states** (Frustration, Confusion, Concentration, Boredom). Rather than relying on reactive next-step prediction or risky online reinforcement learning with live students, ACDT constructs an offline **Digital Twin** using an action-conditioned Probabilistic World Model and Monte Carlo Tree Search (MCTS), generating personalized, empathetic lesson plans through a generative LLM layer and delivering the experience through an interactive MERN dashboard.

---

## 🌟 Key Features

1. **Sensor-Free Affect Detection**: Inferred directly from interaction telemetry (response latencies, hint requests, attempt frequencies).
2. **Dual-State Student Encoder**: Multi-head sequence model (Transformer + LSTM) producing a 128-dimensional joint latent representation $\mathbf{z}_t = [\mathbf{z}_t^{cog} \oplus \mathbf{z}_t^{aff}]$.
3. **Probabilistic Action-Conditioned World Model**: Predicts future learner transitions alongside epistemic uncertainty ($\sigma_{epistemic}$).
4. **Counterfactual MCTS Planner**: Optimizes 3-step pedagogical pathways using Upper Confidence Bound (UCB1) search and multi-objective rewards.
5. **Generative LLM Tutor**: Structured pedagogical translation via OpenRouter with automatic multi-model failover (Gemini $\rightarrow$ LLaMA $\rightarrow$ Claude $\rightarrow$ Mistral).
6. **Real-Time MERN Dashboard**: Interactive UI with Tailwind CSS v3, Recharts (Affect Radar, Mastery Bar), and live Server-Sent Events (SSE) streaming.

---

## 🏗️ System Architecture

```text
Student Interaction Telemetry (ASSISTments 2012-2013)
                     │
                     ▼
       [Phase 1 & 2: Feature Engineering]
                     │
                     ▼
       [Phase 3: Dual-State Student Encoder]
         ├── Cognitive Mastery Head (AUC: 0.76)
         └── Affective Head (Concentration, Frustration, Confusion, Boredom)
                     │
                     ▼ 128-dim Joint Latent State (z_t)
       [Phase 4: Probabilistic World Model]
         └── Evaluates Action-Conditioned Transitions + Uncertainty
                     │
                     ▼
       [Phase 5: MCTS Counterfactual Planner]
         └── UCB1 Tree Search over 53,091 candidate problems
                     │
                     ▼ Optimal 3-Step Sequence
       [Phase 6: Generative LLM Translation (OpenRouter)]
         └── Empathetic Micro-Lessons & Tiered Hints
                     │
                     ▼
       [Phase 7: Full-Stack MERN Dashboard]
         ├── Live SSE Pipeline Stepper
         ├── Student View (Affect Radar, Tiered Hints)
         └── Teacher View (At-Risk Early Warning Flags)
```

---

## 📦 Project Structure

```text
ACDT/
├── data/
│   ├── raw/                              # Raw ASSISTments 2012-2013 CSV
│   └── processed/                        # Cleaned CSVs, NPZ tensors, skill mappings
├── models/                               # Saved model weights (.pt)
├── results/                              # Evaluation metrics, charts, MCTS plans, LLM JSONs
├── src/
│   ├── 01_data_investigation/            # Exploratory data analysis scripts
│   ├── 02_data_pipeline/                 # Preprocessing & sequence construction
│   ├── 03_model/                         # Encoder, World Model, and MCTS scripts (01-17)
│   └── 04_llm/                           # OpenRouter API integration & persona evaluations
├── dashboard/
│   ├── server/                           # Express.js API backend (port 5000)
│   │   ├── index.js
│   │   └── routes/api.js                 # SSE streaming & telemetry endpoints
│   └── client/                           # React + Vite + Tailwind CSS v3 (port 5173)
│       └── src/
│           ├── pages/                    # DemoPage, StudentDashboard, TeacherDashboard
│           └── components/               # AffectRadar, MasteryBar, PipelineStep, etc.
├── .env                                  # API keys & environment config
├── requirements.txt                      # Python dependencies
└── README.md
```

---

## 🚀 Installation & Setup

### Prerequisites
- **Python**: 3.10 to 3.12
- **Node.js**: v18+ (tested on Node v25) & npm v9+
- **MongoDB**: Running locally on `mongodb://localhost:27017/acdt` (or set `MONGO_URI` in `.env`)
- **CUDA (Optional)**: Highly recommended for training PyTorch models on GPU.

---

### Step 1: Clone the Repository & Configure Python Environment

```powershell
# Navigate to the project root
cd d:\ACDT

# Create virtual environment
python -m venv myenv

# Activate virtual environment (Windows PowerShell)
.\myenv\Scripts\Activate.ps1
```

Install Python dependencies:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

*(Optional GPU PyTorch)*:
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

### Step 2: Configure Environment Variables

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
MONGO_URI=mongodb://localhost:27017/acdt
PORT=5000
```

---

### Step 3: Dataset Preparation

Download the **ASSISTments 2012-2013 School Data with Affect Predictions** dataset:
- Source: [ASSISTments Affect Dataset](https://sites.google.com/site/assistmentsdata/home/2012-2013-school-data-with-affect)
- Place the raw CSV in:
  ```text
  data/raw/2012-2013-data-with-predictions-4-final/2012-2013-data-with-predictions-4-final.csv
  ```

---

## 🏃 Running the Machine Learning Pipeline

Execute the modules sequentially:

### 1. Data Cleaning & Feature Engineering
```powershell
python -m src.02_data_pipeline.01_clean_data
python -m src.02_data_pipeline.03_feature_construction
python -m src.02_data_pipeline.04_affect_vector
python -m src.02_data_pipeline.05_sequence_construction
python -m src.02_data_pipeline.07_student_split
```

### 2. Student Encoder (Transformer + Dual Head)
```powershell
python -m src.03_model.01_preprocess_inputs
python -m src.03_model.02_prepare_embeddings
python -m src.03_model.04_train_student_encoder
python -m src.03_model.05_evaluate_student_encoder
```

### 3. World Model & MCTS Planning
```powershell
# Extract transitions & train World Model
python -m src.03_model.07_prepare_world_model_data
python -m src.03_model.09_train_world_model

# Build candidate catalog & run MCTS
python -m src.03_model.13_build_candidate_catalogue
python -m src.03_model.16_mcts_planner
```

### 4. LLM Generation & Skill Lookups
```powershell
# Build lookup dictionaries from interactions
python -m src.04_llm.03_build_skill_lookup

# Generate lesson plan using OpenRouter
python -m src.04_llm.02_generate_lesson

# Multi-persona comparative evaluation
python -m src.04_llm.04_evaluate_lesson_personas
```

---

## 💻 Running the MERN Dashboard

The dashboard provides an interactive visual demonstration of the entire pipeline.

### 1. Start the Express API Server (Backend)
Open a terminal:
```powershell
cd d:\ACDT\dashboard\server
npm install
npm run dev
```
- Backend starts at: `http://localhost:5000`
- Verifies MongoDB connection and mounts `/api/run-pipeline`, `/api/student-state`, etc.

### 2. Start the React/Vite Client (Frontend)
Open a second terminal:
```powershell
cd d:\ACDT\dashboard\client
npm install
npm run dev
```
- Frontend starts at: `http://localhost:5173`

---

## 🖥️ Using the Dashboard

1. **▶ Live Demo Tab**:
   - Choose a student persona (**Frustrated**, **Confused**, **Bored**, **Engaged**) or toggle **Custom State** to adjust mastery and affective sliders.
   - Click **Run Digital Twin Pipeline →**.
   - Watch Server-Sent Events (SSE) stream the 7 execution phases in real-time.
   - When complete, view the live-generated LLM lesson plan, Affect Radar, and 3-step MCTS sequence.
2. **Student View Tab**:
   - Displays cognitive mastery progress, emotional radar dimensions, recommended next problems, and collapsible tiered hints.
3. **Teacher View Tab**:
   - Shows student cohort cards, flagging learners with high frustration or confusion for early teacher intervention.

---

## 🧪 Testing

Run automated tests to verify your environment and dependencies:
```powershell
python -m tests.test_environment
```

---

## 📜 Citation & Credits
- **Project Title**: Affective-Cognitive Digital Twins for Predictive Personalized Learning
- **Institution**: Department of Computer Science and Engineering, Vellore Institute of Technology, Chennai
- **Project Guide**: Dr. L. Shyamala, Professor
- **Authors**: Pragya Sekar (23BCE1491) & Fowzia Nilofer (23BCE5127)
