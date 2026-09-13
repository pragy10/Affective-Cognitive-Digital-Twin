# Affective-Cognitive Digital Twin (ACDT)

Affective-Cognitive Digital Twin (ACDT) is a student modeling system that learns a student's **cognitive and affective state** from interaction history.

The current implementation uses the **ASSISTments 2012–2013 School Data with Affect Predictions** dataset.

---

## 1. Dataset

The project uses:

**ASSISTments 2012–2013 School Data with Affect Predictions**

Dataset source:

https://sites.google.com/site/assistmentsdata/home/2012-2013-school-data-with-affect

Download the dataset from the source above.

After downloading and extracting it, place the CSV inside:

```text
data/raw/2012-2013-data-with-predictions-4-final/
```

The expected file is:

```text
data/raw/
└── 2012-2013-data-with-predictions-4-final/
    └── 2012-2013-data-with-predictions-4-final.csv
```

The dataset should **not be committed to GitHub** because it is large and is excluded through `.gitignore`.

---

## 2. Environment Setup

Clone the repository and open a terminal in the project directory.

Create and activate a Python virtual environment:

```powershell
python -m venv myenv
.\myenv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
pip install -r requirements.txt
```

The project uses PyTorch for the Student Encoder.

For GPU usage, install a PyTorch build compatible with your system. The development environment used for this project currently uses:

```text
PyTorch 2.6.0 + CUDA 11.8
```

---

# 3. Project Structure

```text
ACDT/
│
├── data/
│   ├── raw/
│   │   └── 2012-2013-data-with-predictions-4-final/
│   │       └── 2012-2013-data-with-predictions-4-final.csv
│   │
│   └── processed/
│
├── notebooks/
│
├── results/
│
├── src/
│   │
│   ├── 01_data_investigation/
│   │   ├── 01_basic_inspection.py
│   │   ├── 02_trajectory_inspection.py
│   │   ├── 03_affect_investigation.py
│   │   ├── 04_sequence_length_investigation.py
│   │   ├── 05_data_quality_investigation.py
│   │   ├── 06_feature_feasibility_investigation.py
│   │   └── 07_problem_skill_mapping.py
│   │
│   ├── 02_data_pipeline/
│   │   ├── 01_clean_data.py
│   │   ├── 02_validate_clean_data.py
│   │   ├── 03_feature_construction.py
│   │   ├── 04_affect_vector.py
│   │   ├── 05_sequence_construction.py
│   │   ├── 06_target_and_metadata.py
│   │   ├── 07_student_split.py
│   │   └── 08_validate_pipeline.py
│   │
│   └── 03_model/
│       ├── 01_preprocess_inputs.py
│       ├── 02_prepare_embeddings.py
│       ├── 03_student_encoder.py
│       ├── 04_train_student_encoder.py
│       └── 05_evaluate_student_encoder.py
│
├── tests/
│   └── test_environment.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

# 4. Execution Order

The scripts are organized into sequential phases.

**Do not run the files in random order.**

## Step 0 — Check the environment

```powershell
python -m tests.test_environment
```

This checks the Python and major library versions.

---

# Phase 1 — Data Investigation

These scripts investigate the raw dataset before preprocessing.

Run in this order:

```powershell
python -m src.01_data_investigation.01_basic_inspection
```

```powershell
python -m src.01_data_investigation.02_trajectory_inspection
```

```powershell
python -m src.01_data_investigation.03_affect_investigation
```

```powershell
python -m src.01_data_investigation.04_sequence_length_investigation
```

```powershell
python -m src.01_data_investigation.05_data_quality_investigation
```

```powershell
python -m src.01_data_investigation.06_feature_feasibility_investigation
```

```powershell
python -m src.01_data_investigation.07_problem_skill_mapping
```

These scripts are mainly for understanding the dataset and deciding how the preprocessing pipeline should work.

---

# Phase 2 — Data Pipeline

Run the preprocessing scripts in this exact order.

### 2.1 Clean the raw data

```powershell
python -m src.02_data_pipeline.01_clean_data
```

Creates:

```text
data/processed/clean_interactions.csv
```

### 2.2 Validate the cleaned data

```powershell
python -m src.02_data_pipeline.02_validate_clean_data
```

### 2.3 Construct behavioral features

```powershell
python -m src.02_data_pipeline.03_feature_construction
```

Creates:

```text
data/processed/featured_interactions.csv
```

### 2.4 Construct the affect vector

```powershell
python -m src.02_data_pipeline.04_affect_vector
```

Creates:

```text
data/processed/affect_interactions.csv
```

### 2.5 Construct student sequences

```powershell
python -m src.02_data_pipeline.05_sequence_construction
```

Creates:

```text
data/processed/sequences.npz
```

### 2.6 Create targets and metadata

```powershell
python -m src.02_data_pipeline.06_target_and_metadata
```

Creates:

```text
data/processed/model_sequences.npz
```

### 2.7 Split students

```powershell
python -m src.02_data_pipeline.07_student_split
```

Creates:

```text
data/processed/student_split.npz
```

The split is performed at the **student level** to prevent information from the same student appearing in both training and evaluation sets.

### 2.8 Validate the complete pipeline

```powershell
python -m src.02_data_pipeline.08_validate_pipeline
```

This should pass before proceeding to the model stage.

---

# Phase 3 — Student Encoder

## 3.1 Preprocess model inputs

```powershell
python -m src.03_model.01_preprocess_inputs
```

Creates:

```text
data/processed/model_inputs.npz
data/processed/preprocessing_stats.npz
```

## 3.2 Prepare problem and skill embeddings

```powershell
python -m src.03_model.02_prepare_embeddings
```

Creates:

```text
data/processed/categorical_inputs.npz
data/processed/embedding_vocab.npz
```

## 3.3 Train the Student Encoder

```powershell
python -m src.03_model.04_train_student_encoder
```

The model learns from:

* behavioral features
* affect features
* problem embeddings
* skill embeddings
* missing-skill information
* temporal sequence information

The Student Encoder produces:

* cognitive latent state
* affective latent state
* joint student state
* next correctness prediction
* next 4-dimensional affect prediction

The best trained model is saved to:

```text
models/student_encoder_best.pt
```

The `models/` directory is excluded from GitHub.

## 3.4 Evaluate the Student Encoder

After training:

```powershell
python -m src.03_model.05_evaluate_student_encoder
```

This evaluates the trained model on the held-out test students.

---

# 5. Generated Files

The following files are generated automatically during execution:

```text
data/processed/
├── clean_interactions.csv
├── featured_interactions.csv
├── affect_interactions.csv
├── sequences.npz
├── model_sequences.npz
├── student_split.npz
├── model_inputs.npz
├── preprocessing_stats.npz
├── categorical_inputs.npz
└── embedding_vocab.npz
```

Trained models are generated under:

```text
models/
└── student_encoder_best.pt
```

These generated files are intentionally excluded from version control.

---

# 6. Quick Start

After placing the dataset in `data/raw/`, the main pipeline is:

```powershell
# Environment
python -m tests.test_environment

# Phase 1
python -m src.01_data_investigation.01_basic_inspection
python -m src.01_data_investigation.02_trajectory_inspection
python -m src.01_data_investigation.03_affect_investigation
python -m src.01_data_investigation.04_sequence_length_investigation
python -m src.01_data_investigation.05_data_quality_investigation
python -m src.01_data_investigation.06_feature_feasibility_investigation
python -m src.01_data_investigation.07_problem_skill_mapping

# Phase 2
python -m src.02_data_pipeline.01_clean_data
python -m src.02_data_pipeline.02_validate_clean_data
python -m src.02_data_pipeline.03_feature_construction
python -m src.02_data_pipeline.04_affect_vector
python -m src.02_data_pipeline.05_sequence_construction
python -m src.02_data_pipeline.06_target_and_metadata
python -m src.02_data_pipeline.07_student_split
python -m src.02_data_pipeline.08_validate_pipeline

# Phase 3
python -m src.03_model.01_preprocess_inputs
python -m src.03_model.02_prepare_embeddings
python -m src.03_model.04_train_student_encoder
python -m src.03_model.05_evaluate_student_encoder
```

The investigation scripts in Phase 1 are primarily for dataset analysis. For a fresh run where the preprocessing decisions are already established, the **data-processing pipeline begins with Phase 2**.

---

## Current Implementation Status

The repository currently contains the implementation through the **Student Encoder** stage.

Implemented:

```text
Dataset
   ↓
Data Investigation
   ↓
Data Cleaning
   ↓
Feature Construction
   ↓
Affect Representation
   ↓
Student Sequences
   ↓
Targets + Metadata
   ↓
Student-Level Split
   ↓
Model Input Preprocessing
   ↓
Problem / Skill Embeddings
   ↓
Dual-State Student Encoder
   ↓
Cognitive + Affective Prediction
```

The later ACDT components, including the World Model, counterfactual planning, MCTS, LLM lesson planning, dashboard, and closed-loop retraining, are developed separately from this pipeline.
