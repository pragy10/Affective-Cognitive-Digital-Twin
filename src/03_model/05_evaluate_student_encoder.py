"""
ACDT Phase 3.4 - Student Encoder Evaluation

Evaluates the trained Student Encoder on unseen test students.

Cognitive prediction:
    - MAE
    - RMSE
    - R²
    - Accuracy
    - Precision
    - Recall
    - F1
    - ROC-AUC
    - PR-AUC
    - Confusion matrix

Affective prediction:
    - MAE
    - RMSE
    - R²
    - Pearson correlation
    - Spearman correlation
    - Macro-average metrics

Outputs:
    - Cognitive ROC curve
    - Cognitive Precision-Recall curve
    - Cognitive confusion matrix
    - Affective predicted-vs-actual plots
    - Cognitive metrics CSV
    - Affective metrics CSV
    - Consolidated metrics CSV
    - Raw test predictions

No model training is performed.
"""

import importlib
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

from scipy.stats import pearsonr, spearmanr


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "student_encoder"

MODEL_INPUTS_FILE = PROCESSED_DIR / "model_inputs.npz"
MODEL_SEQUENCES_FILE = PROCESSED_DIR / "model_sequences.npz"
CATEGORICAL_INPUTS_FILE = PROCESSED_DIR / "categorical_inputs.npz"
SPLIT_FILE = PROCESSED_DIR / "student_split.npz"
VOCAB_FILE = PROCESSED_DIR / "embedding_vocab.npz"

CHECKPOINT_FILE = MODEL_DIR / "student_encoder_best.pt"

# Secondary binary evaluation for cognitive prediction.
# Correctness >= 0.5 is treated as "correct".
COGNITIVE_THRESHOLD = 0.5

BATCH_SIZE = 32


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("ACDT STUDENT ENCODER EVALUATION")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading processed inputs...")


# ------------------------------------------------------------
# Processed numerical model inputs
# ------------------------------------------------------------

model_inputs = np.load(
    MODEL_INPUTS_FILE
)

X = model_inputs["X"].astype(
    np.float32
)

mask = model_inputs["mask"].astype(
    np.int64
)


# ------------------------------------------------------------
# Targets and metadata
# ------------------------------------------------------------
# Targets were created during Phase 2.6 and are stored in
# model_sequences.npz, not model_inputs.npz.

model_sequences = np.load(
    MODEL_SEQUENCES_FILE
)

targets = model_sequences["targets"].astype(
    np.float32
)

affect_targets = model_sequences[
    "affect_targets"
].astype(np.float32)

target_mask = model_sequences[
    "target_mask"
].astype(np.int64)

user_ids = model_sequences[
    "user_ids"
]


# ------------------------------------------------------------
# Categorical model inputs
# ------------------------------------------------------------

categorical_inputs = np.load(
    CATEGORICAL_INPUTS_FILE
)

problem_indices = categorical_inputs[
    "problem_indices"
].astype(np.int64)

skill_indices = categorical_inputs[
    "skill_indices"
].astype(np.int64)

skill_missing = categorical_inputs[
    "skill_missing"
].astype(np.float32)


# ------------------------------------------------------------
# Student split
# ------------------------------------------------------------

splits = np.load(
    SPLIT_FILE
)

train_users = splits[
    "train_users"
]

val_users = splits[
    "val_users"
]

test_users = splits[
    "test_users"
]


# ------------------------------------------------------------
# Embedding vocabulary
# ------------------------------------------------------------

vocab = np.load(
    VOCAB_FILE
)

num_problems = (
    len(vocab["problem_ids"]) + 1
)

num_skills = (
    len(vocab["skill_ids"]) + 1
)


print(
    f"Total sequences: {len(X):,}"
)

print(
    f"Test users: {len(test_users):,}"
)


# ============================================================
# SELECT TEST SEQUENCES
# ============================================================

test_user_set = set(
    test_users.tolist()
)

test_sequence_mask = np.array(
    [
        uid in test_user_set
        for uid in user_ids
    ],
    dtype=bool,
)

test_indices = np.where(
    test_sequence_mask
)[0]


print(
    f"Test sequences: {len(test_indices):,}"
)


X_test = X[test_indices]

mask_test = mask[test_indices]

targets_test = targets[test_indices]

affect_targets_test = (
    affect_targets[test_indices]
)

target_mask_test = (
    target_mask[test_indices]
)

problem_indices_test = (
    problem_indices[test_indices]
)

skill_indices_test = (
    skill_indices[test_indices]
)

skill_missing_test = (
    skill_missing[test_indices]
)


# ============================================================
# TRAINING-SET BASELINE
# ============================================================
# The baseline is calculated ONLY from training students.
# It is therefore leakage-free.

train_user_set = set(
    train_users.tolist()
)

train_sequence_mask = np.array(
    [
        uid in train_user_set
        for uid in user_ids
    ],
    dtype=bool,
)

train_indices = np.where(
    train_sequence_mask
)[0]

train_target_mask = (
    target_mask[train_indices]
    .astype(bool)
)

train_cognitive_targets = (
    targets[train_indices][
        train_target_mask
    ]
)

baseline_cognitive = float(
    np.mean(train_cognitive_targets)
)


print(
    f"\nTraining-set cognitive baseline: "
    f"{baseline_cognitive:.6f}"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Student Encoder...")


student_encoder_module = importlib.import_module(
    "src.03_model.03_student_encoder"
)

StudentEncoder = (
    student_encoder_module.StudentEncoder
)


model = StudentEncoder(
    num_problems=num_problems,
    num_skills=num_skills,
)


checkpoint = torch.load(
    CHECKPOINT_FILE,
    map_location=DEVICE,
    weights_only=False,
)


if "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )


model.to(DEVICE)

model.eval()


print(
    "Checkpoint loaded successfully."
)


if isinstance(checkpoint, dict):

    if "epoch" in checkpoint:

        print(
            f"Checkpoint epoch: "
            f"{checkpoint['epoch']}"
        )

    if "val_loss" in checkpoint:

        print(
            f"Checkpoint validation loss: "
            f"{checkpoint['val_loss']:.6f}"
        )


# ============================================================
# MODEL PREDICTION
# ============================================================

print("\nGenerating test predictions...")


all_correctness_predictions = []

all_affect_predictions = []


num_test_sequences = len(X_test)


with torch.no_grad():

    for start in range(
        0,
        num_test_sequences,
        BATCH_SIZE,
    ):

        end = min(
            start + BATCH_SIZE,
            num_test_sequences,
        )


        numeric_batch = (
            torch.from_numpy(
                X_test[start:end]
            ).to(DEVICE)
        )


        problem_batch = (
            torch.from_numpy(
                problem_indices_test[
                    start:end
                ]
            ).to(DEVICE)
        )


        skill_batch = (
            torch.from_numpy(
                skill_indices_test[
                    start:end
                ]
            ).to(DEVICE)
        )


        skill_missing_batch = (
            torch.from_numpy(
                skill_missing_test[
                    start:end
                ]
            ).to(DEVICE)
        )


        padding_mask_batch = (
            torch.from_numpy(
                mask_test[start:end] == 0
            ).to(DEVICE)
        )


        outputs = model(
            numeric_features=numeric_batch,
            problem_indices=problem_batch,
            skill_indices=skill_batch,
            skill_missing=skill_missing_batch,
            padding_mask=padding_mask_batch,
        )


        correctness_prediction = (
            outputs[
                "next_correctness"
            ]
            .detach()
            .cpu()
            .numpy()
        )


        affect_prediction = (
            outputs[
                "next_affect"
            ]
            .detach()
            .cpu()
            .numpy()
        )


        all_correctness_predictions.append(
            correctness_prediction
        )

        all_affect_predictions.append(
            affect_prediction
        )


correctness_predictions = np.concatenate(
    all_correctness_predictions,
    axis=0,
)


affect_predictions = np.concatenate(
    all_affect_predictions,
    axis=0,
)


print(
    f"Prediction shape: "
    f"{correctness_predictions.shape}"
)

print(
    f"Affect prediction shape: "
    f"{affect_predictions.shape}"
)


# ============================================================
# FLATTEN VALID TARGETS
# ============================================================

valid_mask = (
    target_mask_test.astype(bool)
)


y_true_cognitive = (
    targets_test[valid_mask]
)

y_pred_cognitive = (
    correctness_predictions[valid_mask]
)


y_true_affect = (
    affect_targets_test[valid_mask]
)

y_pred_affect = (
    affect_predictions[valid_mask]
)


print(
    f"\nEvaluated cognitive targets: "
    f"{len(y_true_cognitive):,}"
)

print(
    f"Evaluated affect targets: "
    f"{len(y_true_affect):,}"
)


# ============================================================
# COGNITIVE REGRESSION METRICS
# ============================================================

print("\n" + "=" * 70)
print("COGNITIVE PREDICTION")
print("=" * 70)


cognitive_mae = mean_absolute_error(
    y_true_cognitive,
    y_pred_cognitive,
)


cognitive_rmse = np.sqrt(
    mean_squared_error(
        y_true_cognitive,
        y_pred_cognitive,
    )
)


cognitive_r2 = r2_score(
    y_true_cognitive,
    y_pred_cognitive,
)


print("\nRegression metrics:")

print(
    f"MAE:  {cognitive_mae:.6f}"
)

print(
    f"RMSE: {cognitive_rmse:.6f}"
)

print(
    f"R²:   {cognitive_r2:.6f}"
)


# ============================================================
# COGNITIVE BINARY CLASSIFICATION
# ============================================================

print("\nBinary classification metrics:")


y_true_binary = (
    y_true_cognitive
    >= COGNITIVE_THRESHOLD
).astype(int)


y_pred_binary = (
    y_pred_cognitive
    >= COGNITIVE_THRESHOLD
).astype(int)


accuracy = accuracy_score(
    y_true_binary,
    y_pred_binary,
)


precision = precision_score(
    y_true_binary,
    y_pred_binary,
    zero_division=0,
)


recall = recall_score(
    y_true_binary,
    y_pred_binary,
    zero_division=0,
)


f1 = f1_score(
    y_true_binary,
    y_pred_binary,
    zero_division=0,
)


roc_auc = roc_auc_score(
    y_true_binary,
    y_pred_cognitive,
)


pr_auc = average_precision_score(
    y_true_binary,
    y_pred_cognitive,
)


print(
    f"Threshold:  {COGNITIVE_THRESHOLD}"
)

print(
    f"Accuracy:   {accuracy:.6f}"
)

print(
    f"Precision:  {precision:.6f}"
)

print(
    f"Recall:     {recall:.6f}"
)

print(
    f"F1-score:   {f1:.6f}"
)

print(
    f"ROC-AUC:    {roc_auc:.6f}"
)

print(
    f"PR-AUC:     {pr_auc:.6f}"
)


# ============================================================
# COGNITIVE BASELINE COMPARISON
# ============================================================

baseline_cognitive_predictions = (
    np.full(
        len(y_true_cognitive),
        baseline_cognitive,
        dtype=np.float32,
    )
)


baseline_cognitive_mae = (
    mean_absolute_error(
        y_true_cognitive,
        baseline_cognitive_predictions,
    )
)


baseline_cognitive_rmse = np.sqrt(
    mean_squared_error(
        y_true_cognitive,
        baseline_cognitive_predictions,
    )
)


baseline_cognitive_r2 = r2_score(
    y_true_cognitive,
    baseline_cognitive_predictions,
)


# Binary baseline

baseline_binary_predictions = (
    baseline_cognitive_predictions
    >= COGNITIVE_THRESHOLD
).astype(int)


baseline_accuracy = accuracy_score(
    y_true_binary,
    baseline_binary_predictions,
)


baseline_precision = precision_score(
    y_true_binary,
    baseline_binary_predictions,
    zero_division=0,
)


baseline_recall = recall_score(
    y_true_binary,
    baseline_binary_predictions,
    zero_division=0,
)


baseline_f1 = f1_score(
    y_true_binary,
    baseline_binary_predictions,
    zero_division=0,
)


# A constant prediction cannot meaningfully rank examples,
# so ROC-AUC is not used for the constant baseline.
baseline_roc_auc = np.nan

baseline_pr_auc = np.nan


print("\nBaseline comparison:")

print(
    f"Baseline MAE:  "
    f"{baseline_cognitive_mae:.6f}"
)

print(
    f"Encoder MAE:   "
    f"{cognitive_mae:.6f}"
)

print(
    f"MAE improvement: "
    f"{100 * (baseline_cognitive_mae - cognitive_mae) / baseline_cognitive_mae:.2f}%"
)


print(
    f"Baseline RMSE: "
    f"{baseline_cognitive_rmse:.6f}"
)

print(
    f"Encoder RMSE:  "
    f"{cognitive_rmse:.6f}"
)

print(
    f"RMSE improvement: "
    f"{100 * (baseline_cognitive_rmse - cognitive_rmse) / baseline_cognitive_rmse:.2f}%"
)


print(
    f"Baseline R²:   "
    f"{baseline_cognitive_r2:.6f}"
)

print(
    f"Encoder R²:    "
    f"{cognitive_r2:.6f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true_binary,
    y_pred_binary,
)


tn, fp, fn, tp = cm.ravel()


print("\nConfusion matrix:")

print(cm)


print(
    f"\nTrue Negatives:  {tn:,}"
)

print(
    f"False Positives: {fp:,}"
)

print(
    f"False Negatives: {fn:,}"
)

print(
    f"True Positives:  {tp:,}"
)


# ============================================================
# ROC CURVE
# ============================================================

fpr, tpr, roc_thresholds = roc_curve(
    y_true_binary,
    y_pred_cognitive,
)


plt.figure(figsize=(7, 6))


plt.plot(
    fpr,
    tpr,
    label=(
        f"Student Encoder "
        f"(AUC = {roc_auc:.4f})"
    ),
)


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier",
)


plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "Cognitive Prediction ROC Curve"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3,
)

plt.tight_layout()


roc_path = (
    RESULTS_DIR
    / "cognitive_roc_curve.png"
)


plt.savefig(
    roc_path,
    dpi=300,
    bbox_inches="tight",
)


plt.close()


print(
    f"\nROC curve saved: "
    f"{roc_path}"
)


# ============================================================
# PRECISION-RECALL CURVE
# ============================================================

precision_curve, recall_curve, pr_thresholds = (
    precision_recall_curve(
        y_true_binary,
        y_pred_cognitive,
    )
)


positive_rate = float(
    np.mean(y_true_binary)
)


plt.figure(figsize=(7, 6))


plt.plot(
    recall_curve,
    precision_curve,
    label=(
        f"Student Encoder "
        f"(AP = {pr_auc:.4f})"
    ),
)


plt.axhline(
    positive_rate,
    linestyle="--",
    label=(
        f"Class baseline "
        f"({positive_rate:.3f})"
    ),
)


plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Cognitive Prediction "
    "Precision-Recall Curve"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3,
)

plt.tight_layout()


pr_path = (
    RESULTS_DIR
    / "cognitive_precision_recall_curve.png"
)


plt.savefig(
    pr_path,
    dpi=300,
    bbox_inches="tight",
)


plt.close()


print(
    f"PR curve saved: "
    f"{pr_path}"
)


# ============================================================
# CONFUSION MATRIX PLOT
# ============================================================

plt.figure(figsize=(6, 5))


plt.imshow(cm)


plt.title(
    "Cognitive Prediction "
    "Confusion Matrix"
)


plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "Actual Label"
)


plt.xticks(
    [0, 1],
    [
        "Incorrect",
        "Correct",
    ],
)


plt.yticks(
    [0, 1],
    [
        "Incorrect",
        "Correct",
    ],
)


for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            f"{cm[i, j]:,}",
            ha="center",
            va="center",
        )


plt.colorbar()

plt.tight_layout()


cm_path = (
    RESULTS_DIR
    / "cognitive_confusion_matrix.png"
)


plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight",
)


plt.close()


print(
    f"Confusion matrix saved: "
    f"{cm_path}"
)


# ============================================================
# AFFECTIVE METRICS
# ============================================================

print("\n" + "=" * 70)
print("AFFECTIVE PREDICTION")
print("=" * 70)


affect_names = [
    "Frustration",
    "Confusion",
    "Concentration",
    "Boredom",
]


affect_results = []


for i, affect_name in enumerate(
    affect_names
):

    true_values = (
        y_true_affect[:, i]
    )

    predicted_values = (
        y_pred_affect[:, i]
    )


    mae = mean_absolute_error(
        true_values,
        predicted_values,
    )


    rmse = np.sqrt(
        mean_squared_error(
            true_values,
            predicted_values,
        )
    )


    r2 = r2_score(
        true_values,
        predicted_values,
    )


    pearson_r, pearson_p = pearsonr(
        true_values,
        predicted_values,
    )


    spearman_r, spearman_p = spearmanr(
        true_values,
        predicted_values,
    )


    affect_results.append(
        {
            "affect": affect_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Pearson_r": pearson_r,
            "Pearson_p": pearson_p,
            "Spearman_r": spearman_r,
            "Spearman_p": spearman_p,
        }
    )


    print(
        f"\n{affect_name}"
    )

    print(
        f"MAE:         {mae:.6f}"
    )

    print(
        f"RMSE:        {rmse:.6f}"
    )

    print(
        f"R²:          {r2:.6f}"
    )

    print(
        f"Pearson r:   {pearson_r:.6f}"
    )

    print(
        f"Spearman r:  {spearman_r:.6f}"
    )


affect_results_df = pd.DataFrame(
    affect_results
)


# ============================================================
# POOLED AFFECT METRICS
# ============================================================
# These are calculated by flattening all four dimensions.
# They are retained for reference, but macro metrics below
# are preferred for reporting.

pooled_affect_mae = (
    mean_absolute_error(
        y_true_affect.reshape(-1),
        y_pred_affect.reshape(-1),
    )
)


pooled_affect_rmse = np.sqrt(
    mean_squared_error(
        y_true_affect.reshape(-1),
        y_pred_affect.reshape(-1),
    )
)


pooled_affect_r2 = r2_score(
    y_true_affect.reshape(-1),
    y_pred_affect.reshape(-1),
)


print("\nPooled affect metrics:")

print(
    f"MAE:  {pooled_affect_mae:.6f}"
)

print(
    f"RMSE: {pooled_affect_rmse:.6f}"
)

print(
    f"R²:   {pooled_affect_r2:.6f}"
)


# ============================================================
# MACRO-AVERAGE AFFECT METRICS
# ============================================================

macro_affect_mae = (
    affect_results_df["MAE"].mean()
)

macro_affect_rmse = (
    affect_results_df["RMSE"].mean()
)

macro_affect_r2 = (
    affect_results_df["R2"].mean()
)

macro_pearson = (
    affect_results_df["Pearson_r"].mean()
)

macro_spearman = (
    affect_results_df["Spearman_r"].mean()
)


print("\nMacro-average affect metrics:")

print(
    f"MAE:          {macro_affect_mae:.6f}"
)

print(
    f"RMSE:         {macro_affect_rmse:.6f}"
)

print(
    f"R²:           {macro_affect_r2:.6f}"
)

print(
    f"Pearson r:    {macro_pearson:.6f}"
)

print(
    f"Spearman r:   {macro_spearman:.6f}"
)


# ============================================================
# AFFECT PREDICTED VS ACTUAL PLOTS
# ============================================================

for i, affect_name in enumerate(
    affect_names
):

    true_values = (
        y_true_affect[:, i]
    )

    predicted_values = (
        y_pred_affect[:, i]
    )


    plt.figure(
        figsize=(7, 6)
    )


    plt.scatter(
        true_values,
        predicted_values,
        s=4,
        alpha=0.2,
    )


    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )


    plt.xlabel(
        "Actual"
    )

    plt.ylabel(
        "Predicted"
    )


    plt.title(
        f"{affect_name}: "
        f"Predicted vs Actual"
    )


    plt.xlim(
        0,
        1,
    )

    plt.ylim(
        0,
        1,
    )


    plt.grid(
        True,
        alpha=0.3,
    )


    plt.tight_layout()


    filename = (
        affect_name.lower()
        + "_predicted_vs_actual.png"
    )


    path = (
        RESULTS_DIR
        / filename
    )


    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )


    plt.close()


    print(
        f"{affect_name} plot saved: "
        f"{path}"
    )


# ============================================================
# SAVE COGNITIVE METRICS
# ============================================================

cognitive_results = {
    "task": "cognitive",
    "target": "next_correctness",

    "MAE": cognitive_mae,
    "RMSE": cognitive_rmse,
    "R2": cognitive_r2,

    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1": f1,

    "ROC_AUC": roc_auc,
    "PR_AUC": pr_auc,

    "Threshold": COGNITIVE_THRESHOLD,

    "Baseline_MAE": baseline_cognitive_mae,
    "Baseline_RMSE": baseline_cognitive_rmse,
    "Baseline_R2": baseline_cognitive_r2,

    "MAE_Improvement_Percent":
        100
        * (
            baseline_cognitive_mae
            - cognitive_mae
        )
        / baseline_cognitive_mae,

    "RMSE_Improvement_Percent":
        100
        * (
            baseline_cognitive_rmse
            - cognitive_rmse
        )
        / baseline_cognitive_rmse,

    "True_Negatives": tn,
    "False_Positives": fp,
    "False_Negatives": fn,
    "True_Positives": tp,
}


cognitive_results_df = pd.DataFrame(
    [cognitive_results]
)


cognitive_csv = (
    RESULTS_DIR
    / "cognitive_metrics.csv"
)


cognitive_results_df.to_csv(
    cognitive_csv,
    index=False,
)


# ============================================================
# SAVE AFFECTIVE METRICS
# ============================================================

affect_csv = (
    RESULTS_DIR
    / "affective_metrics.csv"
)


affect_results_df.to_csv(
    affect_csv,
    index=False,
)


# ============================================================
# SAVE CONSOLIDATED METRICS
# ============================================================

all_metrics = []


all_metrics.append(
    {
        "task": "cognitive",
        "target": "next_correctness",

        "MAE": cognitive_mae,
        "RMSE": cognitive_rmse,
        "R2": cognitive_r2,

        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,

        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,

        "Pearson_r": np.nan,
        "Spearman_r": np.nan,

        "Baseline_MAE":
            baseline_cognitive_mae,

        "Baseline_RMSE":
            baseline_cognitive_rmse,

        "Baseline_R2":
            baseline_cognitive_r2,
    }
)


for row in affect_results:

    all_metrics.append(
        {
            "task": "affective",
            "target": row["affect"],

            "MAE": row["MAE"],
            "RMSE": row["RMSE"],
            "R2": row["R2"],

            "Accuracy": np.nan,
            "Precision": np.nan,
            "Recall": np.nan,
            "F1": np.nan,

            "ROC_AUC": np.nan,
            "PR_AUC": np.nan,

            "Pearson_r":
                row["Pearson_r"],

            "Spearman_r":
                row["Spearman_r"],

            "Baseline_MAE": np.nan,
            "Baseline_RMSE": np.nan,
            "Baseline_R2": np.nan,
        }
    )


all_metrics_df = pd.DataFrame(
    all_metrics
)


all_metrics_path = (
    RESULTS_DIR
    / "all_metrics.csv"
)


all_metrics_df.to_csv(
    all_metrics_path,
    index=False,
)


# ============================================================
# SAVE RAW PREDICTIONS
# ============================================================

predictions_path = (
    RESULTS_DIR
    / "test_predictions.npz"
)


np.savez_compressed(
    predictions_path,

    cognitive_true=y_true_cognitive,

    cognitive_pred=y_pred_cognitive,

    cognitive_true_binary=y_true_binary,

    cognitive_pred_binary=y_pred_binary,

    affect_true=y_true_affect,

    affect_pred=y_pred_affect,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL EVALUATION SUMMARY")
print("=" * 70)


print("\nCognitive prediction")

print(
    f"MAE:        {cognitive_mae:.6f}"
)

print(
    f"RMSE:       {cognitive_rmse:.6f}"
)

print(
    f"R²:         {cognitive_r2:.6f}"
)

print(
    f"Accuracy:   {accuracy:.6f}"
)

print(
    f"Precision:  {precision:.6f}"
)

print(
    f"Recall:     {recall:.6f}"
)

print(
    f"F1:         {f1:.6f}"
)

print(
    f"ROC-AUC:    {roc_auc:.6f}"
)

print(
    f"PR-AUC:     {pr_auc:.6f}"
)


print("\nCognitive baseline")

print(
    f"MAE:        "
    f"{baseline_cognitive_mae:.6f}"
)

print(
    f"RMSE:       "
    f"{baseline_cognitive_rmse:.6f}"
)

print(
    f"R²:         "
    f"{baseline_cognitive_r2:.6f}"
)


print("\nAffective prediction")

for row in affect_results:

    print(
        f"{row['affect']:<15} "
        f"MAE={row['MAE']:.6f}  "
        f"RMSE={row['RMSE']:.6f}  "
        f"R²={row['R2']:.6f}  "
        f"Pearson={row['Pearson_r']:.6f}  "
        f"Spearman={row['Spearman_r']:.6f}"
    )


print("\nMacro-average affect")

print(
    f"MAE:          "
    f"{macro_affect_mae:.6f}"
)

print(
    f"RMSE:         "
    f"{macro_affect_rmse:.6f}"
)

print(
    f"R²:           "
    f"{macro_affect_r2:.6f}"
)

print(
    f"Pearson r:    "
    f"{macro_pearson:.6f}"
)

print(
    f"Spearman r:   "
    f"{macro_spearman:.6f}"
)


print("\nSaved results:")

print(
    f"  {RESULTS_DIR}"
)

print(
    f"\n  {cognitive_csv}"
)

print(
    f"  {affect_csv}"
)

print(
    f"  {all_metrics_path}"
)

print(
    f"  {predictions_path}"
)


print("\nACDT STUDENT ENCODER EVALUATION PASSED")