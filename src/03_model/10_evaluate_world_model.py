
import importlib.util
import time
from pathlib import Path

import numpy as np
import torch


# ============================================================
# ACDT PHASE 4.4 - WORLD MODEL TEST EVALUATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "processed" / "world_model_transitions.npz"
MODEL_FILE = ROOT / "src" / "03_model" / "08_world_model.py"
CHECKPOINT_PATH = ROOT / "models" / "world_model_best.pt"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 1024
PROBLEM_VOCAB_SIZE = 180000
SKILL_VOCAB_SIZE = 266

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 64)
print("ACDT PHASE 4.4 - WORLD MODEL TEST EVALUATION")
print("=" * 64)
print(f"Device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Transition data not found: {DATA_PATH}")

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")


# ------------------------------------------------------------
# Load test transitions
# ------------------------------------------------------------

print("\nLoading transition data...", flush=True)

data = np.load(DATA_PATH, allow_pickle=False)

split = data["split"]
test_indices = np.flatnonzero(split == 2)

print(f"Test transitions: {len(test_indices):,}")
print(f"Batch size: {BATCH_SIZE}")
print(
    f"Test batches: "
    f"{int(np.ceil(len(test_indices) / BATCH_SIZE)):,}"
)

current_states = data["current_states"]
next_states = data["next_states"]
problem_indices = data["action_problem_indices"]
skill_indices = data["action_skill_indices"]
skill_missing = data["action_skill_missing"]

data.close()


# ------------------------------------------------------------
# Load the exact World Model architecture
# ------------------------------------------------------------

spec = importlib.util.spec_from_file_location(
    "world_model_module",
    str(MODEL_FILE),
)

if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load model definition: {MODEL_FILE}")

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

WorldModel = module.WorldModel

model = WorldModel(
    num_problems=PROBLEM_VOCAB_SIZE,
    num_skills=SKILL_VOCAB_SIZE,
).to(device)

print("\nLoading best checkpoint...", flush=True)

state_dict = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=True,
)
model.load_state_dict(state_dict)
model.eval()

print(f"Checkpoint loaded: {CHECKPOINT_PATH}")


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

total_nll = 0.0
total_squared_error = 0.0
total_absolute_error = 0.0
total_elements = 0

within_one_std = 0
within_196_std = 0
total_predicted_std = 0.0
total_std_elements = 0

num_batches = int(np.ceil(len(test_indices) / BATCH_SIZE))
start_time = time.time()

print("\nEvaluating on held-out test students...", flush=True)

with torch.inference_mode():
    for batch_number, start_idx in enumerate(
        range(0, len(test_indices), BATCH_SIZE),
        start=1,
    ):
        indices = test_indices[start_idx:start_idx + BATCH_SIZE]

        x = torch.from_numpy(
            np.asarray(current_states[indices], dtype=np.float32)
        ).to(device)

        target = torch.from_numpy(
            np.asarray(next_states[indices], dtype=np.float32)
        ).to(device)

        problems = torch.from_numpy(
            np.asarray(problem_indices[indices], dtype=np.int64)
        ).to(device)

        skills = torch.from_numpy(
            np.asarray(skill_indices[indices], dtype=np.int64)
        ).to(device)

        missing = torch.from_numpy(
            np.asarray(skill_missing[indices], dtype=np.float32)
        ).to(device)

        output = model(x, problems, skills, missing)

        mean = output["next_state_mean"]
        std = torch.clamp(
            output["next_state_std"],
            min=1e-3,
            max=10.0,
        )

        if not torch.isfinite(mean).all():
            raise RuntimeError(
                f"Non-finite predictions in batch {batch_number}"
            )

        if not torch.isfinite(std).all():
            raise RuntimeError(
                f"Non-finite predicted uncertainty in batch {batch_number}"
            )

        if (std <= 0).any():
            raise RuntimeError(
                f"Non-positive standard deviation in batch {batch_number}"
            )

        error = target - mean
        variance = std.square()

        # Same Gaussian NLL objective used during training.
        # The constant 0.5 * log(2*pi) is omitted.
        nll = 0.5 * (
            torch.log(variance)
            + error.square() / variance
        )

        total_nll += nll.sum().item()
        total_squared_error += error.square().sum().item()
        total_absolute_error += error.abs().sum().item()
        total_elements += target.numel()

        absolute_error = error.abs()

        within_one_std += (
            absolute_error <= std
        ).sum().item()

        within_196_std += (
            absolute_error <= 1.96 * std
        ).sum().item()

        total_predicted_std += std.sum().item()
        total_std_elements += std.numel()

        if (
            batch_number % 100 == 0
            or batch_number == num_batches
        ):
            print(
                f"  Test: batch {batch_number:,}/{num_batches:,} "
                f"({100 * batch_number / num_batches:.1f}%)",
                flush=True,
            )


# ------------------------------------------------------------
# Calculate final metrics
# ------------------------------------------------------------

test_nll = total_nll / total_elements
test_mse = total_squared_error / total_elements
test_rmse = float(np.sqrt(test_mse))
test_mae = total_absolute_error / total_elements

coverage_1std = within_one_std / total_elements
coverage_196std = within_196_std / total_elements
mean_predicted_std = total_predicted_std / total_std_elements

elapsed = time.time() - start_time

print("\n" + "=" * 64)
print("WORLD MODEL TEST RESULTS")
print("=" * 64)

print(f"Test transitions:                  {len(test_indices):,}")
print(f"Latent dimensions evaluated:       {total_elements:,}")
print(f"Test Gaussian NLL (no constant):   {test_nll:.6f}")
print(f"Test MSE:                          {test_mse:.6f}")
print(f"Test RMSE:                         {test_rmse:.6f}")
print(f"Test MAE:                          {test_mae:.6f}")

print("\nUncertainty diagnostics:")
print(f"Mean predicted standard deviation: {mean_predicted_std:.6f}")
print(f"Residuals within predicted 1 std:  {coverage_1std:.4f}")
print(f"Residuals within predicted 1.96 std: {coverage_196std:.4f}")

print(f"\nEvaluation time: {elapsed / 60:.2f} minutes")

# Save metrics for reporting.
results_path = RESULTS_DIR / "world_model_test_metrics.npz"

np.savez(
    results_path,
    test_transitions=len(test_indices),
    test_nll=test_nll,
    test_mse=test_mse,
    test_rmse=test_rmse,
    test_mae=test_mae,
    mean_predicted_std=mean_predicted_std,
    coverage_1std=coverage_1std,
    coverage_196std=coverage_196std,
)

print(f"Metrics saved to: {results_path}")
print("\nPhase 4.4 evaluation completed.")
