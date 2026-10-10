
import importlib.util
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader  # Not required; retained for compatibility


# ============================================================
# ACDT PHASE 4.3 - MEMORY-EFFICIENT WORLD MODEL TRAINING
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "processed" / "world_model_transitions.npz"
MODEL_FILE = ROOT / "src" / "03_model" / "08_world_model.py"
CHECKPOINT_DIR = ROOT / "models"
CHECKPOINT_PATH = CHECKPOINT_DIR / "world_model_best.pt"

BATCH_SIZE = 1024
EPOCHS = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PROGRESS_EVERY = 100
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("=" * 64)
print("ACDT PHASE 4.3 - WORLD MODEL TRAINING")
print("=" * 64)
print(f"Device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("\nLoading transition data...")
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Transition file not found: {DATA_PATH}")

# Keep the original arrays. Do NOT create separate train/validation
# copies of the large latent-state arrays.
data = np.load(DATA_PATH, allow_pickle=False)

current_states = data["current_states"]
next_states = data["next_states"]
problem_indices = data["action_problem_indices"]
skill_indices = data["action_skill_indices"]
skill_missing = data["action_skill_missing"]
split = data["split"]

total = len(split)

# These are small integer index arrays, not copies of the state data.
train_indices = np.flatnonzero(split == 0)
val_indices = np.flatnonzero(split == 1)
test_indices = np.flatnonzero(split == 2)

print(f"Total transitions:      {total:,}")
print(f"Training transitions:   {len(train_indices):,}")
print(f"Validation transitions: {len(val_indices):,}")
print(f"Test transitions:       {len(test_indices):,}")
print(f"Batch size:             {BATCH_SIZE}")
print(f"Training batches/epoch: {int(np.ceil(len(train_indices) / BATCH_SIZE)):,}")
print(f"Validation batches:     {int(np.ceil(len(val_indices) / BATCH_SIZE)):,}")

# Release the NPZ handle. The arrays above remain available in memory.
data.close()


# ------------------------------------------------------------
# Load the existing World Model class
# ------------------------------------------------------------

spec = importlib.util.spec_from_file_location(
    "world_model_module",
    str(MODEL_FILE),
)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load model definition: {MODEL_FILE}")

world_model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(world_model_module)

WorldModel = world_model_module.WorldModel

# The vocabulary sizes match Phase 3's saved embedding vocabulary.
vocab_path = ROOT / "data" / "processed" / "embedding_vocab.npz"

if not vocab_path.exists():
    raise FileNotFoundError(
        f"Embedding vocabulary not found: {vocab_path}"
    )

vocab = np.load(vocab_path, allow_pickle=False)

# These sizes are grounded in the Phase 3 vocabulary.
num_problems = 180000
num_skills = 266

print("\nLoading World Model...")
model = WorldModel(
    num_problems=num_problems,
    num_skills=num_skills,
).to(device)

print(f"Problem vocabulary: {num_problems:,}")
print(f"Skill vocabulary:   {num_skills:,}")
print(f"Total parameters:   {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ------------------------------------------------------------
# Batch preparation
# ------------------------------------------------------------

def make_batch(indices):
    """Fetch only one batch from the original arrays."""
    x = torch.from_numpy(
        np.asarray(current_states[indices], dtype=np.float32)
    ).to(device)

    y = torch.from_numpy(
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

    return x, y, problems, skills, missing


def gaussian_nll(target, mean, std):
    """Gaussian negative log-likelihood, averaged over latent dimensions."""
    std = torch.clamp(std, min=1e-3, max=10.0)
    variance = std.square()

    loss = 0.5 * (
        torch.log(variance)
        + (target - mean).square() / variance
    )
    return loss.mean()


def predict_batch(batch):
    x, y, problems, skills, missing = batch

    output = model(x, problems, skills, missing)
    mean = output["next_state_mean"]
    std = output["next_state_std"]

    loss = gaussian_nll(y, mean, std)
    mse = F.mse_loss(mean, y)

    return loss, mse


# ------------------------------------------------------------
# Validation: iterate over batches without copying all val data
# ------------------------------------------------------------

def evaluate(indices, label="Validation"):
    model.eval()

    total_nll = 0.0
    total_squared_error = 0.0
    total_elements = 0
    total_rows = len(indices)
    num_batches = int(np.ceil(total_rows / BATCH_SIZE))
    start = time.time()

    with torch.inference_mode():
        for batch_number, start_idx in enumerate(
            range(0, total_rows, BATCH_SIZE), start=1
        ):
            batch_indices = indices[start_idx:start_idx + BATCH_SIZE]
            batch = make_batch(batch_indices)

            x, y, problems, skills, missing = batch
            output = model(x, problems, skills, missing)

            mean = output["next_state_mean"]
            std = torch.clamp(
                output["next_state_std"], min=1e-3, max=10.0
            )

            variance = std.square()
            nll = 0.5 * (
                torch.log(variance)
                + (y - mean).square() / variance
            )

            total_nll += nll.sum().item()
            total_squared_error += (mean - y).square().sum().item()
            total_elements += y.numel()

            if (
                batch_number % PROGRESS_EVERY == 0
                or batch_number == num_batches
            ):
                print(
                    f"  {label}: batch {batch_number:,}/{num_batches:,} "
                    f"({100 * batch_number / num_batches:.1f}%)",
                    flush=True,
                )

    avg_nll = total_nll / total_elements
    avg_mse = total_squared_error / total_elements
    elapsed = time.time() - start

    print(
        f"{label} complete | NLL: {avg_nll:.6f} | "
        f"MSE: {avg_mse:.6f} | Time: {elapsed / 60:.1f} min",
        flush=True,
    )

    return avg_nll, avg_mse


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

best_val_nll = float("inf")
rng = np.random.default_rng(SEED)

print("\nStarting training...", flush=True)

try:
    for epoch in range(1, EPOCHS + 1):
        model.train()
        rng.shuffle(train_indices)

        num_batches = int(np.ceil(len(train_indices) / BATCH_SIZE))
        total_train_loss = 0.0
        total_train_elements = 0
        epoch_start = time.time()

        print(f"\n{'-' * 64}", flush=True)
        print(f"Epoch {epoch}/{EPOCHS}", flush=True)
        print(f"{'-' * 64}", flush=True)

        for batch_number, start_idx in enumerate(
            range(0, len(train_indices), BATCH_SIZE), start=1
        ):
            batch_indices = train_indices[
                start_idx:start_idx + BATCH_SIZE
            ]
            batch = make_batch(batch_indices)

            optimizer.zero_grad(set_to_none=True)

            loss, _ = predict_batch(batch)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss at epoch {epoch}, "
                    f"batch {batch_number}: {loss.item()}"
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            optimizer.step()

            batch_elements = batch[1].numel()
            total_train_loss += loss.item() * batch_elements
            total_train_elements += batch_elements

            if (
                batch_number % PROGRESS_EVERY == 0
                or batch_number == num_batches
            ):
                elapsed = time.time() - epoch_start
                print(
                    f"  Train: batch {batch_number:,}/{num_batches:,} "
                    f"({100 * batch_number / num_batches:.1f}%) | "
                    f"batch NLL: {loss.item():.6f} | "
                    f"elapsed: {elapsed / 60:.1f} min",
                    flush=True,
                )

        train_nll = total_train_loss / total_train_elements
        print(f"\nTraining NLL: {train_nll:.6f}", flush=True)

        val_nll, val_mse = evaluate(val_indices)

        if val_nll < best_val_nll:
            best_val_nll = val_nll

            # Save a plain state_dict so it can be loaded using
            # model.load_state_dict(torch.load(...)).
            torch.save(model.state_dict(), CHECKPOINT_PATH)

            print(
                f"Best checkpoint saved: {CHECKPOINT_PATH}",
                flush=True,
            )

        epoch_minutes = (time.time() - epoch_start) / 60
        print(
            f"Epoch {epoch}/{EPOCHS} finished in {epoch_minutes:.1f} min | "
            f"Train NLL: {train_nll:.6f} | "
            f"Val NLL: {val_nll:.6f} | "
            f"Val MSE: {val_mse:.6f}",
            flush=True,
        )

    print("\n" + "=" * 64)
    print("WORLD MODEL TRAINING FINISHED")
    print("=" * 64)
    print(f"Best validation NLL: {best_val_nll:.6f}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

except KeyboardInterrupt:
    print(
        "\nTraining interrupted. Any checkpoint saved after a completed "
        "validation remains available.",
        flush=True,
    )
    raise

finally:
    vocab.close()
