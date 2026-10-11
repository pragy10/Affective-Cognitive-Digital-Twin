
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

ENCODER_FILE = ROOT / "src" / "03_model" / "03_student_encoder.py"
WORLD_MODEL_FILE = ROOT / "src" / "03_model" / "08_world_model.py"

ENCODER_CHECKPOINT = ROOT / "models" / "student_encoder_best.pt"
WORLD_CHECKPOINT = ROOT / "models" / "world_model_best.pt"

INITIAL_STATE_FILE = ROOT / "results" / "mcts" / "test_initial_state.npz"
CATALOGUE_FILE = ROOT / "results" / "mcts" / "candidate_catalogue.csv"
REWARD_RESULTS_FILE = ROOT / "results" / "mcts" / "reward_test_results.csv"

OUTPUT_DIR = ROOT / "results" / "mcts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MCTS_OUTPUT = OUTPUT_DIR / "mcts_fair_plan.csv"
GREEDY_OUTPUT = OUTPUT_DIR / "greedy_fair_baseline.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "fair_comparison_summary.csv"
CANDIDATE_OUTPUT = OUTPUT_DIR / "fair_comparison_candidates.csv"

SEED = 42
SIMULATIONS = 100
HORIZON = 3
CANDIDATE_COUNT = 20
DISCOUNT = 0.90
EXPLORATION_CONSTANT = 1.4
BATCH_SIZE = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

METRIC_NAMES = [
    "correctness",
    "frustration",
    "confusion",
    "concentration",
    "boredom",
]


# ============================================================
# 2. LOAD MODEL CLASSES AND CHECKPOINTS
# ============================================================

def load_class(file_path, class_name, module_name):
    spec = importlib.util.spec_from_file_location(
        module_name, str(file_path)
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import model from {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return getattr(module, class_name)


StudentEncoder = load_class(
    ENCODER_FILE,
    "StudentEncoder",
    "student_encoder_module",
)

WorldModel = load_class(
    WORLD_MODEL_FILE,
    "WorldModel",
    "world_model_module",
)


def load_state_dict(path):
    checkpoint = torch.load(
        path,
        map_location=DEVICE,
        weights_only=True,
    )

    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint:
                return checkpoint[key]

    return checkpoint


def load_models():
    encoder_state = load_state_dict(ENCODER_CHECKPOINT)
    world_state = load_state_dict(WORLD_CHECKPOINT)

    num_problems = encoder_state[
        "problem_embedding.weight"
    ].shape[0]

    num_skills = encoder_state[
        "skill_embedding.weight"
    ].shape[0]

    if world_state["problem_embedding.weight"].shape[0] != num_problems:
        raise ValueError("Problem vocabulary mismatch between checkpoints.")

    if world_state["skill_embedding.weight"].shape[0] != num_skills:
        raise ValueError("Skill vocabulary mismatch between checkpoints.")

    print(f"Checkpoint problem vocabulary: {num_problems}")
    print(f"Checkpoint skill vocabulary: {num_skills}")

    encoder = StudentEncoder(
        num_problems=num_problems,
        num_skills=num_skills,
    ).to(DEVICE)

    encoder.load_state_dict(encoder_state)
    encoder.eval()

    world_model = WorldModel(
        num_problems=num_problems,
        num_skills=num_skills,
    ).to(DEVICE)

    world_model.load_state_dict(world_state)
    world_model.eval()

    return encoder, world_model


# ============================================================
# 3. LOAD CANDIDATE CATALOGUE
# ============================================================

def find_column(df, names, description):
    for name in names:
        if name in df.columns:
            return name

    raise ValueError(
        f"Cannot find {description}. "
        f"Catalogue columns: {list(df.columns)}"
    )


def load_catalogue():
    df = pd.read_csv(CATALOGUE_FILE)

    problem_col = find_column(
        df,
        [
            "problem_index",
            "problem_indices",
            "problem_idx",
            "problem_embedding_index",
            "problem_vocab_index",
        ],
        "problem embedding index",
    )

    skill_col = find_column(
        df,
        [
            "skill_index",
            "skill_indices",
            "skill_idx",
            "skill_embedding_index",
            "skill_vocab_index",
        ],
        "skill embedding index",
    )

    original_problem_col = find_column(
        df,
        ["original_problem_id", "problem_id", "problem_ids"],
        "original problem ID",
    )

    original_skill_col = find_column(
        df,
        ["original_skill_id", "skill_id", "skill_ids"],
        "original skill ID",
    )

    df = df.rename(
        columns={
            problem_col: "problem_index",
            skill_col: "skill_index",
            original_problem_col: "original_problem_id",
            original_skill_col: "original_skill_id",
        }
    )

    columns = [
        "problem_index",
        "skill_index",
        "original_problem_id",
        "original_skill_id",
    ]

    df = df[columns].copy()

    df["problem_index"] = pd.to_numeric(
        df["problem_index"], errors="raise"
    ).astype(np.int64)

    df["skill_index"] = pd.to_numeric(
        df["skill_index"], errors="raise"
    ).astype(np.int64)

    if df["problem_index"].duplicated().any():
        raise ValueError("Duplicate problem indices in catalogue.")

    if df["skill_index"].le(0).any():
        raise ValueError("Candidate catalogue contains unknown skill indices.")

    if len(df) < CANDIDATE_COUNT:
        raise ValueError(
            f"Need at least {CANDIDATE_COUNT} candidates; found {len(df)}."
        )

    return df.reset_index(drop=True)


def build_shared_candidate_set(catalogue):
    """
    Construct one fixed set of 20 candidates.

    Include the top 10 available Phase 5.4 candidates, then
    sample the remaining candidates with a reproducible seed.

    Both MCTS and greedy use this exact same set at every step.
    """
    rng = np.random.default_rng(SEED)
    selected = []

    if REWARD_RESULTS_FILE.exists():
        reward_df = pd.read_csv(REWARD_RESULTS_FILE)

        if "original_problem_id" in reward_df.columns:
            top_ids = reward_df[
                "original_problem_id"
            ].head(10).tolist()

            id_to_row = {
                str(problem_id): row
                for row, problem_id in enumerate(
                    catalogue["original_problem_id"]
                )
            }

            for problem_id in top_ids:
                row = id_to_row.get(str(problem_id))

                if row is not None and row not in selected:
                    selected.append(row)

    remaining = np.array(
        [
            row for row in range(len(catalogue))
            if row not in selected
        ],
        dtype=np.int64,
    )

    needed = CANDIDATE_COUNT - len(selected)

    if needed > 0:
        sampled = rng.choice(
            remaining,
            size=needed,
            replace=False,
        )
        selected.extend(sampled.tolist())

    selected = np.asarray(selected, dtype=np.int64)

    if len(selected) != CANDIDATE_COUNT:
        raise ValueError("Shared candidate set has an unexpected size.")

    if len(np.unique(selected)) != CANDIDATE_COUNT:
        raise ValueError("Shared candidate set contains duplicates.")

    return selected


# ============================================================
# 4. PREDICTION AND REWARD
# ============================================================

@torch.no_grad()
def predict_metrics(encoder, states):
    """
    Convert 128D latent states to:
    correctness, frustration, confusion, concentration, boredom.
    """
    z_cog = states[:, :64]
    z_aff = states[:, 64:]

    correctness = torch.sigmoid(
        encoder.correctness_head(z_cog)
    ).reshape(-1)

    affect = torch.sigmoid(
        encoder.affect_head(z_aff)
    )

    return torch.cat(
        [correctness.unsqueeze(1), affect],
        dim=1,
    )


def reward_tensor(current, next_metrics, next_std):
    """
    Preserve the Phase 5.4 reward weights.

    R = change in correctness
        + 0.15 * change in concentration
        - 0.05 * change in frustration
        - 0.05 * change in confusion
        - 0.025 * change in boredom
        - 0.05 * mean predicted latent standard deviation
    """
    delta = next_metrics - current

    return (
        delta[:, 0]
        + 0.15 * delta[:, 3]
        - 0.05 * delta[:, 1]
        - 0.05 * delta[:, 2]
        - 0.025 * delta[:, 4]
        - 0.05 * next_std.mean(dim=1)
    )


# ============================================================
# 5. PLANNER AND TRANSITION PREDICTIONS
# ============================================================

class Planner:
    def __init__(
        self,
        encoder,
        world_model,
        catalogue,
        shared_candidate_rows,
    ):
        self.encoder = encoder
        self.world_model = world_model
        self.catalogue = catalogue

        self.shared_candidate_rows = np.asarray(
            shared_candidate_rows, dtype=np.int64
        )

        self.problem_indices = catalogue[
            "problem_index"
        ].to_numpy(dtype=np.int64)

        self.skill_indices = catalogue[
            "skill_index"
        ].to_numpy(dtype=np.int64)

        self.original_problem_ids = catalogue[
            "original_problem_id"
        ].to_numpy()

        self.original_skill_ids = catalogue[
            "original_skill_id"
        ].to_numpy()

    @torch.no_grad()
    def evaluate_actions(self, state, current_metrics, action_rows):
        """
        Predict the outcome of each candidate from the supplied state.
        """
        action_rows = np.asarray(action_rows, dtype=np.int64)

        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=DEVICE,
        ).reshape(1, -1)

        current_tensor = torch.as_tensor(
            current_metrics,
            dtype=torch.float32,
            device=DEVICE,
        ).reshape(1, -1)

        outcomes = []

        for start in range(0, len(action_rows), BATCH_SIZE):
            batch_rows = action_rows[start:start + BATCH_SIZE]
            batch_size = len(batch_rows)

            problems = torch.as_tensor(
                self.problem_indices[batch_rows],
                dtype=torch.long,
                device=DEVICE,
            )

            skills = torch.as_tensor(
                self.skill_indices[batch_rows],
                dtype=torch.long,
                device=DEVICE,
            )

            # All candidates in this catalogue have known skill mappings.
            missing = torch.zeros_like(skills)

            states = state_tensor.expand(batch_size, -1)
            current = current_tensor.expand(batch_size, -1)

            prediction = self.world_model(
                states,
                problems,
                skills,
                missing,
            )

            next_states = prediction["next_state_mean"]
            next_std = prediction["next_state_std"]

            next_metrics = predict_metrics(
                self.encoder,
                next_states,
            )

            rewards = reward_tensor(
                current.expand(batch_size, -1),
                next_metrics,
                next_std,
            )

            if not (
                torch.isfinite(next_states).all()
                and torch.isfinite(next_std).all()
                and torch.isfinite(next_metrics).all()
                and torch.isfinite(rewards).all()
            ):
                raise ValueError(
                    "Non-finite prediction or reward detected."
                )

            for i, row in enumerate(batch_rows):
                outcomes.append(
                    {
                        "catalogue_row": int(row),
                        "next_state": (
                            next_states[i].cpu().numpy()
                        ),
                        "next_metrics": (
                            next_metrics[i].cpu().numpy()
                        ),
                        "next_std": next_std[i].cpu().numpy(),
                        "reward": float(rewards[i].item()),
                    }
                )

        return outcomes

    def available_actions(self, used_rows):
        used = set(used_rows)

        return np.asarray(
            [
                row for row in self.shared_candidate_rows
                if int(row) not in used
            ],
            dtype=np.int64,
        )


# ============================================================
# 6. MCTS TREE STRUCTURE
# ============================================================

class Node:
    def __init__(self, state, metrics, depth, used_rows):
        self.state = np.asarray(state, dtype=np.float32)
        self.metrics = np.asarray(metrics, dtype=np.float32)
        self.depth = depth
        self.used_rows = frozenset(used_rows)

        self.visits = 0
        self.children = None


def expand_node(node, planner):
    if node.children is not None:
        return

    action_rows = planner.available_actions(node.used_rows)

    if len(action_rows) == 0:
        node.children = []
        return

    outcomes = planner.evaluate_actions(
        node.state,
        node.metrics,
        action_rows,
    )

    node.children = []

    for outcome in outcomes:
        row = outcome["catalogue_row"]

        child_used = set(node.used_rows)
        child_used.add(row)

        outcome["visits"] = 0
        outcome["value_sum"] = 0.0

        outcome["child_node"] = Node(
            outcome["next_state"],
            outcome["next_metrics"],
            node.depth + 1,
            child_used,
        )

        node.children.append(outcome)


def select_child(node, rng):
    unvisited = [
        edge for edge in node.children
        if edge["visits"] == 0
    ]

    if unvisited:
        return unvisited[int(rng.integers(len(unvisited)))]

    log_parent = math.log(max(1, node.visits))

    def uct(edge):
        mean_value = edge["value_sum"] / edge["visits"]

        exploration = EXPLORATION_CONSTANT * math.sqrt(
            log_parent / edge["visits"]
        )

        return mean_value + exploration

    return max(node.children, key=uct)


# ============================================================
# 7. RUN MONTE CARLO TREE SEARCH
# ============================================================

def run_mcts(planner, initial_state, initial_metrics):
    rng = np.random.default_rng(SEED)

    root = Node(
        initial_state,
        initial_metrics,
        depth=0,
        used_rows=set(),
    )

    for simulation in range(SIMULATIONS):
        node = root
        path = []

        while node.depth < HORIZON:
            node.visits += 1

            if node.children is None:
                expand_node(node, planner)

            if not node.children:
                break

            edge = select_child(node, rng)
            path.append(edge)
            node = edge["child_node"]

        # Discount rewards backward from the end of the simulated path.
        return_value = 0.0

        for edge in reversed(path):
            return_value = (
                edge["reward"] + DISCOUNT * return_value
            )

            edge["visits"] += 1
            edge["value_sum"] += return_value

        if (simulation + 1) % 20 == 0:
            print(
                f"MCTS simulations: {simulation + 1}/{SIMULATIONS}"
            )

    # Reconstruct the plan by following the most-visited edge
    # at each node. The tree's used_rows prevents repetition.
    plan = []
    node = root

    while node.depth < HORIZON:
        if node.children is None:
            expand_node(node, planner)

        if not node.children:
            raise RuntimeError("MCTS reached a node without actions.")

        visited = [
            edge for edge in node.children
            if edge["visits"] > 0
        ]

        if visited:
            edge = max(
                visited,
                key=lambda item: (
                    item["visits"],
                    item["value_sum"] / item["visits"],
                ),
            )
        else:
            edge = max(
                node.children,
                key=lambda item: item["reward"],
            )

        plan.append(edge)
        node = edge["child_node"]

    validate_no_repeated_problems(plan, "MCTS")

    return plan, root


# ============================================================
# 8. GREEDY BASELINE
# ============================================================

def run_greedy(planner, initial_state, initial_metrics):
    """
    Greedy chooses the best immediate reward at each step,
    using the same candidate set and no-repeat rule as MCTS.
    """
    state = np.asarray(initial_state, dtype=np.float32)
    metrics = np.asarray(initial_metrics, dtype=np.float32)

    used_rows = set()
    plan = []

    for depth in range(HORIZON):
        print(
            f"Greedy baseline: evaluating step "
            f"{depth + 1}/{HORIZON}"
        )

        available = planner.available_actions(used_rows)

        outcomes = planner.evaluate_actions(
            state,
            metrics,
            available,
        )

        if not outcomes:
            raise RuntimeError("Greedy baseline ran out of actions.")

        best = max(outcomes, key=lambda item: item["reward"])
        plan.append(best)

        used_rows.add(best["catalogue_row"])

        state = best["next_state"]
        metrics = best["next_metrics"]

    validate_no_repeated_problems(plan, "Greedy")

    return plan


# ============================================================
# 9. VALIDATION AND OUTPUT
# ============================================================

def validate_no_repeated_problems(plan, method_name):
    rows = [edge["catalogue_row"] for edge in plan]

    if len(rows) != HORIZON:
        raise ValueError(
            f"{method_name} returned {len(rows)} steps; "
            f"expected {HORIZON}."
        )

    if len(set(rows)) != len(rows):
        raise ValueError(
            f"{method_name} repeated a problem in its plan."
        )

    for edge in plan:
        if not np.isfinite(edge["reward"]):
            raise ValueError(f"{method_name} produced a non-finite reward.")

        if not np.isfinite(edge["next_state"]).all():
            raise ValueError(f"{method_name} produced a non-finite state.")


def plan_to_dataframe(plan, planner, method):
    rows = []
    cumulative_reward = 0.0

    for step, edge in enumerate(plan, start=1):
        catalogue_row = edge["catalogue_row"]
        reward = float(edge["reward"])

        cumulative_reward += (
            DISCOUNT ** (step - 1)
        ) * reward

        rows.append(
            {
                "method": method,
                "step": step,
                "original_problem_id":
                    planner.original_problem_ids[catalogue_row],
                "original_skill_id":
                    planner.original_skill_ids[catalogue_row],
                "predicted_correctness":
                    float(edge["next_metrics"][0]),
                "predicted_frustration":
                    float(edge["next_metrics"][1]),
                "predicted_confusion":
                    float(edge["next_metrics"][2]),
                "predicted_concentration":
                    float(edge["next_metrics"][3]),
                "predicted_boredom":
                    float(edge["next_metrics"][4]),
                "mean_latent_std":
                    float(np.mean(edge["next_std"])),
                "immediate_reward": reward,
                "discounted_cumulative_reward":
                    cumulative_reward,
                "mcts_visits": edge.get("visits", 0),
            }
        )

    return pd.DataFrame(rows)


def main():
    print(f"Device: {DEVICE}")
    print("Loading candidate catalogue...")

    catalogue = load_catalogue()
    print(f"Full catalogue: {len(catalogue)} candidates")

    shared_rows = build_shared_candidate_set(catalogue)

    shared_catalogue = catalogue.iloc[shared_rows].copy()
    shared_catalogue.to_csv(CANDIDATE_OUTPUT, index=False)

    print(f"Shared comparison candidates: {len(shared_rows)}")
    print("Both planners will use the same fixed candidate set.")

    print("\nLoading initial student state...")
    with np.load(INITIAL_STATE_FILE, allow_pickle=False) as data:
        initial_state = data["initial_state"].astype(np.float32)

    if initial_state.shape != (128,):
        raise ValueError(
            f"Expected initial state shape (128,), got "
            f"{initial_state.shape}."
        )

    encoder, world_model = load_models()
    print("Both trained models loaded.")

    planner = Planner(
        encoder,
        world_model,
        catalogue,
        shared_rows,
    )

    initial_tensor = torch.as_tensor(
        initial_state,
        dtype=torch.float32,
        device=DEVICE,
    ).reshape(1, -1)

    with torch.no_grad():
        initial_metrics = predict_metrics(
            encoder, initial_tensor
        )[0].cpu().numpy()

    print("\nStarting predicted metrics:")
    for name, value in zip(METRIC_NAMES, initial_metrics):
        print(f"  {name}: {value:.4f}")

    print("\nRunning MCTS...")
    mcts_plan, root = run_mcts(
        planner,
        initial_state,
        initial_metrics,
    )

    print("\nRunning greedy baseline...")
    greedy_plan = run_greedy(
        planner,
        initial_state,
        initial_metrics,
    )

    mcts_df = plan_to_dataframe(
        mcts_plan, planner, "MCTS"
    )

    greedy_df = plan_to_dataframe(
        greedy_plan, planner, "Greedy"
    )

    mcts_df.to_csv(MCTS_OUTPUT, index=False)
    greedy_df.to_csv(GREEDY_OUTPUT, index=False)

    mcts_total = float(
        mcts_df.iloc[-1]["discounted_cumulative_reward"]
    )

    greedy_total = float(
        greedy_df.iloc[-1]["discounted_cumulative_reward"]
    )

    summary = pd.DataFrame(
        [
            {
                "method": "MCTS",
                "candidate_count": len(shared_rows),
                "horizon": HORIZON,
                "discount": DISCOUNT,
                "simulations": SIMULATIONS,
                "discounted_cumulative_reward": mcts_total,
                "root_visits": root.visits,
                "unique_problems": mcts_df[
                    "original_problem_id"
                ].nunique(),
            },
            {
                "method": "Greedy",
                "candidate_count": len(shared_rows),
                "horizon": HORIZON,
                "discount": DISCOUNT,
                "simulations": 0,
                "discounted_cumulative_reward": greedy_total,
                "root_visits": 0,
                "unique_problems": greedy_df[
                    "original_problem_id"
                ].nunique(),
            },
        ]
    )

    summary.to_csv(SUMMARY_OUTPUT, index=False)

    print("\n========== SHARED CANDIDATE SET ==========")
    print(
        shared_catalogue[
            ["original_problem_id", "original_skill_id"]
        ].to_string(index=False)
    )

    print("\n========== MCTS PLAN ==========")
    print(
        mcts_df[
            [
                "step",
                "original_problem_id",
                "original_skill_id",
                "predicted_correctness",
                "immediate_reward",
                "discounted_cumulative_reward",
                "mcts_visits",
            ]
        ].to_string(index=False)
    )

    print("\n========== GREEDY BASELINE ==========")
    print(
        greedy_df[
            [
                "step",
                "original_problem_id",
                "original_skill_id",
                "predicted_correctness",
                "immediate_reward",
                "discounted_cumulative_reward",
            ]
        ].to_string(index=False)
    )

    print("\n========== FAIR COMPARISON ==========")
    print(summary.to_string(index=False))

    print("\nValidation:")
    print(f"  MCTS unique problems: {mcts_df['original_problem_id'].nunique()}/{HORIZON}")
    print(f"  Greedy unique problems: {greedy_df['original_problem_id'].nunique()}/{HORIZON}")
    print(f"  Shared candidate count: {len(shared_rows)}")
    print("  All predicted rewards and states: finite")
    print("  Both plans use the same candidate set: yes")

    print("\nSaved files:")
    print(f"  {MCTS_OUTPUT}")
    print(f"  {GREEDY_OUTPUT}")
    print(f"  {SUMMARY_OUTPUT}")
    print(f"  {CANDIDATE_OUTPUT}")

    print(
        "\nNote: these are exploratory model-based estimates. "
        "The comparison is fairer in candidate access, but 100 "
        "simulations and a 20-problem subset do not establish that "
        "MCTS is generally better than greedy or improves real "
        "student learning."
    )


if __name__ == "__main__":
    main()
