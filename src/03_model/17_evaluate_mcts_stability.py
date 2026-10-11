
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

PLANNER_FILE = ROOT / "src" / "03_model" / "16_mcts_planner.py"
INITIAL_STATE_FILE = ROOT / "results" / "mcts" / "test_initial_state.npz"

OUTPUT_DIR = ROOT / "results" / "mcts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS_FILE = OUTPUT_DIR / "mcts_stability_runs.csv"
SUMMARY_FILE = OUTPUT_DIR / "mcts_stability_summary.csv"
SEQUENCES_FILE = OUTPUT_DIR / "mcts_stability_sequences.csv"
GREEDY_FILE = OUTPUT_DIR / "mcts_stability_greedy.csv"

# Three search budgets, tested across three random seeds.
SIMULATION_BUDGETS = [100, 300, 500]
RANDOM_SEEDS = [11, 42, 123]

# Keep the candidate set identical across all experiments.
CANDIDATE_SET_SEED = 42


# ============================================================
# 2. IMPORT THE EXISTING PLANNER
# ============================================================

def import_planner_module():
    spec = importlib.util.spec_from_file_location(
        "existing_mcts_planner",
        str(PLANNER_FILE),
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot import planner from {PLANNER_FILE}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def calculate_discounted_reward(plan, discount):
    return float(
        sum(
            (discount ** step) * float(edge["reward"])
            for step, edge in enumerate(plan)
        )
    )


def extract_problem_sequence(plan, planner):
    return [
        planner.original_problem_ids[edge["catalogue_row"]]
        for edge in plan
    ]


def extract_skill_sequence(plan, planner):
    return [
        planner.original_skill_ids[edge["catalogue_row"]]
        for edge in plan
    ]


def validate_plan(plan, horizon):
    if len(plan) != horizon:
        raise ValueError(
            f"Expected {horizon} steps, received {len(plan)}."
        )

    problem_rows = [
        edge["catalogue_row"] for edge in plan
    ]

    if len(problem_rows) != len(set(problem_rows)):
        raise ValueError("A plan contains repeated problem selections.")

    for edge in plan:
        if not np.isfinite(edge["reward"]):
            raise ValueError("Non-finite reward in plan.")

        if not np.isfinite(edge["next_state"]).all():
            raise ValueError("Non-finite predicted state in plan.")


# ============================================================
# 4. MAIN EVALUATION
# ============================================================

def main():
    print("========== MCTS STABILITY EVALUATION ==========")

    planner_module = import_planner_module()

    device = planner_module.DEVICE
    print(f"Device: {device}")

    # Build the candidate set exactly once.
    planner_module.SEED = CANDIDATE_SET_SEED

    print("\nLoading candidate catalogue...")
    catalogue = planner_module.load_catalogue()

    shared_rows = planner_module.build_shared_candidate_set(
        catalogue
    )

    shared_catalogue = catalogue.iloc[shared_rows].copy()

    print(f"Full catalogue: {len(catalogue)}")
    print(f"Fixed candidate set: {len(shared_rows)}")
    print(f"Simulation budgets: {SIMULATION_BUDGETS}")
    print(f"Random seeds: {RANDOM_SEEDS}")

    # Load the same initial student state for every run.
    with np.load(
        INITIAL_STATE_FILE,
        allow_pickle=False,
    ) as data:
        initial_state = data["initial_state"].astype(np.float32)

    if initial_state.shape != (128,):
        raise ValueError(
            f"Expected a 128D initial state; got {initial_state.shape}"
        )

    print("\nLoading trained models...")
    encoder, world_model = planner_module.load_models()

    planner = planner_module.Planner(
        encoder,
        world_model,
        catalogue,
        shared_rows,
    )

    initial_tensor = torch.as_tensor(
        initial_state,
        dtype=torch.float32,
        device=device,
    ).reshape(1, -1)

    with torch.no_grad():
        initial_metrics = planner_module.predict_metrics(
            encoder,
            initial_tensor,
        )[0].cpu().numpy()

    print("Models loaded. Initial student state prepared.")

    # --------------------------------------------------------
    # Greedy baseline: calculate once on the same candidate set.
    # --------------------------------------------------------

    print("\n========== GREEDY BASELINE ==========")

    greedy_plan = planner_module.run_greedy(
        planner,
        initial_state,
        initial_metrics,
    )

    validate_plan(
        greedy_plan,
        planner_module.HORIZON,
    )

    greedy_reward = calculate_discounted_reward(
        greedy_plan,
        planner_module.DISCOUNT,
    )

    greedy_problems = extract_problem_sequence(
        greedy_plan,
        planner,
    )

    greedy_skills = extract_skill_sequence(
        greedy_plan,
        planner,
    )

    greedy_df = planner_module.plan_to_dataframe(
        greedy_plan,
        planner,
        "Greedy",
    )

    greedy_df.to_csv(GREEDY_FILE, index=False)

    print(f"Greedy discounted reward: {greedy_reward:.6f}")
    print(f"Greedy problem sequence: {greedy_problems}")

    # --------------------------------------------------------
    # Run MCTS repeatedly across budgets and random seeds.
    # --------------------------------------------------------

    run_rows = []
    sequence_rows = []

    total_runs = (
        len(SIMULATION_BUDGETS) * len(RANDOM_SEEDS)
    )
    run_number = 0

    for budget in SIMULATION_BUDGETS:
        for seed in RANDOM_SEEDS:
            run_number += 1

            print(
                f"\n========== RUN {run_number}/{total_runs} "
                f"| budget={budget} | seed={seed} =========="
            )

            # The existing planner reads these module-level values.
            planner_module.SIMULATIONS = budget
            planner_module.SEED = seed

            plan, root = planner_module.run_mcts(
                planner,
                initial_state,
                initial_metrics,
            )

            validate_plan(
                plan,
                planner_module.HORIZON,
            )

            reward = calculate_discounted_reward(
                plan,
                planner_module.DISCOUNT,
            )

            problems = extract_problem_sequence(
                plan,
                planner,
            )

            skills = extract_skill_sequence(
                plan,
                planner,
            )

            sequence_key = " -> ".join(
                str(problem) for problem in problems
            )

            row = {
                "simulations": budget,
                "seed": seed,
                "discounted_cumulative_reward": reward,
                "greedy_reward": greedy_reward,
                "reward_difference_vs_greedy": (
                    reward - greedy_reward
                ),
                "root_visits": root.visits,
                "unique_problems": len(set(problems)),
                "problem_sequence": sequence_key,
                "first_problem_id": problems[0],
                "first_skill_id": skills[0],
            }

            run_rows.append(row)

            for step, edge in enumerate(plan, start=1):
                catalogue_row = edge["catalogue_row"]

                sequence_rows.append(
                    {
                        "simulations": budget,
                        "seed": seed,
                        "step": step,
                        "original_problem_id":
                            planner.original_problem_ids[catalogue_row],
                        "original_skill_id":
                            planner.original_skill_ids[catalogue_row],
                        "immediate_reward": float(edge["reward"]),
                        "mean_latent_std": float(
                            np.mean(edge["next_std"])
                        ),
                    }
                )

            print(f"Discounted reward: {reward:.6f}")
            print(f"Greedy reward:     {greedy_reward:.6f}")
            print(f"Difference:        {reward - greedy_reward:+.6f}")
            print(f"Problem sequence:  {sequence_key}")
            print(f"Root visits:       {root.visits}")
            print("Plan validation:   passed")

    # --------------------------------------------------------
    # Save raw run-level results.
    # --------------------------------------------------------

    runs_df = pd.DataFrame(run_rows)
    sequences_df = pd.DataFrame(sequence_rows)

    runs_df.to_csv(RUNS_FILE, index=False)
    sequences_df.to_csv(SEQUENCES_FILE, index=False)

    # --------------------------------------------------------
    # Summarize reward stability by simulation budget.
    # --------------------------------------------------------

    summary_rows = []

    for budget in SIMULATION_BUDGETS:
        subset = runs_df[
            runs_df["simulations"] == budget
        ]

        rewards = subset[
            "discounted_cumulative_reward"
        ].to_numpy(dtype=float)

        sequence_counts = subset[
            "problem_sequence"
        ].nunique()

        summary_rows.append(
            {
                "simulations": budget,
                "number_of_runs": len(subset),
                "mean_reward": float(np.mean(rewards)),
                "std_reward": float(np.std(rewards, ddof=1)),
                "minimum_reward": float(np.min(rewards)),
                "maximum_reward": float(np.max(rewards)),
                "unique_sequences": int(sequence_counts),
                "greedy_reward": greedy_reward,
                "mean_difference_vs_greedy": float(
                    np.mean(rewards - greedy_reward)
                ),
                "runs_beating_greedy": int(
                    np.sum(rewards > greedy_reward)
                ),
                "first_problem_frequency": int(
                    subset["first_problem_id"].value_counts().max()
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_FILE, index=False)

    # --------------------------------------------------------
    # Final report.
    # --------------------------------------------------------

    print("\n\n========== RUN-LEVEL RESULTS ==========")
    print(
        runs_df[
            [
                "simulations",
                "seed",
                "discounted_cumulative_reward",
                "reward_difference_vs_greedy",
                "problem_sequence",
            ]
        ].to_string(index=False)
    )

    print("\n========== STABILITY SUMMARY ==========")
    print(summary_df.to_string(index=False))

    print("\n========== OUTPUT FILES ==========")
    print(f"Run-level results: {RUNS_FILE}")
    print(f"Stability summary: {SUMMARY_FILE}")
    print(f"Selected sequences: {SEQUENCES_FILE}")
    print(f"Greedy baseline:    {GREEDY_FILE}")

    print("\nAll completed plans passed length, uniqueness, and "
          "finite-value checks.")

    print(
        "\nInterpretation: this evaluates repeatability and reward "
        "variation for one fixed student state and one fixed candidate "
        "set. It does not establish real-world learning improvement "
        "or prove that a higher simulation budget is always better."
    )


if __name__ == "__main__":
    main()
