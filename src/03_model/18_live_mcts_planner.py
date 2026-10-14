"""
18_live_mcts_planner.py

Executes real-time Monte Carlo Tree Search (MCTS) with the trained PyTorch
Student Encoder and World Model.
Accepts a JSON input file with student cognitive and affective metrics:
    { "correctness": 0.48, "frustration": 0.87, "confusion": 0.61, "concentration": 0.12, "boredom": 0.05 }

Outputs valid JSON to stdout containing the exact MCTS 3-step trajectory.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SKILL_LOOKUP_FILE = ROOT / "data" / "processed" / "skill_id_to_name.json"
PLANNER_FILE = ROOT / "src" / "03_model" / "16_mcts_planner.py"
INITIAL_STATE_FILE = ROOT / "results" / "mcts" / "test_initial_state.npz"

def load_planner_module():
    spec = importlib.util.spec_from_file_location("planner_mod", str(PLANNER_FILE))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def load_skill_name_dict():
    if not SKILL_LOOKUP_FILE.exists():
        return {}
    with open(SKILL_LOOKUP_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def project_latent_state(encoder, base_state, mastery, frust, conf, conc, bored, device):
    """
    Analytically projects the 128D latent vector to produce the target
    cognitive mastery and affective metrics via the encoder's linear prediction heads.
    """
    # Clamp values safely away from 0 and 1 for logit
    mastery = float(np.clip(mastery, 0.01, 0.99))
    frust = float(np.clip(frust, 0.01, 0.99))
    conf = float(np.clip(conf, 0.01, 0.99))
    conc = float(np.clip(conc, 0.01, 0.99))
    bored = float(np.clip(bored, 0.01, 0.99))

    p_cog = torch.logit(torch.tensor([mastery], device=device))
    p_aff = torch.logit(torch.tensor([frust, conf, conc, bored], device=device))

    W_cog = encoder.correctness_head.weight.data
    b_cog = encoder.correctness_head.bias.data
    W_aff = encoder.affect_head.weight.data
    b_aff = encoder.affect_head.bias.data

    base_tensor = torch.tensor(base_state, device=device)
    z_cog_base = base_tensor[:64]
    z_aff_base = base_tensor[64:]

    # Minimum-norm delta projection
    delta_cog = p_cog - (W_cog @ z_cog_base + b_cog)
    z_cog = z_cog_base + W_cog.T @ torch.linalg.inv(W_cog @ W_cog.T) @ delta_cog

    delta_aff = p_aff - (W_aff @ z_aff_base + b_aff)
    z_aff = z_aff_base + W_aff.T @ torch.linalg.inv(W_aff @ W_aff.T) @ delta_aff

    z_state = torch.cat([z_cog, z_aff]).cpu().numpy()
    return z_state

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No input file provided"}))
        sys.exit(1)

    input_file = Path(sys.argv[1])
    if not input_file.exists():
        print(json.dumps({"error": f"Input file not found: {input_file}"}))
        sys.exit(1)

    with open(input_file, "r", encoding="utf-8") as f:
        params = json.load(f)

    correctness   = float(params.get("correctness", 0.75))
    frustration   = float(params.get("frustration", 0.35))
    confusion     = float(params.get("confusion", 0.20))
    concentration = float(params.get("concentration", 0.55))
    boredom       = float(params.get("boredom", 0.15))
    simulations   = int(params.get("simulations", 100))

    skill_names = load_skill_name_dict()

    import contextlib

    # Load real MCTS planner and neural networks
    with contextlib.redirect_stdout(sys.stderr):
        mod = load_planner_module()
        catalogue = mod.load_catalogue()
        shared_rows = mod.build_shared_candidate_set(catalogue)
        encoder, world_model = mod.load_models()
        planner = mod.Planner(encoder, world_model, catalogue, shared_rows)

        with np.load(INITIAL_STATE_FILE) as data:
            base_state = data["initial_state"]

        # Real projection to student state
        z_initial = project_latent_state(
            encoder, base_state,
            correctness, frustration, confusion, concentration, boredom,
            mod.DEVICE
        )

        initial_tensor = torch.tensor(z_initial, device=mod.DEVICE).unsqueeze(0)
        with torch.no_grad():
            initial_metrics = mod.predict_metrics(encoder, initial_tensor)[0].cpu().numpy()

        # Run genuine MCTS
        plan, root = mod.run_mcts(planner, z_initial, initial_metrics)

    # Convert plan to clean JSON
    path_steps = []
    cumulative_reward = 0.0

    for step_num, edge in enumerate(plan, 1):
        cat_row = edge["catalogue_row"]
        prob_id = int(planner.original_problem_ids[cat_row])
        skill_id = int(planner.original_skill_ids[cat_row])
        s_name = skill_names.get(str(skill_id), f"Skill {skill_id}")
        r = float(edge["reward"])
        cumulative_reward += (mod.DISCOUNT ** (step_num - 1)) * r

        next_m = edge["next_metrics"]
        next_std = edge["next_std"]

        path_steps.append({
            "step": step_num,
            "problem_id": prob_id,
            "skill_id": skill_id,
            "skill_name": s_name,
            "reward": round(r, 3),
            "predicted_correctness": round(float(next_m[0]), 3),
            "predicted_frustration": round(float(next_m[1]), 3),
            "predicted_confusion": round(float(next_m[2]), 3),
            "predicted_concentration": round(float(next_m[3]), 3),
            "predicted_boredom": round(float(next_m[4]), 3),
            "mean_latent_std": round(float(np.mean(next_std)), 3),
            "mcts_visits": int(edge.get("visits", 0)),
        })

    best_reward = round(cumulative_reward, 3)

    result = {
        "simulations": simulations,
        "horizon": len(path_steps),
        "best_reward": str(best_reward),
        "predicted_correctness": str(path_steps[0]["predicted_correctness"]),
        "predicted_frustration": str(path_steps[0]["predicted_frustration"]),
        "epistemic_uncertainty": str(path_steps[0]["mean_latent_std"]),
        "path": path_steps,
    }

    print(json.dumps(result))

if __name__ == "__main__":
    main()
