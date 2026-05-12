#!/usr/bin/env python3
"""Evaluate the trained model with subsets of available colors."""

import csv
import time
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch

from RL import ALL_CODES, N_ACTIONS, POSITIONS, load_trained_model

RUNS_PER_CODE = 10
OUTPUT_PATH = Path("avg_steps_by_color_count.csv")


def build_allowed_indices(allowed_colors: Sequence[int]) -> List[int]:
    """Return action indices whose codes use only allowed colors."""
    allowed_set = set(allowed_colors)
    return [
        idx
        for idx, code in enumerate(ALL_CODES)
        if all(color in allowed_set for color in code)
    ]


def select_action(state_vec, policy_net, allowed_indices: Sequence[int]) -> int:
    """Select the best valid action from the model for the given state."""
    with torch.no_grad():
        state_tensor = torch.from_numpy(state_vec).unsqueeze(0)
        q_values = policy_net(state_tensor)

        valid_actions = np.where(state_vec[:N_ACTIONS] > 0)[0]
        if allowed_indices:
            valid_actions = np.intersect1d(valid_actions, allowed_indices, assume_unique=False)

        if len(valid_actions) == 0:
            return 0

        q_masked = q_values.clone()
        invalid_mask = torch.ones(N_ACTIONS, dtype=torch.bool)
        invalid_mask[valid_actions] = False
        q_masked[0, invalid_mask] = float("-inf")
        return int(q_masked.argmax().cpu().numpy())


def run_game(env, policy_net, secret_code, allowed_indices: Sequence[int]) -> int:
    """Run one game for a fixed secret code and return steps used."""
    state = env.reset(secret=secret_code)
    steps = 0
    done = False

    while not done and steps < env.max_steps:
        action = select_action(state, policy_net, allowed_indices)
        state, _, done, info = env.step(action)
        steps += 1

        if info["feedback"][0] == POSITIONS:
            return steps

    return steps


def evaluate_for_colors(
    policy_net,
    env,
    allowed_colors: Sequence[int],
    runs_per_code: int,
) -> float:
    allowed_indices = build_allowed_indices(allowed_colors)
    allowed_codes = [ALL_CODES[idx] for idx in allowed_indices]

    steps_list = []
    for code in allowed_codes:
        for _ in range(runs_per_code):
            steps_list.append(run_game(env, policy_net, code, allowed_indices))

    return float(np.mean(steps_list)) if steps_list else float("inf")


def main() -> None:
    policy_net, env = load_trained_model()
    policy_net.eval()

    print("Evaluating with action masking (first k colors allowed)")
    print(f"Runs per code: {RUNS_PER_CODE}")
    print("-")

    start_time = time.time()
    results = []

    for k in range(1, 7):
        allowed_colors = list(range(k))
        avg_steps = evaluate_for_colors(policy_net, env, allowed_colors, RUNS_PER_CODE)
        results.append((k, avg_steps))
        print(f"k={k} colors -> avg steps: {avg_steps:.3f}")

    elapsed = time.time() - start_time
    print("-")
    print(f"Done in {elapsed:.1f}s")

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["num_colors", "avg_steps"])
        for num_colors, avg_steps in results:
            writer.writerow([num_colors, f"{avg_steps:.6f}"])

    print(f"Saved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
