#!/usr/bin/env python3
"""Evaluate the trained model with color-restricted codes and save results."""

import csv
import time
from pathlib import Path

import numpy as np
import torch

from RL import ALL_CODES, N_ACTIONS, POSITIONS, load_trained_model

RUNS_PER_CODE = 3


def build_allowed_indices(num_colors: int):
    """Return action indices whose codes use only the first num_colors."""
    allowed_colors = set(range(num_colors))
    return [
        idx
        for idx, code in enumerate(ALL_CODES)
        if all(color in allowed_colors for color in code)
    ]


def select_action(state_vec, policy_net, allowed_indices):
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


def run_game(env, policy_net, secret_code, allowed_indices):
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


def evaluate_for_colors(policy_net, env, num_colors: int):
    allowed_indices = build_allowed_indices(num_colors)
    allowed_codes = [ALL_CODES[idx] for idx in allowed_indices]

    total_codes = len(allowed_codes)
    output_path = Path(f"avg_guesses_by_code_{num_colors}.csv")

    start_time = time.time()

    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([f"pos_{i}" for i in range(POSITIONS)] + ["avg_guesses"])

        for idx, code in enumerate(allowed_codes, start=1):
            steps_list = [run_game(env, policy_net, code, allowed_indices) for _ in range(RUNS_PER_CODE)]
            avg_steps = float(np.mean(steps_list))
            writer.writerow(list(code) + [f"{avg_steps:.2f}"])

            if idx % 50 == 0 or idx == total_codes:
                elapsed = time.time() - start_time
                rate = elapsed / idx
                remaining = total_codes - idx
                eta = remaining * rate
                print(
                    f"colors={num_colors} | {idx}/{total_codes} | "
                    f"elapsed {elapsed:.1f}s | ETA {eta:.1f}s"
                )

    print(f"Saved {output_path} ({total_codes} codes, {RUNS_PER_CODE} runs each)")


def main() -> None:
    policy_net, env = load_trained_model()
    policy_net.eval()

    for num_colors in range(1, 7):
        evaluate_for_colors(policy_net, env, num_colors)


if __name__ == "__main__":
    main()
