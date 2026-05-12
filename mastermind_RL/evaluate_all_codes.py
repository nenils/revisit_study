#!/usr/bin/env python3
"""Evaluate the trained RL model against every possible secret code."""

import csv
import time
from pathlib import Path

import numpy as np
import torch

from RL import ALL_CODES, N_ACTIONS, POSITIONS, load_trained_model

RUNS_PER_CODE = 10
OUTPUT_PATH = Path("avg_guesses_by_code.csv")


def select_action(state_vec, policy_net):
    """Select the best valid action from the model for the given state."""
    with torch.no_grad():
        state_tensor = torch.from_numpy(state_vec).unsqueeze(0)
        q_values = policy_net(state_tensor)

        valid_actions = np.where(state_vec[:N_ACTIONS] > 0)[0]
        if len(valid_actions) == 0:
            return 0

        q_masked = q_values.clone()
        invalid_mask = torch.ones(N_ACTIONS, dtype=torch.bool)
        invalid_mask[valid_actions] = False
        q_masked[0, invalid_mask] = float("-inf")
        return int(q_masked.argmax().cpu().numpy())


def run_game(env, policy_net, secret_code):
    """Run one game for a fixed secret code and return steps used."""
    state = env.reset(secret=secret_code)
    steps = 0
    done = False

    while not done and steps < env.max_steps:
        action = select_action(state, policy_net)
        state, _, done, info = env.step(action)
        steps += 1

        if info["feedback"][0] == POSITIONS:
            return steps

    return steps


def main() -> None:
    policy_net, env = load_trained_model()
    policy_net.eval()

    total_codes = len(ALL_CODES)
    start_time = time.time()
    last_report = start_time

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([f"pos_{i}" for i in range(POSITIONS)] + ["avg_guesses"])

        for idx, code in enumerate(ALL_CODES, start=1):
            steps_list = [run_game(env, policy_net, code) for _ in range(RUNS_PER_CODE)]
            avg_steps = float(np.mean(steps_list))
            writer.writerow(list(code) + [f"{avg_steps:.2f}"])

            # Progress update every 50 codes or at the end
            if idx % 50 == 0 or idx == total_codes:
                now = time.time()
                elapsed = now - start_time
                rate = elapsed / idx
                remaining = total_codes - idx
                eta = remaining * rate
                print(
                    f"Progress: {idx}/{total_codes} | "
                    f"elapsed {elapsed:.1f}s | ETA {eta:.1f}s"
                )
                last_report = now

    print(
        f"Wrote average guesses for {len(ALL_CODES)} codes "
        f"({RUNS_PER_CODE} runs each) to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
