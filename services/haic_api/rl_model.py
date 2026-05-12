from __future__ import annotations

import json
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any


COLOR_ORDER = ["Blue", "Green", "Red", "Yellow", "Orange", "Purple"]
POSITIONS = 4
ALL_CODES = list(product(range(len(COLOR_ORDER)), repeat=POSITIONS))
N_ACTIONS = len(ALL_CODES)


class MastermindRLModel:
    """Adapter for the BillXan/Mastermind_RL DQN checkpoint.

    The trained checkpoint is expected at services/haic_api/models/mastermind_dqn_model.pth
    unless RL_MODEL_PATH overrides it. If loading or inference fails, the adapter
    falls back to a deterministic consistency filter so the study remains usable.
    """

    def __init__(self, model_path: str | None = None):
        default_path = Path(__file__).parent / "models" / "mastermind_dqn_model.pth"
        self.model_path = Path(model_path).expanduser() if model_path else default_path
        self.model: Any = None
        self.load_error: str | None = None
        self.training_info: dict[str, Any] | None = None

        try:
            self.model = self._load_model(self.model_path)
        except Exception as exc:
            self.load_error = str(exc)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, state: Any) -> dict[str, Any]:
        if self.model is not None:
            try:
                guess = self._predict_with_dqn(state)
                return {
                    "guess": guess,
                    "model": "BillXan/Mastermind_RL DQN",
                    "usingFallback": False,
                    "modelPath": str(self.model_path),
                    "trainingInfo": self.training_info,
                }
            except Exception as exc:
                self.load_error = str(exc)

        return {
            "guess": self._dummy_consistency_filter(state),
            "model": "dummy-consistency-filter",
            "usingFallback": True,
            "modelPath": str(self.model_path),
            "loadError": self.load_error,
            "note": "Using fallback because the BillXan/Mastermind_RL checkpoint could not be loaded or queried.",
        }

    def _load_model(self, model_path: Path) -> Any:
        if not model_path.exists():
            raise FileNotFoundError(f"RL model artifact not found: {model_path}")

        if model_path.suffix.lower() == ".json":
            return json.loads(model_path.read_text(encoding="utf-8"))

        if model_path.suffix.lower() not in {".pt", ".pth"}:
            raise ValueError("Unsupported RL model artifact. Use .json, .pt, or .pth.")

        import torch
        import torch.nn as nn

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "policy_net_state_dict" not in checkpoint:
            raise ValueError("Expected a BillXan/Mastermind_RL checkpoint with policy_net_state_dict.")

        hyperparams = checkpoint.get("hyperparameters", {})
        input_dim = int(hyperparams.get("input_dim", N_ACTIONS + 4))
        output_dim = int(hyperparams.get("output_dim", N_ACTIONS))
        hidden = int(hyperparams.get("hidden", 512))

        class DQN(nn.Module):
            def __init__(self, input_dim: int, output_dim: int, hidden: int = 512):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden),
                    nn.LayerNorm(hidden),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden, hidden // 2),
                    nn.LayerNorm(hidden // 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden // 2, hidden // 4),
                    nn.ReLU(),
                    nn.Linear(hidden // 4, output_dim),
                )

            def forward(self, x: Any) -> Any:
                return self.net(x)

        model = DQN(input_dim=input_dim, output_dim=output_dim, hidden=hidden)
        model.load_state_dict(checkpoint["policy_net_state_dict"])
        model.eval()
        self.training_info = checkpoint.get("training_info")
        return model

    def _predict_with_dqn(self, state: Any) -> list[str]:
        import numpy as np
        import torch

        state_dict = _state_to_dict(state)
        state_vec = _state_vector(state_dict)
        valid_actions = np.where(state_vec[:N_ACTIONS] > 0)[0]
        if len(valid_actions) == 0:
            return self._dummy_consistency_filter(state_dict)

        with torch.no_grad():
            state_tensor = torch.from_numpy(state_vec).unsqueeze(0)
            q_values = self.model(state_tensor)
            q_masked = q_values.clone()
            invalid_mask = torch.ones(N_ACTIONS, dtype=torch.bool)
            invalid_mask[valid_actions] = False
            q_masked[0, invalid_mask] = float("-inf")
            action = int(q_masked.argmax().cpu().numpy())

        return _code_to_colors(ALL_CODES[action], state_dict["availableColors"])

    def _dummy_consistency_filter(self, state: Any) -> list[str]:
        state_dict = _state_to_dict(state)
        colors = state_dict["availableColors"] or ["Blue"]
        code_length = state_dict["codeLength"] or POSITIONS
        candidates = [list(candidate) for candidate in product(colors, repeat=code_length)]

        for attempt, guess in state_dict["guessHistory"].items():
            feedback = state_dict["feedbackHistory"].get(str(attempt))
            if not feedback:
                continue
            expected = {
                "black": int(feedback.get("black", 0)),
                "white": int(feedback.get("white", 0)),
            }
            candidates = [
                candidate
                for candidate in candidates
                if score_feedback(candidate, guess) == expected
            ]

        if not candidates:
            attempt = int(state_dict.get("attempt", 1))
            return [colors[(idx + attempt - 1) % len(colors)] for idx in range(code_length)]

        return max(candidates, key=lambda guess: (len(set(guess)), tuple(colors.index(c) for c in guess)))


def score_feedback(secret: list[str], guess: list[str]) -> dict[str, int]:
    black = sum(secret_color == guess_color for secret_color, guess_color in zip(secret, guess))
    secret_counts = Counter(secret)
    guess_counts = Counter(guess)
    common = sum(min(secret_counts[color], guess_counts[color]) for color in secret_counts)
    return {"black": black, "white": common - black}


def _state_to_dict(state: Any) -> dict[str, Any]:
    if hasattr(state, "model_dump"):
        state = state.model_dump()
    elif not isinstance(state, dict):
        state = dict(state)

    available_colors = [
        color for color in list(state.get("availableColors") or ["Blue"])
        if color in COLOR_ORDER
    ] or ["Blue"]

    return {
        "round": int(state.get("round", 1)),
        "codeLength": int(state.get("codeLength", POSITIONS)),
        "availableColors": available_colors,
        "maxAttemptsPerRound": int(state.get("maxAttemptsPerRound", 10)),
        "attempt": int(state.get("attempt", 1)),
        "guessHistory": {str(k): list(v) for k, v in dict(state.get("guessHistory") or {}).items()},
        "feedbackHistory": {str(k): dict(v) for k, v in dict(state.get("feedbackHistory") or {}).items()},
    }


def _state_vector(state: dict[str, Any]) -> Any:
    import numpy as np

    possible_vec = np.zeros(N_ACTIONS, dtype=np.float32)
    history = _numeric_history(state)
    available_indices = {COLOR_ORDER.index(color) for color in state["availableColors"]}

    possible_codes = []
    for idx, code in enumerate(ALL_CODES):
        if not all(color in available_indices for color in code):
            continue
        if consistent_with_history(code, history):
            possible_vec[idx] = 1.0
            possible_codes.append(code)

    last_black, last_white = (0, 0) if not history else history[-1][1]
    additional_features = np.array(
        [
            len(possible_codes) / N_ACTIONS,
            len(history) / max(1, state["maxAttemptsPerRound"]),
            last_black / POSITIONS,
            last_white / POSITIONS,
        ],
        dtype=np.float32,
    )
    return np.concatenate([possible_vec, additional_features])


def _numeric_history(state: dict[str, Any]) -> list[tuple[tuple[int, ...], tuple[int, int]]]:
    history: list[tuple[tuple[int, ...], tuple[int, int]]] = []
    for attempt in sorted(state["guessHistory"], key=lambda item: int(item)):
        guess = state["guessHistory"][attempt]
        feedback = state["feedbackHistory"].get(str(attempt))
        if not feedback:
            continue
        try:
            numeric_guess = tuple(COLOR_ORDER.index(color) for color in guess)
        except ValueError:
            continue
        history.append((numeric_guess, (int(feedback.get("black", 0)), int(feedback.get("white", 0)))))
    return history


def consistent_with_history(code: tuple[int, ...], history: list[tuple[tuple[int, ...], tuple[int, int]]]) -> bool:
    for guess, expected_feedback in history:
        if numeric_feedback(code, guess) != expected_feedback:
            return False
    return True


def numeric_feedback(code: tuple[int, ...], guess: tuple[int, ...]) -> tuple[int, int]:
    black = sum(code_color == guess_color for code_color, guess_color in zip(code, guess))
    code_counts = Counter(code)
    guess_counts = Counter(guess)
    common = sum(min(code_counts[color], guess_counts[color]) for color in code_counts)
    return black, common - black


def _code_to_colors(code: tuple[int, ...], available_colors: list[str]) -> list[str]:
    colors = [COLOR_ORDER[color] for color in code]
    if not all(color in available_colors for color in colors):
        raise ValueError(f"RL model selected unavailable color in guess: {colors}")
    return colors
