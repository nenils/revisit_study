from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any


class MastermindRLModel:
    """Adapter for the trained Mastermind RL model used by the Advisor condition.

    Put the trained artifact in services/haic_api/models/ and set RL_MODEL_PATH.
    If no artifact is configured, the adapter uses a deterministic dummy policy
    that keeps only guesses consistent with previous Mastermind feedback.
    """

    def __init__(self, model_path: str | None = None):
        self.model_path = Path(model_path).expanduser() if model_path else None
        self.model: Any = None
        self.load_error: str | None = None

        if self.model_path:
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
                guess = self._predict_with_loaded_model(state)
                return {
                    "guess": guess,
                    "model": "trained-mastermind-rl",
                    "usingFallback": False,
                    "modelPath": str(self.model_path),
                }
            except Exception as exc:
                self.load_error = str(exc)

        return {
            "guess": self._dummy_consistency_filter(state),
            "model": "dummy-consistency-filter",
            "usingFallback": True,
            "modelPath": str(self.model_path) if self.model_path else None,
            "loadError": self.load_error,
            "note": "Place the trained model in services/haic_api/models/ and set RL_MODEL_PATH in .env.docker.",
        }

    def _load_model(self, model_path: Path) -> Any:
        if not model_path.exists():
            raise FileNotFoundError(f"RL model artifact not found: {model_path}")

        if model_path.suffix.lower() == ".json":
            return json.loads(model_path.read_text(encoding="utf-8"))

        if model_path.suffix.lower() in {".pt", ".pth"}:
            import torch

            return torch.load(model_path, map_location="cpu")

        raise ValueError("Unsupported RL model artifact. Use .json, .pt, or .pth.")

    def _predict_with_loaded_model(self, state: Any) -> list[str]:
        state_dict = _state_to_dict(state)

        if isinstance(self.model, dict):
            return self._predict_from_json_policy(state_dict)

        if hasattr(self.model, "predict"):
            return _normalize_guess(self.model.predict(state_dict), state_dict["availableColors"], state_dict["codeLength"])

        if callable(self.model):
            return _normalize_guess(self.model(state_dict), state_dict["availableColors"], state_dict["codeLength"])

        raise TypeError("Loaded RL model must be a JSON policy, callable, or expose predict(state).")

    def _predict_from_json_policy(self, state: dict[str, Any]) -> list[str]:
        colors = state["availableColors"]
        code_length = state["codeLength"]

        opening = self.model.get("openingGuess") or self.model.get("defaultGuess")
        if opening and not state["guessHistory"]:
            return _normalize_guess(opening, colors, code_length)

        policy = self.model.get("policy", {})
        key = _state_key(state)
        if key in policy:
            return _normalize_guess(policy[key], colors, code_length)

        return self._dummy_consistency_filter(state)

    def _dummy_consistency_filter(self, state: Any) -> list[str]:
        state_dict = _state_to_dict(state)
        colors = state_dict["availableColors"] or ["Blue"]
        code_length = state_dict["codeLength"] or 4
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
    black = 0
    white = 0
    secret_work = secret[:]
    guess_work = guess[:]

    for idx, color in enumerate(guess_work):
        if color == secret_work[idx]:
            black += 1
            secret_work[idx] = None
            guess_work[idx] = None

    for idx, color in enumerate(guess_work):
        if color is None:
            continue
        try:
            match_idx = secret_work.index(color)
        except ValueError:
            continue
        white += 1
        secret_work[match_idx] = None

    return {"black": black, "white": white}


def _state_to_dict(state: Any) -> dict[str, Any]:
    if hasattr(state, "model_dump"):
        state = state.model_dump()
    elif not isinstance(state, dict):
        state = dict(state)

    return {
        "round": int(state.get("round", 1)),
        "codeLength": int(state.get("codeLength", 4)),
        "availableColors": list(state.get("availableColors") or ["Blue"]),
        "maxAttemptsPerRound": int(state.get("maxAttemptsPerRound", 10)),
        "attempt": int(state.get("attempt", 1)),
        "guessHistory": {str(k): list(v) for k, v in dict(state.get("guessHistory") or {}).items()},
        "feedbackHistory": {str(k): dict(v) for k, v in dict(state.get("feedbackHistory") or {}).items()},
    }


def _normalize_guess(raw_guess: Any, colors: list[str], code_length: int) -> list[str]:
    if isinstance(raw_guess, str):
        raw_guess = raw_guess.split(",")

    if not isinstance(raw_guess, list):
        raise ValueError("RL model returned a guess that is not a list.")

    normalized: list[str] = []
    for raw_color in raw_guess:
        color = str(raw_color).strip()
        match = next((candidate for candidate in colors if candidate.lower() == color.lower()), None)
        if not match:
            raise ValueError(f"RL model returned invalid color: {color}")
        normalized.append(match)

    if len(normalized) != code_length:
        raise ValueError(f"RL model returned {len(normalized)} colors, expected {code_length}.")

    return normalized


def _state_key(state: dict[str, Any]) -> str:
    compact = {
        "colors": state["availableColors"],
        "guessHistory": state["guessHistory"],
        "feedbackHistory": state["feedbackHistory"],
    }
    return json.dumps(compact, sort_keys=True, separators=(",", ":"))
