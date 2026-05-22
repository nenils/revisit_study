import json
import os
import urllib.error
import urllib.request
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rl_model import MastermindRLModel


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-70b-instruct")
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL",
    "https://openrouter.ai/api/v1/chat/completions",
)
STUDY_PUBLIC_URL = os.getenv("STUDY_PUBLIC_URL", "http://localhost:8080")
RL_MODEL_PATH = os.getenv("RL_MODEL_PATH", "").strip()
RL_MODEL = MastermindRLModel(RL_MODEL_PATH or None)
MAX_LLM_MESSAGES_PER_ROUND = int(os.getenv("MAX_LLM_MESSAGES_PER_ROUND", "20"))
LLM_MESSAGE_COUNTS: dict[tuple[str, int], int] = {}
ALLOWED_STUDY_ORIGINS = {
    STUDY_PUBLIC_URL,
    "http://iivm6.cit.tum.de",
    "https://iivm6.cit.tum.de",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
}

app = FastAPI(title="HAIC Study API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=sorted(ALLOWED_STUDY_ORIGINS),
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


class RLState(BaseModel):
    reason: str | None = None
    round: int = 1
    codeLength: int = 4
    availableColors: list[str] = Field(default_factory=lambda: ["Blue"])
    maxAttemptsPerRound: int = 10
    attempt: int = 1
    guessHistory: dict[str, list[str]] = Field(default_factory=dict)
    feedbackHistory: dict[str, dict[str, int]] = Field(default_factory=dict)


class ChatPayload(BaseModel):
    participantId: str | None = None
    studyRound: int | None = None
    model: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: list[str] | None = None
    messages: list[dict[str, Any]]


def dummy_llm_response(payload: ChatPayload) -> dict[str, Any]:
    text = "\n".join(str(message.get("content", "")) for message in payload.messages)
    prediction = None
    final_guess = None

    for marker in ("PREDICTION:", "FINAL GUESS:"):
        marker_idx = text.upper().rfind(marker)
        if marker_idx == -1:
            continue
        bracket_start = text.find("[", marker_idx)
        bracket_end = text.find("]", bracket_start)
        if bracket_start != -1 and bracket_end != -1:
            colors = [part.strip() for part in text[bracket_start + 1:bracket_end].split(",")]
            if len(colors) == 4:
                if marker == "PREDICTION:":
                    prediction = colors
                else:
                    final_guess = colors

    if prediction:
        content = (
            "The dummy LLM proxy is active. The trained RL model suggested this guess "
            "because it is consistent with the current game state supplied by the frontend. "
            f"PREDICTION: [{', '.join(prediction)}]"
        )
    elif final_guess:
        content = (
            "The dummy LLM proxy is active. I would keep this proposed guess for now because "
            "no external LLM token is configured to provide a deeper review. "
            f"FINAL GUESS: [{', '.join(final_guess)}]"
        )
    else:
        content = (
            "The dummy LLM proxy is active. Configure OPENROUTER_API_KEY in the Docker "
            "environment for full explanations. FINAL GUESS: [Blue, Blue, Blue, Blue]"
        )

    return {"choices": [{"message": {"content": content}}], "dummy": True}


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "rlModelLoaded": RL_MODEL.is_loaded,
        "rlModelPath": str(RL_MODEL.model_path),
        "rlModelLoadError": RL_MODEL.load_error,
    }


@app.post("/api/rl/predict")
def predict_rl(state: RLState) -> dict[str, Any]:
    return RL_MODEL.predict(state)


@app.post("/api/llm/chat")
def llm_chat(payload: ChatPayload) -> dict[str, Any]:
    participant_id = (payload.participantId or "anonymous").strip()[:128] or "anonymous"
    study_round = payload.studyRound or 0
    counter_key = (participant_id, study_round)
    current_count = LLM_MESSAGE_COUNTS.get(counter_key, 0)
    if current_count >= MAX_LLM_MESSAGES_PER_ROUND:
        raise HTTPException(
            status_code=429,
            detail=(
                "The AI message limit for this round has been reached. "
                "Please continue with the game board; the AI chat will be available again in the next round."
            ),
        )
    LLM_MESSAGE_COUNTS[counter_key] = current_count + 1

    if not OPENROUTER_API_KEY:
        return dummy_llm_response(payload)

    body = payload.model_dump(exclude={"participantId", "studyRound"})
    body["model"] = body.get("model") or OPENROUTER_MODEL
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        OPENROUTER_BASE_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": STUDY_PUBLIC_URL,
            "X-Title": "HAIC Mastermind Study",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(status_code=exc.code, detail=detail) from exc
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=502, detail=str(exc.reason)) from exc
