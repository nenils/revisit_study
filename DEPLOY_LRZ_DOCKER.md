# Deploying the HAIC Study on an LRZ VM with Docker

This setup runs two containers:

- `study`: builds the ReVISit frontend and serves it with Nginx.
- `haic-api`: provides `/api/rl/predict` with a dummy Mastermind RL model and `/api/llm/chat` as a server-side LLM proxy.

## 1. Prepare the VM

Install Docker and the Docker Compose plugin on the LRZ VM, then copy this repository to the VM.

## 2. Configure tokens and URLs

The compose setup uses a dedicated Docker env file. Create it from the example:

```bash
cp .env.docker.example .env.docker
```

Put your LLM token in `.env.docker`:

```bash
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=meta-llama/llama-3.1-70b-instruct
STUDY_PUBLIC_URL=https://your-lrz-vm.example
```

Do not put the LLM token in any HTML file. The browser calls `/api/llm/chat`; the `haic-api` container attaches the token server-side.

## 3. Build and run

```bash
docker compose up --build -d
```

Open:

```text
http://<vm-hostname-or-ip>:8080
```

If you use a reverse proxy or LRZ-provided HTTPS endpoint, point it to port `8080` and set `STUDY_PUBLIC_URL` to the public HTTPS URL.

## 4. Dummy RL model

The dummy model lives in:

```text
services/haic_api/app.py
```

Replace `dummy_rl_guess(...)` with your trained Mastermind RL model inference. The frontend sends:

```json
{
  "round": 3,
  "codeLength": 4,
  "availableColors": ["Blue", "Green", "Red"],
  "attempt": 2,
  "guessHistory": { "1": ["Blue", "Green", "Red", "Blue"] },
  "feedbackHistory": { "1": { "black": 1, "white": 2 } }
}
```

Return:

```json
{ "guess": ["Green", "Blue", "Blue", "Red"] }
```

## 5. Useful commands

```bash
docker compose logs -f
docker compose restart
docker compose down
docker compose up --build -d
```

Health check:

```bash
curl http://localhost:8080/api/health
```
