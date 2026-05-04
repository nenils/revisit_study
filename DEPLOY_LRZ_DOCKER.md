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

Important: `localhost` means "this same machine". If the browser is running on your laptop, `http://localhost:8080` points to your laptop, not to the LRZ VM.

For testing from your laptop without a domain, use an SSH tunnel:

```bash
ssh -L 8080:localhost:8080 neni@<vm-hostname-or-ip>
```

Keep that SSH session open, then open this on your laptop:

```text
http://localhost:8080
```

If you open a browser directly on the VM, then `http://localhost:8080` is also correct there. If you want to access the study without an SSH tunnel, use `http://<vm-hostname-or-ip>:8080` and make sure LRZ/firewall rules allow inbound TCP traffic on port `8080`.

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

Route check:

```bash
curl -I http://localhost:8080
curl -I http://localhost:8080/HAIC_study
```

Both should return `200 OK` from the VM. If these work on the VM but the link fails in your laptop browser, the problem is network access to the VM, not the study build.
