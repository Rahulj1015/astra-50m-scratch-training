# Scratch LLM

This repository is for a self-hosted language model implemented and trained from fundamentals.

The core model will not use Ollama, hosted AI APIs, or pretrained language-model weights.
PyTorch and NumPy are engineering tools for tensor operations, automatic differentiation,
GPU execution, and data handling.

## Current status

Step 1: reproducible local development environment.
Step 2: tensor operations, forward computation, loss, gradients, and gradient descent.
Step 3: tiny character-level causal Transformer trained from random initialization.

The architecture and completion gates are documented in
[`docs/architecture.md`](docs/architecture.md) and
[`docs/roadmap.md`](docs/roadmap.md). The repository is intentionally not
claiming to contain a finished LLM or production platform yet.

## Run the working tiny model

```powershell
.\.venv\Scripts\python.exe experiments\02_tiny_lm.py --steps 800 --prompt "The model"
```

This creates `checkpoints/tiny_lm.pt` and generates text. It is a learning
model, not a ChatGPT replacement: its vocabulary, corpus, parameter count,
and context are intentionally tiny.

## Run the local assistant UI

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. The API is available at `/api/chat` and
`/api/health`. This is a local development server, not an internet-hardened
public service.
