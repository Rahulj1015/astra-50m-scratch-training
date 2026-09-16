# Astra 50M: A Scratch-Trained Language Model

Astra is a small language-model project built to understand the mechanics of
training a causal Transformer from the ground up. The model starts from random
weights and is trained locally with PyTorch. It does not use pretrained model
weights, Ollama, or a hosted AI API as part of its core training path.

This repository is kept as a working reference for the 50 million parameter
version of Astra: how it was structured, trained, checkpointed, and exposed
through a simple local interface.

## What is included

- A 50.5M parameter character-level causal Transformer
- Training from random initialization with AdamW and mixed precision on CUDA
- Tensor, gradient, and optimisation experiments
- JSON metadata and model checkpoints tracked with Git LFS
- A small FastAPI service and browser-based chat interface
- Architecture notes and a roadmap for future experiments

The main model uses a hidden width of 512, 16 Transformer layers, and a
context length of 64. The current corpus is intentionally small, so this is an
educational and experimental model rather than a general-purpose assistant.

## Run the 50M model

Create a virtual environment, install the dependencies, and train the model:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe experiments\02_tiny_lm.py --checkpoint checkpoints\astra_50m.pt
```

The default configuration reports approximately 50.5 million parameters and
writes the checkpoint and its metadata to `checkpoints/`. Training can be
adjusted with options such as `--steps`, `--batch-size`, and
`--gradient-accumulation`.

## Run the local interface

```powershell
.\.venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. The service also exposes `/api/health` and
`/api/chat`. It is intended for local development and experimentation, not
public deployment.

## Model checkpoints

Checkpoint files are stored with Git LFS because the trained weights are too
large for ordinary GitHub file storage. The `.json` files alongside them record
the parameter count, training device, token count, and runtime metadata.

## Limitations

Astra currently has a character-level vocabulary and a very small training
corpus. It can demonstrate the training loop and generate text from that
corpus, but it does not have the data, tokenizer, context length, or evaluation
needed for reliable general-purpose language understanding. The next meaningful
improvements are better data, a proper tokenizer, evaluation benchmarks, and a
more modular training pipeline.

## Project notes

See [`docs/architecture.md`](docs/architecture.md) for the system boundaries
and [`docs/roadmap.md`](docs/roadmap.md) for planned work.
