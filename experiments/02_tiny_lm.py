"""Train and run a tiny character-level causal language model from scratch."""

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


CORPUS = """
You are learning how language models work from first principles.
This small model predicts the next character from the characters before it.
Tokens are numbers, embeddings turn numbers into vectors, attention mixes context,
and the output layer produces a probability for every character in the vocabulary.
The model is small, but the training loop is real: forward pass, causal masking,
cross entropy loss, backpropagation, Adam updates, checkpointing, and sampling.
Build carefully, measure honestly, and improve one component at a time.
""" * 80


class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, width: int = 128, layers: int = 2) -> None:
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.position_embedding = nn.Embedding(context_length, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=4,
            dim_feedforward=4 * width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=layers)
        self.normalization = nn.LayerNorm(width)
        self.output = nn.Linear(width, vocab_size, bias=False)
        self.output.weight = self.token_embedding.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, length = tokens.shape
        positions = torch.arange(length, device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        mask = torch.triu(
            torch.ones(length, length, device=tokens.device, dtype=torch.bool), diagonal=1
        )
        hidden = self.blocks(hidden, mask=mask)
        return self.output(self.normalization(hidden))


def make_batch(data: torch.Tensor, context_length: int, batch_size: int, device: torch.device):
    starts = torch.randint(0, len(data) - context_length - 1, (batch_size,))
    inputs = torch.stack([data[index : index + context_length] for index in starts])
    targets = torch.stack([data[index + 1 : index + context_length + 1] for index in starts])
    return inputs.to(device), targets.to(device)


@torch.no_grad()
def generate(model, prompt: str, encode, decode, device, tokens_to_generate: int, temperature: float):
    tokens = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    model.eval()
    for _ in range(tokens_to_generate):
        context = tokens[:, -model.context_length :]
        probabilities = F.softmax(model(context)[:, -1, :] / temperature, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        tokens = torch.cat((tokens, next_token), dim=1)
    return decode(tokens[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--prompt", default="The model")
    parser.add_argument("--generate", type=int, default=240)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--checkpoint", default="checkpoints/astra_50m.pt")
    args = parser.parse_args()

    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    characters = sorted(set(CORPUS))
    encode_map = {character: index for index, character in enumerate(characters)}
    decode_map = {index: character for character, index in encode_map.items()}
    encode = lambda text: [encode_map.get(character, encode_map[" "]) for character in text]
    decode = lambda ids: "".join(decode_map[index] for index in ids)
    data = torch.tensor(encode(CORPUS), dtype=torch.long)
    split = int(0.9 * len(data))
    train_data, validation_data = data[:split], data[split:]
    context_length = args.context_length
    model = TinyLanguageModel(
        len(characters), context_length, width=args.width, layers=args.layers
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    start = time.perf_counter()

    print(f"device={device}")
    print(f"parameters={parameter_count:,}")
    print(f"vocabulary={len(characters)} context_length={context_length}")
    print(f"width={args.width} layers={args.layers} batch={args.batch_size} accumulation={args.gradient_accumulation}")
    print(f"mixed_precision={use_amp}")

    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.gradient_accumulation):
            inputs, targets = make_batch(train_data, context_length, args.batch_size, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(inputs)
                loss = F.cross_entropy(logits.reshape(-1, len(characters)), targets.reshape(-1))
            scaler.scale(loss / args.gradient_accumulation).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % 100 == 0 or step == args.steps:
            model.eval()
            validation_inputs, validation_targets = make_batch(
                validation_data, context_length, args.batch_size, device
            )
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                validation_logits = model(validation_inputs)
            validation_loss = F.cross_entropy(
                validation_logits.reshape(-1, len(characters)), validation_targets.reshape(-1)
            )
            print(
                f"step={step:04d} train_loss={loss.item():.4f} "
                f"validation_loss={validation_loss.item():.4f}"
            )

    checkpoint = {
        "model": model.state_dict(),
        "vocabulary": characters,
        "context_length": context_length,
        "width": args.width,
        "layers": args.layers,
    }
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    metadata = {
        "parameters": parameter_count,
        "tokens": len(data),
        "seconds": round(time.perf_counter() - start, 2),
        "device": str(device),
        "width": args.width,
        "layers": args.layers,
    }
    checkpoint_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(f"checkpoint={checkpoint_path}")
    print("--- generated text ---")
    print(generate(model, args.prompt, encode, decode, device, args.generate, 0.8))


if __name__ == "__main__":
    main()