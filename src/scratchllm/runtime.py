from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, width: int = 128, layers: int = 2) -> None:
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.position_embedding = nn.Embedding(context_length, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=4, dim_feedforward=4 * width,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
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


class ScratchAssistant:
    def __init__(self, checkpoint_path: str = "checkpoints/tiny_lm.pt") -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(Path(checkpoint_path), map_location=self.device, weights_only=False)
        self.characters = checkpoint["vocabulary"]
        self.encode_map = {character: index for index, character in enumerate(self.characters)}
        self.decode_map = {index: character for character, index in enumerate(self.characters)}
        self.model = TinyLanguageModel(
            len(self.characters), checkpoint["context_length"],
            width=checkpoint.get("width", 128), layers=checkpoint.get("layers", 2)
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

    def _encode(self, text: str) -> list[int]:
        fallback = self.encode_map[" "]
        return [self.encode_map.get(character, fallback) for character in text]

    def _decode(self, ids: list[int]) -> str:
        return "".join(self.characters[index] for index in ids)

    @torch.no_grad()
    def generate(self, prompt: str, token_count: int = 180, temperature: float = 0.8) -> str:
        tokens = torch.tensor([self._encode(prompt)], dtype=torch.long, device=self.device)
        for _ in range(token_count):
            context = tokens[:, -self.model.context_length :]
            probabilities = F.softmax(self.model(context)[:, -1, :] / temperature, dim=-1)
            tokens = torch.cat((tokens, torch.multinomial(probabilities, 1)), dim=1)
        return self._decode(tokens[0].tolist())