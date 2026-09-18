"""
model.py
--------
Thin wrapper around the fine-tuned DistilBERT model. Loads the model and
tokenizer ONCE (singleton pattern) so `main.py` can reuse the same
in-memory model for every request instead of reloading it per call, which
is what keeps inference latency low.
"""

import os
from typing import List, TypedDict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MAX_TOKENS = 512


class ModerationResult(TypedDict):
    label: str
    confidence: float


class ContentModerationModel:
    """Loads a fine-tuned DistilBERT classifier and exposes predict methods."""

    def __init__(self, model_dir: str = "model_artifacts", device: str | None = None):
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"Model directory '{model_dir}' not found. Run train.py first "
                "to fine-tune and save the model, or point model_dir at a "
                "valid model folder."
            )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()  # inference mode: disables dropout etc.

        # id2label was saved during training (0: safe, 1: toxic)
        self.id2label = self.model.config.id2label

    @torch.inference_mode()
    def _predict_logits(self, texts: List[str]) -> torch.Tensor:
        """Tokenize a batch of texts and run a single forward pass."""
        encoded = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_TOKENS,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**encoded)
        return outputs.logits

    def predict(self, text: str) -> ModerationResult:
        result = self.predict_batch([text])
        return result[0]

    def predict_batch(self, texts: List[str]) -> List[ModerationResult]:
        logits = self._predict_logits(texts)
        probs = torch.softmax(logits, dim=-1)  # shape: (batch, 2)

        results: List[ModerationResult] = []
        for row in probs:
            pred_id = int(torch.argmax(row).item())
            confidence = float(row[pred_id].item())
            label = self.id2label[pred_id]
            results.append({"label": label, "confidence": round(confidence, 4)})
        return results

    def count_tokens(self, text: str) -> int:
        """Number of tokens BEFORE truncation, used for input validation."""
        return len(self.tokenizer.encode(text, add_special_tokens=True))
