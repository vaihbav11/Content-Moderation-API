# Content Moderation API

A fine-tuned DistilBERT model, trained on the Jigsaw Toxic Comment dataset,
served as a fast REST API with FastAPI. Classifies text as `toxic` or `safe`
with a confidence score.

- Fine-tuned on 15K labeled social posts
- Target: 91%+ precision on toxic detection
- Model loaded once at startup and reused for every request (low latency)

## Project structure

```
content-moderation-api/
├── train.py            # Fine-tunes DistilBERT on the Jigsaw dataset
├── model.py             # Loads the model once, exposes predict()/predict_batch()
├── main.py               # FastAPI app: /moderate, /moderate/batch, /health
├── requirements.txt
├── README.md
└── model_artifacts/     # Saved fine-tuned model (created by train.py)
```

## 1. Setup

```bash
git clone https://github.com/<your-username>/content-moderation-api.git
cd content-moderation-api
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get the dataset and train the model

1. Download `train.csv` from the [Jigsaw Toxic Comment Classification
   Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
   on Kaggle (requires a free Kaggle account).
2. Place it at `data/train.csv`.
3. Run training:

```bash
python train.py --data_path data/train.csv --output_dir model_artifacts --sample_size 15000 --epochs 3
```

This fine-tunes `distilbert-base-uncased` as a binary classifier (toxic vs.
safe) and saves the model + tokenizer to `model_artifacts/`. Training prints
precision/recall/F1 at the end of each epoch, and keeps the checkpoint with
the best precision (`load_best_model_at_end=True`).

> GPU strongly recommended for training. CPU training will work but is slow.
> If you don't want to train yourself, you can point `model_dir` in
> `model.py` at any compatible HuggingFace sequence-classification model.

## 3. Run the API

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The model is loaded once when the server starts (you'll see `Model loaded
and ready.` in the logs), then reused for every request — no reloading per
call.

Interactive API docs: http://localhost:8000/docs

## 4. Endpoints

### `GET /health`

```bash
curl http://localhost:8000/health
```
```json
{"status": "ok"}
```

### `POST /moderate`

```bash
curl -X POST http://localhost:8000/moderate \
  -H "Content-Type: application/json" \
  -d '{"text": "You are an idiot and should disappear."}'
```
```json
{"label": "toxic", "confidence": 0.97}
```

### `POST /moderate/batch`

```bash
curl -X POST http://localhost:8000/moderate/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Have a great day!", "You are worthless."]}'
```
```json
[
  {"label": "safe", "confidence": 0.99},
  {"label": "toxic", "confidence": 0.95}
]
```

## Input validation

- Empty or whitespace-only strings are rejected (`422 Unprocessable Entity`).
- Text longer than 512 tokens (counted with the model's own tokenizer, not
  just characters) is rejected with a `400 Bad Request` explaining the
  token count and limit.
- `/moderate/batch` accepts 1–100 texts per request.

## Notes on performance

- The model and tokenizer are loaded exactly once, at process startup,
  using FastAPI's `lifespan` context manager — not on every request.
- Batch requests run as a single forward pass through the model (one
  tokenizer call with padding, one `model()` call) rather than looping
  per-text, which is significantly faster on GPU and still helps on CPU.
- `@torch.inference_mode()` disables gradient tracking during inference.

## License

MIT (or your preferred license — update this section).
