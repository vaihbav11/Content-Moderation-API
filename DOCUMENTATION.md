# How This Project Works

A plain-language walkthrough of every file, for your own understanding
(and so you can explain it in an interview or on your portfolio).

## The big picture

There are two separate phases:

1. **Training** (`train.py`) — happens once, offline, on your machine or a
   GPU box. Takes raw labeled data and produces a saved model on disk.
2. **Serving** (`model.py` + `main.py`) — happens continuously, in
   production. Loads that saved model into memory and answers HTTP
   requests with it.

Think of training as "baking the cake" and serving as "a waiter who hands
out slices all day without re-baking."

---

## 1. `train.py` — teaching DistilBERT to spot toxicity

**What DistilBERT is:** a smaller, faster version of BERT — a neural
network pretrained on huge amounts of general text so it already
"understands" language (grammar, meaning, context). We don't train it from
scratch; we *fine-tune* it, i.e. nudge its existing knowledge toward one
specific task: telling toxic text from safe text.

**Step by step:**

1. **Load the Jigsaw dataset.** Each row is a comment plus six 0/1 columns
   (`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`,
   `identity_hate`). We collapse those six into one binary label: `1` if
   *any* of them is `1` ("toxic"), else `0` ("safe"). This matches the
   simple toxic/safe output the API promises.

2. **Sample 15K rows, stratified.** "Stratified" means we keep the same
   ratio of toxic-to-safe examples in the sample as in the full dataset,
   so the model doesn't get a skewed picture (toxic comments are a
   minority in this dataset — most comments online are fine).

3. **Tokenize.** Neural nets don't read words — they read numbers. The
   tokenizer chops each comment into subword pieces and converts them to
   IDs from DistilBERT's vocabulary, padding/truncating everything to a
   fixed length (512 tokens, DistilBERT's max).

4. **Split train/eval.** 85% of the data trains the model; 15% is held
   back to *check* how well it generalizes to text it hasn't seen.

5. **Fine-tune.** HuggingFace's `Trainer` handles the training loop:
   feed batches through the model, compare predictions to true labels,
   compute a loss, and adjust the model's weights (backpropagation) to
   reduce that loss. This repeats for 3 epochs (full passes over the
   training data).

6. **Evaluate with precision as the priority metric.** After each epoch we
   compute accuracy, precision, recall, and F1 on the held-out eval set.
   The script keeps the checkpoint with the *best precision*, because the
   goal stated for this project is "91%+ precision" — i.e., when the
   model says something is toxic, it should be right 91%+ of the time.
   (High precision matters here because a false positive means silencing
   or flagging someone who did nothing wrong.)

   > Precision vs. recall, quickly: **precision** = "of the things I
   > flagged as toxic, how many actually were?" **recall** = "of all the
   > actually-toxic comments, how many did I catch?" You can trade one for
   > the other by moving the decision threshold — e.g. only call
   > something "toxic" if the model is >70% confident, instead of the
   > default >50%, which raises precision but lowers recall.

7. **Save the model.** `trainer.save_model()` and
   `tokenizer.save_pretrained()` write the fine-tuned weights and the
   tokenizer's vocab/config to `model_artifacts/`. This folder is what the
   API loads later — training never has to run again unless you want to
   retrain.

---

## 2. `model.py` — the reusable inference wrapper

This file's whole job is: **load the model once, then answer predictions
fast, over and over.**

- `ContentModerationModel.__init__` loads the tokenizer and the
  fine-tuned model from `model_artifacts/` and moves them onto GPU if one
  is available, otherwise CPU. `self.model.eval()` switches the model into
  inference mode (turns off dropout, a training-only regularization
  trick, so predictions are deterministic).

- `predict_batch()` is the core method. It tokenizes a *list* of texts
  together in one call (`self.tokenizer(texts, ...)`), pads them to the
  same length, and runs them through the model in a **single forward
  pass**. That's the key performance trick: classifying 20 texts in one
  batched call is much faster than calling the model 20 separate times,
  especially on a GPU which is built for parallel matrix math.

- The model outputs raw scores called **logits** — one number per class
  (safe, toxic), not yet probabilities. `torch.softmax` converts them into
  a probability distribution that sums to 1, so `{"safe": 0.03, "toxic":
  0.97}` is something we can present as a confidence score.

- `@torch.inference_mode()` tells PyTorch "don't bother tracking gradients
  for this," which saves memory and speeds things up — gradients are only
  needed during training, not when just asking the model for an answer.

- `count_tokens()` tokenizes text *without* truncating, purely to count
  how many tokens it would need — this is what `main.py` uses to reject
  overly long inputs with a clear error instead of silently truncating
  them (which would mean judging only part of what someone wrote).

---

## 3. `main.py` — the web layer

FastAPI turns Python functions into HTTP endpoints, and Pydantic models
define the shape of the JSON going in and out (and validate it
automatically).

- **`lifespan`** is a FastAPI hook that runs code on startup and shutdown.
  Here it loads `ContentModerationModel` into `model_holder["model"]`
  exactly once when the server process starts — not on every request.
  Every incoming request then reuses that same in-memory model via
  `get_model()`.

- **`ModerateRequest` / `ModerateBatchRequest`** are Pydantic schemas.
  FastAPI uses them to parse and validate the incoming JSON body
  automatically — if `text` is missing or not a string, the client gets a
  `422` error before your code even runs. The custom `field_validator`
  methods add the "must not be empty/whitespace-only" rule the spec asked
  for.

- **`_validate_length`** uses `model.count_tokens()` to enforce the
  512-token limit, returning a `400` with a helpful message if a text is
  too long, rather than truncating it silently.

- **The three endpoints** are thin: validate → call the model → shape the
  response with `ModerateResponse`. All the actual ML logic lives in
  `model.py`, which keeps `main.py` focused purely on HTTP concerns —
  a common and useful separation (web layer vs. inference layer).

---

## 4. Why this design is "architected for throughput and low latency"

- **Load once, serve many:** the expensive part (loading weights onto
  GPU/CPU) happens once at startup, not per-request.
- **Batching:** `/moderate/batch` processes many texts in one forward
  pass instead of a Python loop calling the model repeatedly.
- **No gradient tracking at inference:** `torch.inference_mode()` skips
  bookkeeping PyTorch would otherwise do to support training.
- **Fixed max length (512 tokens):** bounds how much compute any single
  request can demand, so one huge input can't stall the server or blow up
  memory.
- **Stateless process:** the API holds no per-user session state, so you
  can run multiple copies behind a load balancer (e.g. several `uvicorn`
  workers, or multiple containers) to scale throughput horizontally.

---

## 5. What you'd add for a production deployment (not included here, but worth knowing)

- **Containerize** with a `Dockerfile` so the environment (Python +
  CUDA/torch versions) is reproducible.
- **Model hosting:** commit `model_artifacts/` to Git LFS, or push it to
  the HuggingFace Hub / an S3 bucket and download it at container build or
  startup time — raw weights are usually too large for a normal git repo.
- **Multiple workers:** `uvicorn main:app --workers 4` (or a process
  manager like Gunicorn+Uvicorn workers) to use multiple CPU cores /
  handle concurrent requests.
- **Monitoring:** log prediction latency and label distribution over time
  to catch model or data drift.
- **Rate limiting / auth:** add an API key check if this were exposed
  publicly.
