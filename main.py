"""
main.py
-------
FastAPI REST API for the Content Moderation model.

Endpoints:
  POST /moderate        -> classify a single text
  POST /moderate/batch  -> classify a list of texts
  GET  /health           -> liveness check

Run with:
  uvicorn main:app --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from model import ContentModerationModel, MAX_TOKENS

# ---------------------------------------------------------------------------
# Model lifecycle: load once at startup, reuse for every request.
# ---------------------------------------------------------------------------
model_holder: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_holder["model"] = ContentModerationModel(model_dir="model_artifacts")
    print("Model loaded and ready.")
    yield
    model_holder.clear()


app = FastAPI(
    title="Content Moderation API",
    description="Fine-tuned DistilBERT classifier for flagging toxic text.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class ModerateRequest(BaseModel):
    text: str = Field(..., description="Text to classify")

    @field_validator("text")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("text must not be empty")
        return v


class ModerateBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=100)

    @field_validator("texts")
    @classmethod
    def no_empty_items(cls, v: List[str]) -> List[str]:
        if any(not t or not t.strip() for t in v):
            raise ValueError("texts must not contain empty strings")
        return v


class ModerateResponse(BaseModel):
    label: str
    confidence: float


class HealthResponse(BaseModel):
    status: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _validate_length(model: ContentModerationModel, text: str) -> None:
    token_count = model.count_tokens(text)
    if token_count > MAX_TOKENS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Text is too long ({token_count} tokens). "
                f"Maximum allowed is {MAX_TOKENS} tokens."
            ),
        )


def get_model() -> ContentModerationModel:
    model = model_holder.get("model")
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded yet.",
        )
    return model


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/moderate", response_model=ModerateResponse)
def moderate(payload: ModerateRequest) -> ModerateResponse:
    model = get_model()
    _validate_length(model, payload.text)
    result = model.predict(payload.text)
    return ModerateResponse(**result)


@app.post("/moderate/batch", response_model=List[ModerateResponse])
def moderate_batch(payload: ModerateBatchRequest) -> List[ModerateResponse]:
    model = get_model()
    for text in payload.texts:
        _validate_length(model, text)
    results = model.predict_batch(payload.texts)
    return [ModerateResponse(**r) for r in results]
