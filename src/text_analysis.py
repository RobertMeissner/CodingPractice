import hashlib
import time
from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from transformers import pipeline

app = FastAPI()

MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

sentiment_pipeline = pipeline("text-classification", model=MODEL_NAME, device=-1)


class WordCountResponse(BaseModel):
    word_count: int
    text_length: int
    sentiment: str
    timing_info: Optional[dict] = None


class WordCountRequest(BaseModel):
    text: str = Field(min_length=0, max_length=10000, description="Text")

    # doubles with pydantic, just in case
    @field_validator("text")
    @classmethod
    def text_must_not_be_none(_, v):
        # due to Field init useless
        if v is None:
            raise ValueError("Text cannot be None")
        return v


def _count_words(text: str) -> WordCountResponse:
    text_length = len(text)
    word_count = len(text.split())

    text_hash = hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()
    sentiment = _cached_sentiment(text_hash, text)

    return WordCountResponse(word_count=word_count, text_length=text_length, sentiment=sentiment)


@lru_cache(maxsize=10)
def _cached_sentiment(_text_hash: str, text: str) -> str:
    results = sentiment_pipeline(text)
    return results[0]["label"]


@app.post("/count/words")
def count_words(request: WordCountRequest, dev: bool = False) -> WordCountResponse:
    try:
        start_time = time.time()
        result = _count_words(request.text)
        if dev:
            total_time = time.time() - start_time
            result.timing_info = {
                "total_time_ms": round(total_time * 1000, 2),
                "model_loaded": sentiment_pipeline is not None,
                "text_length": len(request.text),
            }
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return "ok"


if __name__ == "__main__":
    pass
