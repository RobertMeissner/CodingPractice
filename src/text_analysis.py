import hashlib
import time
import uuid
from datetime import datetime
from functools import lru_cache
from typing import Optional
import numpy as np

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field, field_validator
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Float
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from transformers import pipeline

app = FastAPI()

MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

sentiment_pipeline = pipeline("text-classification", model=MODEL_NAME, device=-1)

engine = create_engine("sqlite:///./text_analysis.db")
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class AnalysisCache(Base):
    __tablename__ = "analysis_cache"

    text_hash = Column(String, primary_key=True)
    text = Column(String)
    word_count = Column(Integer)
    sentiment = Column(String)
    processing_time_ms = Column(Float)
    created_at = Column(DateTime, default=datetime.now())


Base.metadata.create_all(bind=engine)


def _get_from_db_cache(text_hash: str, db: Session) -> Optional[str]:
    cached = db.query(AnalysisCache).filter(AnalysisCache.text_hash == text_hash).first()
    return cached.sentiment if cached else None


def _save_to_db_cache(text_hash: str, text: str, sentiment: str, db: Session) -> None:
    analysis = AnalysisCache(text_hash=text_hash, text=text[:500], sentiment=sentiment, word_count=len(text.split()), processing_time_ms=0)
    db.add(analysis)
    db.commit()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


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


def _count_words(text: str, db) -> WordCountResponse:
    text_length = len(text)
    word_count = len(text.split())

    # TODO: not yet optimum, should be Memory → Disk → Compute
    text_hash = hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()
    sentiment = _get_from_db_cache(text_hash, db)
    if not sentiment:
        sentiment = _cached_sentiment(text_hash, text)
        _save_to_db_cache(text_hash, text, sentiment, db)

    return WordCountResponse(word_count=word_count, text_length=text_length, sentiment=sentiment)


@lru_cache(maxsize=10)
def _cached_sentiment(_text_hash: str, text: str) -> str:
    results = sentiment_pipeline(text)
    return results[0]["label"]


@app.post("/count/words")
def count_words(request: WordCountRequest, dev: bool = False, db: Session = Depends(get_db)) -> WordCountResponse:
    try:
        start_time = time.time()
        result = _count_words(request.text, db)
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


documents: dict[str, dict] = {}


class DocumentInput(BaseModel):
    text: str
    title: str


class DocumentInputResponse(BaseModel):
    doc_id: str


class SimilarityResponse(BaseModel):
    title: str
    doc_id: str
    similarity: float


@app.post("/document")
def add_document(doc: DocumentInput) -> DocumentInputResponse:
    doc_id = str(uuid.uuid4())
    documents[doc_id] = {"title": doc.title, "text": doc.text}
    return DocumentInputResponse(doc_id=doc_id)


semantic_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


def semantic_similarity(text1: str, text2: str) -> float:
    embeddings = semantic_model.encode([text1, text2])
    cosine = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
    return cosine


@app.get("/similar/{doc_id}")
def document_similarity(doc_id: str) -> SimilarityResponse:
    if doc_id not in documents.keys():
        raise HTTPException(status_code=400, detail=f"Document {doc_id} not found.")

    # target = documents[doc_id]
    similarities: list[SimilarityResponse] = []

    for second_id, second_doc in documents.items():
        if second_id != doc_id:
            similarities.append(
                SimilarityResponse(
                    title=second_doc["title"],
                    doc_id=second_id,
                    similarity=semantic_similarity(second_doc["text"], documents[doc_id]["text"]),
                )
            )

    similarities.sort(key=lambda item: item.similarity, reverse=True)
    return similarities[0]


@app.get("/health")
def health():
    return "ok"


@app.get("/analytics")
def get_analytics(db: Session = Depends(get_db)):
    total_requests = db.query(AnalysisCache).count()
    return {"total_requests": total_requests, "cache_size": total_requests}


if __name__ == "__main__":
    pass
