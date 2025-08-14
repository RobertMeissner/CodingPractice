from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from transformers import pipeline

app = FastAPI()
sentiment_pipeline = pipeline("sentiment-analysis")


class WordCountResponse(BaseModel):
    word_count: int
    text_length: int
    sentiment: str


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

    results = sentiment_pipeline(text)
    print(results)
    sentiment = results[0]["label"]

    return WordCountResponse(word_count=word_count, text_length=text_length, sentiment=sentiment)


@app.post("/count/words")
def count_words(request: WordCountRequest) -> WordCountResponse:
    try:
        return _count_words(request.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return "ok"


if __name__ == "__main__":
    pass
