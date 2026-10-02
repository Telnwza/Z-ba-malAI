"""API: POST /translate

Run: uvicorn src.backend.app:app --reload   then open http://127.0.0.1:8000/docs
"""
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .translator import Translator

app = FastAPI(title="Thai Gen Z Translator")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
translator = Translator()


class TranslateRequest(BaseModel):
    """Request body of POST /translate. Out-of-range values get an automatic 422 from FastAPI."""
    text: str = Field(min_length=1, max_length=500)
    direction: Literal["to_genz", "to_normal"]
    max_style_level: int = Field(3, ge=1, le=4)
    allow_profanity: bool = False
    top_k: int = Field(5, ge=1, le=10)


@app.post("/translate")
def translate(req: TranslateRequest):
    """Translate one Thai text between normal language and Gen Z slang.

    Request body:
    - text (str, required): text to translate, 1-500 characters.
    - direction ("to_genz" | "to_normal", required): "to_genz" turns normal Thai into
      Gen Z slang; "to_normal" turns Gen Z slang back into normal Thai.
    - max_style_level (int 1-4, default 3): only use rules whose Style Level is at most this.
    - allow_profanity (bool, default false): true also allows rules labelled Profanity/Insult.
    - top_k (int 1-10, default 5): maximum number of entries returned in `candidates`.

    Response 200: {output, applied_rules, candidates} (field details in README).
    Invalid or out-of-range values return FastAPI's standard 422.
    """
    return translator.translate(
        req.text, req.direction, req.max_style_level, req.allow_profanity, req.top_k
    )
