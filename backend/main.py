import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.generation.rag import answer_question

load_dotenv()

app = FastAPI(title="PhoenixAI API")

MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() == "true"


class HistoryMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    text: str = Field(min_length=1, max_length=2000)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=8)


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=AnswerResponse)
def chat(request: QuestionRequest):
    if MOCK_MODE:
        return AnswerResponse(
            answer=f"זוהי תשובת בדיקה של PhoenixAI. השאלה שהתקבלה: {request.question}",
            sources=[],
        )

    try:
        history = [
            {"role": message.role, "text": message.text}
            for message in request.history
        ]

        result = answer_question(
            question=request.question,
            history=history,
        )

        return AnswerResponse(
            answer=result["answer"],
            sources=result["sources"],
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="אירעה שגיאה בעיבוד השאלה.",
        )