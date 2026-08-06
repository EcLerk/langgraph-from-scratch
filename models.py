from typing import Literal, List

from pydantic import BaseModel, Field


class MessageClassification(BaseModel):
    message_type: Literal["review", "question"] = Field(
        description="Тип сообщения: отзыв или вопрос"
    )
    confidence: float = Field(
        description="Уверенность в классификации от 0.0 до 1.0",
        ge=0.0,
        le=1.0,
    )


class ReviewAnalysis(BaseModel):
    sentiment: Literal["pos", "neg", "neu"] = Field(
        description="Тональность отзыва"
    )
    confidence: float = Field(
        description="Уверенность в анализе от 0.0 до 1.0",
        ge=0.0,
        le=1.0,
    )
    key_topics: List[str] = Field(
        description="Ключевые темы из отзыва",
        max_length=5
    )
    summary: str = Field(
        description="Краткое резюме в одном предложении",
        max_length=150
    )