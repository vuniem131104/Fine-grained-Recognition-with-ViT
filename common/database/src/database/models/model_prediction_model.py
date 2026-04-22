from __future__ import annotations

from typing import Any

from sqlalchemy import Float, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .date import Date


class ModelPrediction(Date):
    __tablename__ = "model_predictions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    base64_image: Mapped[str] = mapped_column(Text, nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_class: Mapped[str] = mapped_column(String(255), nullable=False)
    alternatives: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
