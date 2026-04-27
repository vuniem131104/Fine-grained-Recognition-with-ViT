from __future__ import annotations

from typing import Optional

from pydantic import BaseModel
from pydantic import HttpUrl
from pydantic import SecretStr


class LiteLLMSetting(BaseModel):
    url: HttpUrl
    token: Optional[SecretStr] = None
    model: str
    frequency_penalty: float
    n: int
    temperature: float
    top_p: float
    max_completion_tokens: int
    dimension: int
    embedding_model: str
