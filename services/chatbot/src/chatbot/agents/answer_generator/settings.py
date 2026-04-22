from pydantic import BaseModel


class AnswerGeneratorSetting(BaseModel):
    model: str
    temperature: float
    top_p: float
    n: int
    frequency_penalty: float
    max_completion_tokens: int
    reasoning_effort: str | None = None