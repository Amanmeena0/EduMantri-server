# src/schemas/api_schemas.py
from typing import List, Optional
from pydantic import BaseModel, Field

class QueryInput(BaseModel):
    query: str = Field(..., description="The query string sent by the student.")
    session_id: Optional[str] = Field(None, description="Unique conversation session identifier.")

class BotResponse(BaseModel):
    response: str = Field(..., description="The generated response text from the AI system.")
    followups: List[str] = Field(default_factory=list, description="Follow-up suggestions for continuous learning.")
