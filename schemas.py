from pydantic import BaseModel, Field
from typing import Literal

Horizon = Literal["short","swing","position"]
Action = Literal["BUY","SHORT","WAIT"]

class SignalModel(BaseModel):
    symbol: str
    horizon: Horizon
    action: Action
    confidence: float = Field(ge=0, le=1)
    entry: float
    tp1: float
    tp2: float
    sl: float
    key_mark: float
    upper_zone: float
    lower_zone: float
    rationale: str
