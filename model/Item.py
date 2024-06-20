from pydantic import BaseModel

class Item(BaseModel):
    userId: str
    timestamp: int
    observation: str
    importance: float
    isEventScene: bool
    reasonIds: str