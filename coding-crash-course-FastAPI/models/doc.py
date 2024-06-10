from pydantic import BaseModel
from typing import Optional


class BaseDoc(BaseModel):
    textOCR: str


class Doc(BaseDoc):
    id: Optional[int] = None
    isCompleted: bool = False


class DocResponse(BaseDoc):
    pass

