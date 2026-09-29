from celery.utils.serialization import strtobool
from pydantic import BaseModel

class Citations(BaseModel):
    chunk : int
    source : str
    title : str

class RAGResponse(BaseModel):
    answer : str
    citations : list[int]