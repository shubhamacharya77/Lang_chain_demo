from pydantic import BaseModel,Field
from typing import Literal
class StudentInput(BaseModel):
    name:str
    marks:dict
class StudentMarks(BaseModel):
    name:str
    result:float
class passOrfail(BaseModel):
    name:str
    result:str

class Sentiment(BaseModel):
    sentiment:Literal["positive","negative"]