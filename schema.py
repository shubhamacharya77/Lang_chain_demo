from pydantic import BaseModel,Field

class StudentInput(BaseModel):
    name:str
    marks:dict
class StudentMarks(BaseModel):
    name:str
    result:float
class passOrfail(BaseModel):
    name:str
    result:str