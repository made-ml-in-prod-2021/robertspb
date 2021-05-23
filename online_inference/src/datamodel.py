from pydantic import BaseModel


class RequestDataModel(BaseModel):
    id: int
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


class ResponseDataModel(BaseModel):
    id: int
    target: int
