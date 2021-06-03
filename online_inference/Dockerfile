FROM python:3.8-slim-buster

COPY . .

RUN pip install -r requirements.txt

ENV MODEL_PATH="models/model.pkl"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
