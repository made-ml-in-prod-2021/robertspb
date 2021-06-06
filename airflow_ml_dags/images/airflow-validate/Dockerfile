FROM python:3.6-slim-stretch

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache -r requirements.txt

COPY . .

ENTRYPOINT [ "python", "validate.py" ]
