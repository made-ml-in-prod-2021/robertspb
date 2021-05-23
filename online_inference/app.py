import logging
import os
import uvicorn

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from typing import List

from src.main import load_model_from_path, get_predictions
from src.datamodel import RequestDataModel, ResponseDataModel


logger = logging.getLogger(__name__)

app = FastAPI()


@app.on_event('startup')
def log_startup():
    logger.info('Running application...')


@app.on_event('startup')
def load_model():
    path = os.path.abspath(os.getenv('MODEL_PATH', 'models/model.pkl'))
    logger.info(f'Loading model from path: {path}')
    try:
        model = load_model_from_path(path)
    except FileNotFoundError as error:
        logger.error(error)
        return
    global classifier, transformer
    classifier = model['model']
    transformer = model['transformer']


@app.on_event('shutdown')
def log_shutdown():
    logger.info('Application shutdown')


@app.get('/', response_class=HTMLResponse)
def main():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Heart disease predictor</title>
    </head>
    <body>
        <h1>Heart disease predictor</h1>
        <p>go to <a href="/health">/health</a> to check if REST API works</p>
        <p>make a post request to <a href="/predict">/predict</a> to get predictions</p>
        <p>or go to <a href="/docs">/docs</a> to read API documentation</p>
    </body>
    </html>
    """


@app.get('/health')
def health() -> bool:
    return classifier is not None and \
           transformer is not None


@app.post('/predict', response_model=List[ResponseDataModel])
def predict(request: List[RequestDataModel]):
    if not health():
        logger.error('Model or transformer is not loaded')
        raise HTTPException(status_code=500, detail='Model or transformer is not loaded.')
    return get_predictions(request, classifier, transformer)


if __name__ == '__main__':
    uvicorn.run('app:app', host='127.0.0.1', port=os.getenv('PORT', 8000))
