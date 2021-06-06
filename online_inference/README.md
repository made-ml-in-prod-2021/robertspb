homework2
============
python 3.8

Pull and run docker
------------
`
docker pull robspb/online_inference:v1
docker run -p 8000:8000 -it robspb/online_inference:v1
`

Service API
-----------
`/` - GET, root directory
`/health` - GET, check service health
`/predict` - POST, get predictions for input data
`/docs` - fastapi built-in API documentation

Request script to API
----------
Run `make_request.py` script to make test request to `/predict`.

Data sample is provided and can be founded in `data` folder.

Test /predict
-----------
Run `python -m pytest tests` for testing `/predict` API.
