import pytest

from fastapi.testclient import TestClient

from ..app import app, load_model


test_json_data = '[{"id":0,"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1},' \
                 '{"id":1,"age":37,"sex":1,"cp":2,"trestbps":130,"chol":250,"fbs":0,"restecg":1,"thalach":187,"exang":0,"oldpeak":3.5,"slope":0,"ca":0,"thal":2}]'


client = TestClient(app)


@pytest.fixture(scope="session", autouse=True)
def get_model():
    load_model()


def test_predict():
    response = client.post(
        "/predict",
        data=test_json_data,
    )
    assert response.status_code == 200
    assert len(response.json()) == 2
    assert response.json() == [
        {'id': 0, 'target': 1},
        {'id': 1, 'target': 1}
    ]
