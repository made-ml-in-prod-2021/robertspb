import pytest

from airflow.models import DagBag
from pathlib import Path


DAG_PATH = Path(__file__).parent.parent / 'dags'


@pytest.fixture()
def dag_bag():
    return DagBag(dag_folder=DAG_PATH, include_examples=False)


def test_dag_bag(dag_bag):
    assert 3 == len(dag_bag.dags)
    assert 0 == len(dag_bag.import_errors)


def test_dag_download(dag_bag):
    assert 'download' in dag_bag.dags

    tasks_num = len(dag_bag.bags['download'].tasks)
    assert 1 == tasks_num


def test_dag_train_pipeline(dag_bag):
    assert 'train_pipeline' in dag_bag.dags

    tasks_num = len(dag_bag.bags['train_pipeline'].tasks)
    assert 7 == tasks_num


def test_dag_predict(dag_bag):
    assert 'predict' in dag_bag.dags

    tasks_num = len(dag_bag.bags['train_pipeline'].tasks)
    assert 5 == tasks_num
