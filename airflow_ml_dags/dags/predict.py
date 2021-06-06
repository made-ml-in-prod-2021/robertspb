import airflow.utils.dates

from airflow import DAG
from airflow.models import Variable
from airflow.providers.docker.operators.docker import DockerOperator
from datetime import timedelta


DATA_PATH = '/data/raw/{{ ds }}'
MODEL_PATH = Variable.get('prod_model_path')
SAVE_TO_PATH = '/data/predictions/{{ ds }}'

default_args = {
    'owner': 'airflow',
    'email': ['airflow@example.com'],
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='predict',
    default_args=default_args,
    start_date=airflow.utils.dates.days_ago(1),
    schedule_interval='@daily',
) as dag:
    predict = DockerOperator(
        image='airflow-predict',
        command=f'--data_path {DATA_PATH} --model_path /data/models/{MODEL_PATH} --save_to {SAVE_TO_PATH}',
        network_mode='bridge',
        task_id='predict',
        do_xcom_push=False,
        auto_remove=True,
        volumes=['/c/users/роберт/pycharmprojects/ml-in-prod-hw-1/airflow_ml_dags/data:/data'],
    )

    predict
