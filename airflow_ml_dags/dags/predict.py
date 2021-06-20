import airflow.utils.dates
import os

from airflow import DAG
from airflow.models import Variable
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from datetime import timedelta


DATA_PATH = '/data/raw/{{ ds }}'
MODEL_PATH = Variable.get('prod_model_path')
SAVE_TO_PATH = '/data/predictions/{{ ds }}'
MAIL_FROM = 'AIRFLOW__SMTP__SMTP_MAIL_FROM'
if MAIL_FROM in dict(os.environ):
    MAIL = os.environ['AIRFLOW__SMTP__SMTP_MAIL_FROM']
else:
    MAIL = 'example@example.com'

default_args = {
    'owner': 'airflow',
    'email': [MAIL],
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': True,
    'email_on_retry': False,
}

with DAG(
    dag_id='predict',
    default_args=default_args,
    start_date=airflow.utils.dates.days_ago(1),
    schedule_interval='@daily',
) as dag:

    start = DummyOperator(task_id='start-predict')

    data_sensor = FileSensor(
        task_id='predict-data-sensor',
        filepath='data/raw/{{ ds }}/data.csv',
        poke_interval=10,
        retries=100,
    )

    model_sensor = FileSensor(
        task_id='model-sensor',
        filepath=f'data/models/{MODEL_PATH}/model.pkl',
        poke_interval=10,
        retries=100,
    )

    transformer_sensor = FileSensor(
        task_id='transformer-sensor',
        filepath='data/models/{{ ds }}/transformer.pkl',
        poke_interval=10,
        retries=100,
    )

    predict = DockerOperator(
        image='airflow-predict',
        command=f'--data_path {DATA_PATH} --model_path /data/models/{MODEL_PATH} --save_to {SAVE_TO_PATH}',
        network_mode='bridge',
        task_id='predict',
        do_xcom_push=False,
        auto_remove=True,
        volumes=['/c/users/роберт/pycharmprojects/ml-in-prod-hw-1/airflow_ml_dags/data:/data'],
    )

    start >> [data_sensor, model_sensor, transformer_sensor] >> predict
