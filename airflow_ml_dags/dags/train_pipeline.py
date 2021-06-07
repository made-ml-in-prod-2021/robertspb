import airflow.utils.dates

from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from datetime import timedelta


default_args = {
    'owner': 'airflow',
    'email': ['airflow@example.com'],
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': True,
    'email_on_retry': False,
}


with DAG(
    dag_id='train_pipeline',
    default_args=default_args,
    start_date=airflow.utils.dates.days_ago(1),
    schedule_interval='@weekly',
) as dag:

    start = DummyOperator(task_id='start')

    data_sensor = FileSensor(
        task_id='data-sensor',
        filepath='data/raw/{{ ds }}/data.csv',
        poke_interval=10,
        retries=100,
    )

    target_sensor = FileSensor(
        task_id='target-sensor',
        filepath='data/raw/{{ ds }}/target.csv',
        poke_interval=10,
        retries=100,
    )

    load = DockerOperator(
        image='airflow-load',
        command='--data_path /data/raw/{{ ds }} --save_to /data/processed/{{ ds }} --model_path /data/models/{{ ds }}',
        network_mode='bridge',
        task_id='docker-airflow-load',
        do_xcom_push=False,
        auto_remove=True,
        volumes=['/c/users/роберт/pycharmprojects/ml-in-prod-hw-1/airflow_ml_dags/data:/data'],
    )

    split = DockerOperator(
        image='airflow-split',
        command='--data_path /data/processed/{{ ds }} --save_to /data/split/{{ ds }}',
        network_mode='bridge',
        task_id='docker-airflow-split',
        do_xcom_push=False,
        auto_remove=True,
        volumes=['/c/users/роберт/pycharmprojects/ml-in-prod-hw-1/airflow_ml_dags/data:/data'],
    )

    train = DockerOperator(
        image='airflow-train',
        command='--data_path /data/split/{{ ds }} --save_to /data/models/{{ ds }}',
        network_mode='bridge',
        task_id='docker-airflow-train',
        do_xcom_push=False,
        auto_remove=True,
        volumes=['/c/users/роберт/pycharmprojects/ml-in-prod-hw-1/airflow_ml_dags/data:/data'],
    )

    validate = DockerOperator(
        image='airflow-validate',
        command='--data_path /data/split/{{ ds }} --model_path /data/models/{{ ds }}',
        network_mode='bridge',
        task_id='docker-airflow-validate',
        do_xcom_push=False,
        auto_remove=True,
        volumes=['/c/users/роберт/pycharmprojects/ml-in-prod-hw-1/airflow_ml_dags/data:/data'],
    )

    start >> [data_sensor, target_sensor] >> load >> split >> train >> validate
