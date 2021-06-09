import airflow.utils.dates
import os

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from datetime import timedelta


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
    dag_id='download',
    default_args=default_args,
    start_date=airflow.utils.dates.days_ago(1),
    schedule_interval='@daily',
) as dag:
    load_data = DockerOperator(
        image='airflow-download',
        command='--output_dir /data/raw/{{ ds }}',
        network_mode='bridge',
        task_id='docker-airflow-download',
        do_xcom_push=False,
        volumes=['/c/users/роберт/pycharmprojects/ml-in-prod-hw-1/airflow_ml_dags/data:/data']
    )

    load_data
