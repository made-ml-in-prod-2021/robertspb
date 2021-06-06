import airflow.utils.dates

from datetime import timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator


default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id='download',
    default_args=default_args,
    start_date=airflow.utils.dates.days_ago(1),
    schedule_interval='@daily',
) as dag:
    load_data = DockerOperator(
        image="airflow-download",
        command="--output_dir /data/raw/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-download",
        do_xcom_push=False,
        volumes=['/c/users/роберт/pycharmprojects/ml-in-prod-hw-1/airflow_ml_dags/data:/data']
    )

    load_data
