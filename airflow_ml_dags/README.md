homework3
==============================

Apache Airflow 2.1.0
docker-compose.yaml based on Airflow documentation example

Before running Airflow you need to set env variables for e-mail 
alerts (yandex mail smtp with OAuth):

For Windows:

    set YA_USERNAME=*yandex_login*
    set YA_PASSWORD=*password*
    set SEND_MAIL_FROM=*e-mail for yandex_login*

For Linux:  

    export YA_USERNAME=*yandex_login*
    export YA_PASSWORD=*password*
    export SEND_MAIL_FROM=*e-mail for yandex_login*

To init, build and run Airflow use:
1. `docker-compose up airflow-init` (for first launch)
2. `docker-compose up --build`

Airflow available at `http://localhost:8080/`, to login use:
* login = `airflow`
* password = `airflow`

Before running predict DAG do next:
1. Go to `Admin` — `Variables`
2. Create new variable `prod_model_path` with value like `yyyy-mm-dd` 
   – date of creation of model which will be used for predictions

To shut-down Airflow use:
1. Press `CTRL+C` while in CLI
2. Use command `docker-compose down`

Tests
------------
