FROM apache/airflow:3.2.0-python3.11

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && apt-get autoremove -yqq --purge \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

RUN pip install --no-cache-dir \
    "dvc" \
    "mlflow==3.10.1" \
    "apache-airflow-providers-fab" \
    "apache-airflow-providers-docker"