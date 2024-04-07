from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from src.kafka_client.kafka_stream_data import CoinCapProducer
from src.spark_pgsql.spark_streaming import CoinCapSparkConsumer
from datetime import datetime, timedelta

def produce_data():
    producer = CoinCapProducer()
    producer.create_topic()
    data = producer.fetch_data()
    producer.publish_data(data)

def consume_data():
    consumer = CoinCapSparkConsumer()
    kafka_df = consumer.read_from_kafka()
    consumer.process_stream(kafka_df)

default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 3, 17),
    "retries": 1,
    "retry_delay": timedelta(seconds=5),
}

with DAG(
    dag_id="kafka_spark_dag_Coincap",
    default_args=default_args,
    schedule_interval=timedelta(minutes=60),
    catchup=False,
) as dag:

    kafka_stream_task = PythonOperator(
        task_id="coin_cap_producer",
        python_callable=produce_data,
        dag=dag,
    )

    spark_stream_task = DockerOperator(
        task_id="coin_cap_consumer",
        image="coincapstreamline-spark:latest",
        api_version="auto",
        auto_remove=True,
        command="spark-submit \
                    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.0.1 \
                    --jars /home/jovyan/data/postgresql-42.2.5.jar \
                    /home/jovyan/src/spark_pgsql/spark_streaming.py",
        docker_url='tcp://docker-proxy:2375',
        environment={'SPARK_LOCAL_HOSTNAME': 'localhost'},
        network_mode="airflow-kafka",
        dag=dag,
    )

    kafka_stream_task >> spark_stream_task

