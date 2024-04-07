import os
from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.operators.bigquery import BigQueryCreateExternalTableOperator
from google.cloud import storage
import pandas as pd 

PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
BUCKET = os.environ.get("GCP_GCS_BUCKET")

dataset_file = "movies.csv"
#dataset_url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{dataset_file}"
dataset_url = "https://github.com/Georgeshermann/getting_data/blob/main/movies.csv"
path_to_local_home = os.environ.get("AIRFLOW_HOME", "/opt/airflow/")
BIGQUERY_DATASET = os.environ.get("BIGQUERY_DATASET", 'trips_data_all')

def upload_to_gcs(bucket, object_name, local_file):
    """
    Ref: https://cloud.google.com/storage/docs/uploading-objects#storage-upload-object-python
    :param bucket: GCS bucket name
    :param object_name: target path & file-name
    :param local_file: source path & file-name
    :return:
    """
    # Adjustments for large files, if necessary
    # storage.blob._MAX_MULTIPART_SIZE = 5 * 1024 * 1024  # 5 MB
    # storage.blob._DEFAULT_CHUNKSIZE = 5 * 1024 * 1024  # 5 MB

    client = storage.Client()
    bucket = client.bucket(bucket)
    blob = bucket.blob(object_name)
    blob.upload_from_filename(local_file)

def process_data():
    file_path = f'{path_to_local_home}/{dataset_file}'
    movies_df = pd.read_csv(file_path)

    # Fonction pour nettoyer les champs de texte
    def clean_text(text):
        if pd.isnull(text) or isinstance(text, float):
            return None
        return text.strip().replace('\n', '').strip()

    # Fonction pour nettoyer et catégoriser le champ 'YEAR'
    def clean_year(year):
        if pd.isnull(year) or isinstance(year, float):
            return None
        year = year.strip('() ').split('–')[0]
        return year if year.isdigit() else None

    # Fonction pour diviser 'STARS' en 'DIRECTORS' et 'STARS'
    def split_directors_stars(text):
        if pd.isnull(text):
            return (None, None)
        text = clean_text(text)
        parts = text.split('|')
        director = parts[0].replace('Director:', '').strip() if 'Director:' in parts[0] else None
        stars = parts[1].replace('Stars:', '').strip() if len(parts) > 1 else None
        return (director, stars)

    # Nettoyer la colonne 'YEAR'
    movies_df['YEAR'] = movies_df['YEAR'].apply(clean_year)
    movies_df['YEAR'] = pd.Categorical(movies_df['YEAR'])

    # Nettoyer la colonne 'GENRE'
    movies_df['GENRE'] = movies_df['GENRE'].apply(clean_text)

    # Diviser 'STARS' en 'DIRECTORS' et 'STARS'
    movies_df[['DIRECTORS', 'STARS']] = movies_df['STARS'].apply(lambda x: pd.Series(split_directors_stars(x)))

    # Nettoyer les autres champs de texte
    movies_df['ONE-LINE'] = movies_df['ONE-LINE'].apply(clean_text)

    # Nettoyer la colonne 'VOTES', convertir en float et gérer les virgules et NaNs
    movies_df['VOTES'] = movies_df['VOTES'].astype(str).str.replace(',', '').apply(pd.to_numeric, errors='coerce')

    # Nettoyer la colonne 'RunTime', supposer NaNs comme 0, convertir en int
    movies_df['RunTime'] = movies_df['RunTime'].fillna(0).astype(int)

    # La colonne 'Gross' sera nettoyée en tant que chaîne; envisagez de gérer les NaNs et les conversions en fonction des besoins d'analyse ultérieurs
    movies_df['Gross'] = movies_df['Gross'].astype(str).apply(clean_text)

    # Enregistrez le DataFrame nettoyé pour une utilisation ultérieure
    processed_file_path = f'{path_to_local_home}/processed_movies.csv'
    movies_df.to_csv(processed_file_path, index=False)



default_args = {
    "owner": "airflow",
    "start_date": days_ago(1),
    "depends_on_past": False,
    "retries": 1,
}

with DAG(
    dag_id="data_github_to_gcs_dag",
    schedule_interval="@daily",
    default_args=default_args,
    catchup=False,
    max_active_runs=1,
    tags=['dtc-de'],
) as dag:

    download_dataset_task = BashOperator(
        task_id="download_dataset_task",
        bash_command=f"curl -sSL {dataset_url} > {path_to_local_home}/{dataset_file}"
    )

    process_data_task = PythonOperator(
        task_id="process_data_task",
        python_callable=process_data
    )

    local_to_gcs_task = PythonOperator(
        task_id="local_to_gcs_task",
        python_callable=upload_to_gcs,
        op_kwargs={
            "bucket": BUCKET,
            "object_name": f"raw/{dataset_file}",
            "local_file": f"{path_to_local_home}/{dataset_file}",
        },
    )

    bigquery_external_table_task = BigQueryCreateExternalTableOperator(
        task_id="bigquery_external_table_task",
        table_resource={
            "tableReference": {
                "projectId": PROJECT_ID,
                "datasetId": BIGQUERY_DATASET,
                "tableId": "external_table",
            },
            "externalDataConfiguration": {
                "sourceFormat": "CSV",
                "sourceUris": [f"gs://{BUCKET}/raw/{dataset_file}"],
            },
        },
    )

     # Définir la séquence des tâches
    download_dataset_task >> process_data_task >> local_to_gcs_task >> bigquery_external_table_task
