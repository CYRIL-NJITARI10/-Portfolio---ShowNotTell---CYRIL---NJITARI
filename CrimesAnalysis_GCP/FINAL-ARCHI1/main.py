from google.cloud import bigquery
from google.cloud import storage
import os

def is_data_ready(bucket_name, storage_client):
    for blob in storage_client.list_blobs(bucket_name):
        if blob.name.endswith('_SUCCESS_END'):
            return True
    return False

def sort_question2_table(client, table_ref):
    query = f"""
    CREATE OR REPLACE TABLE `{table_ref.dataset_id}.{table_ref.table_id}` AS
    SELECT * FROM `{table_ref.dataset_id}.{table_ref.table_id}`
    ORDER BY Year DESC
    """
    query_job = client.query(query)
    query_job.result()  
    print(f"Table {table_ref.table_id} sorted for question 2 logic")

def delete_duplicates(client, table_ref):
    query = f"""
    CREATE OR REPLACE TABLE `{table_ref.dataset_id}.{table_ref.table_id}` AS
    SELECT DISTINCT * FROM `{table_ref.dataset_id}.{table_ref.table_id}`
    """
    query_job = client.query(query)
    query_job.result()
    print(f"Duplicates removed from {table_ref.table_id}")

def sort_table(client, table_ref, sort_column):
    query = f"""
    CREATE OR REPLACE TABLE `{table_ref.dataset_id}.{table_ref.table_id}` AS
    SELECT * FROM `{table_ref.dataset_id}.{table_ref.table_id}`
    ORDER BY {sort_column} DESC
    """
    query_job = client.query(query)
    query_job.result()
    print(f"Table {table_ref.table_id} sorted by {sort_column} in descending order")

def create_bigquery_table(event, context):
    project_id = os.getenv('PROJECT_ID')
    dataset_id = os.getenv('BIGQUERY_DATASET')
    bucket_name = event['bucket']

    storage_client = storage.Client()
    if not is_data_ready(bucket_name, storage_client):
        print(f"Data not ready (no _SUCCESS_END file found in {bucket_name} bucket).")
        return

    client = bigquery.Client(project=project_id)
    dataset_ref = client.dataset(dataset_id)
    job_config = bigquery.LoadJobConfig()
    job_config.source_format = bigquery.SourceFormat.PARQUET
    job_config.autodetect = True
    
    unique_folders = set()

    for blob in storage_client.list_blobs(bucket_name):
        if blob.name.endswith('.parquet'):
            folder_name = blob.name.split('/')[0]
            unique_folders.add(folder_name)

    sort_columns = {
        "question1_output": "Month",
        #"question2_output": "TotalThefts",
        "question3_output": "Year",
        "question4_output": "TotalCrimes",
        "question5_output": "TotalArrests"
    }

    for folder in unique_folders:
        table_name = folder.split('.')[0]
        table_ref = dataset_ref.table(table_name)
        uri = f"gs://{bucket_name}/{folder}/*.parquet"

        try:
            load_job = client.load_table_from_uri(uri, table_ref, job_config=job_config)
            print(f"Starting job {load_job.job_id} for table {table_name}")
            load_job.result()
            print(f"Job finished for table {table_name}.")

            delete_duplicates(client, table_ref)
            if table_name == "question2_output":
                sort_question2_table(client, table_ref)
            else:
                if table_name in sort_columns:
                    sort_table(client, table_ref, sort_columns[table_name])

        except Exception as e:
            print(f"An error occurred while processing table {table_name}: {e}")
