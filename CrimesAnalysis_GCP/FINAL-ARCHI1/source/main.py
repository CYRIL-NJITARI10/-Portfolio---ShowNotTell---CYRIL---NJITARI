from google.cloud import dataproc_v1
import base64
import json
import logging

def submit_dataproc_spark_job(event, context):
    """
    Background Cloud Function to be triggered by Pub/Sub.
    This function submits a job to Cloud Dataproc.

    Args:
        event (dict): Event payload.
        context (google.cloud.functions.Context): Metadata for the event.
    """
    pubsub_message = base64.b64decode(event['data']).decode('utf-8')
    message = json.loads(pubsub_message)

    project_id = message['project_id']
    region = message['region']
    cluster_name = message['cluster_name']
    bucket_name = message['bucket_name']
    main_python_file_uri = f'gs://{bucket_name}/{message["py_file"]}'

    # Configure the job
    job_config = {
        "placement": {"cluster_name": cluster_name},
        "pyspark_job": {
            "main_python_file_uri": main_python_file_uri
            # You can add args or other configuration as needed.
        }
    }

    # Create a Dataproc job client
    job_client = dataproc_v1.JobControllerClient(client_options={
        "api_endpoint": f"{region}-dataproc.googleapis.com:443"
    })

    # Submit the job to the cluster
    result = job_client.submit_job(
        request={"project_id": project_id, "region": region, "job": job_config}
    )
    job_id = result.reference.job_id

    logging.info(f'Submitted job ID "{job_id}" to cluster "{cluster_name}"')

    return f"Job {job_id} submitted to cluster {cluster_name}"

