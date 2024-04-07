# Add your settings here
api_key = "API_KEY"
bucket_name = "crypto_bucket"
local_file_path = "LOCAL_FILE_PATH"
remote_file_path = "REMOTE_FILE_PATH"

# GCP Configuration
PROJECT_ID = "YOUR_PROJECT_ID"
REGION = "europe-west1"
CLUSTER_NAME = "crypto_cluster"

# Dataproc Configuration
CLUSTER_CONFIG = {
    "master_config": {
        "num_instances": 1,
        "machine_type_uri": "n1-standard-2",
        "disk_config": {"boot_disk_type": "pd-standard", "boot_disk_size_gb": 100},
    },
    "worker_config": {
        "num_instances": 2,
        "machine_type_uri": "n1-standard-2",
        "disk_config": {"boot_disk_type": "pd-standard", "boot_disk_size_gb": 100},
    }
}

PYSPARK_JOB = {
    "reference": {"project_id": PROJECT_ID},
    "placement": {"cluster_name": CLUSTER_NAME},
    "pyspark_job": {"main_python_file_uri": "gs://data-bucket-crypto/URI/pyspark_transformation.py"}
}
