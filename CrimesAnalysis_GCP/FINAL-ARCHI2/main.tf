terraform {
  required_version = ">= 1.0"
  backend "local" {} 
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 4.34.0"
    }
  }
}

provider "google" {
  credentials = file(var.service_account_key_file)
  project     = var.project_id
  region      = var.region
  zone        = var.zone
}

resource "google_storage_bucket" "crimes_data_bucket" {
  name     = var.bucket_name
  location = var.region
}

resource "google_storage_bucket" "crimes_data_bucket_output" {
  name     = var.out_bucket
  location = var.region
}

resource "google_bigquery_dataset" "crimes_dataset" {
  dataset_id = var.dataset_id
  location   = var.region
}

resource "google_dataproc_cluster" "crimes_cluster" {
  name   = var.dataproc_cluster_name
  region = var.region

  cluster_config {   
    software_config {
      image_version = "1.5-debian10"
      override_properties = {
        "spark:spark.jars.packages" = "com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.21.0"
      }
    }
    master_config {
      num_instances    = 1
      machine_type     = "n1-standard-2"
      disk_config {
        boot_disk_size_gb = 30  
      }
    }
    worker_config {
      num_instances    = 2
      machine_type     = "n1-standard-2"
      disk_config {
        boot_disk_size_gb = 30  
      }
    }
    gce_cluster_config {
      service_account = var.service_account_email
      // Spécifiez les scopes si nécessaire
      service_account_scopes = [
        "https://www.googleapis.com/auth/cloud-platform",
      ]
    }
  }
}

resource "google_storage_bucket" "cloud_function_source" {
  name     = "${var.bucket_name}-function-source2"
  location = var.region
}   

resource "google_storage_bucket_object" "function_source" {
  name   = "source2.zip"
  bucket = google_storage_bucket.cloud_function_source.name
  source = "source2.zip"
} 


resource "google_pubsub_topic" "dataproc_job_trigger_spark" {
  name = "dataproc_job_trigger_spark_2"
}

resource "google_cloudfunctions2_function" "submit_dataproc_job_final" {
  name        = "submit-dataproc-job_a2"
  description = "Dataproc Job Submission Function"
  location    = var.region

  build_config {
    entry_point = "submit_dataproc_spark_job_2"
    runtime     = "python39"
    source {
      storage_source {
        bucket = google_storage_bucket.cloud_function_source.name
        object = google_storage_bucket_object.function_source.name
      }
    }
  }

  service_config {
    available_memory = "256M"
    service_account_email = var.service_account_email
    environment_variables = {
      BUCKET_NAME      = var.bucket_name
      DATAPROC_CLUSTER = var.dataproc_cluster_name
      PROJECT_ID       = var.project_id
      REGION           = var.region
    }
  }

  event_trigger {
    event_type   = "google.cloud.pubsub.topic.v1.messagePublished"
    pubsub_topic = "projects/${var.project_id}/topics/${google_pubsub_topic.dataproc_job_trigger_spark.name}"
  }
}


