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

resource "google_storage_bucket" "crypto_bucket" {
  name     = var.bucket_name
  location = var.region
}

resource "google_bigquery_dataset" "crypto_dataset" {
  dataset_id = var.dataset_id
  location   = var.region
}

resource "google_dataproc_cluster" "crypto_cluster" {
  name   = var.dataproc_cluster_name
  region = var.region

  cluster_config {
    master_config {
      num_instances    = 1
      machine_type     = "n1-standard-2"
      disk_config {
        boot_disk_size_gb = 100
      }
    }
    worker_config {
      num_instances    = 2
      machine_type     = "n1-standard-2"
      disk_config {
        boot_disk_size_gb = 100
      }
    }
  }
}
