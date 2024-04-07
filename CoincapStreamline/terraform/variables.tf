variable "project_id" {
  description = "The ID of the GCP project"
  default = "cyril-njitari"
}

variable "region" {
  description = "The GCP region to use"
  default     = "europe-west1"
}

variable "zone" {
  description = "The GCP zone to use"
  default     = "europe-west1-b"
}

variable "bucket_name" {
  description = "Name of the Google Cloud Storage bucket"
  default = "crypto_bucket"
}

variable "dataset_id" {
  description = "ID of the BigQuery dataset"
  default = "crypto_dataset"
}

variable "dataproc_cluster_name" {
  description = "Name of the Dataproc cluster"
  default = "crypto_cluster"
}

variable "service_account_key_file" {
  description = "credentials.json"
  default = "credentials.json"
}

variable "service_account_email" {
  description = "L'e-mail du compte de service"
  type        = string
  default     = "mini-projet-gcp@cyril-njitari.iam.gserviceaccount.com"
}


variable "function_source_path" {
  description = "Path to the source code of the Cloud Function"
  default = "source.zip"
}