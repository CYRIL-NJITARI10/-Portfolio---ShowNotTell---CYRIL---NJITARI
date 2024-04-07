from google.cloud import storage


class GCSHandler:
    def __init__(self, bucket_name):
        self.bucket_name = bucket_name
        self.client = storage.Client()

    def upload_file(self, local_file_path, remote_file_path):
        bucket = self.client.bucket(self.bucket_name)
        blob = bucket.blob(remote_file_path)
        blob.upload_from_filename(local_file_path)
