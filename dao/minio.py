import yaml
from minio import Minio

class MinioProxyClient:
    def __init__(self, conf_path="conf/app.yaml", is_https=False):
        with open(conf_path, 'r') as f:
            config = yaml.safe_load(f)
        self.minio_config = config["minio"]
        self.client = Minio(
            self.minio_config['endpoint'],
            access_key=self.minio_config['access_key'],
            secret_key=self.minio_config['secret_key'],
            secure=is_https
        )
    
    def bucket_exists(self, bucket_name):
        return self.client.bucket_exists(bucket_name)
    
    def download_file(self, bucket_name, object_name, file_path):
        if not self.bucket_exists(bucket_name):
            raise ValueError("Bucket '{}' does not exist.".format(bucket_name))
        try:
            self.client.fget_object(bucket_name, object_name, file_path)
        except Exception as e:
            raise RuntimeError("Failed to download file: {}".format(e))


