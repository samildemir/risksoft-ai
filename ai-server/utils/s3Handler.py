import s3fs
import magic
import logging
from constants.env_variables import  S3_ACCESS_KEY, S3_SECRET_ACCESS, BUCKET_NAME

class S3Handler():
    def __init__(self):
        self.s3_client = s3fs.S3FileSystem(
            anon=False,
            key=S3_ACCESS_KEY,
            secret=S3_SECRET_ACCESS,
        )
        self.magic = magic.Magic(mime=True)

    def upload_file(self, path: str, data: str) -> bool:
        try:
            with self.s3_client.open(f"{BUCKET_NAME}/{path}", 'wb') as f:
                f.write(data.encode("utf-8"))
            return True
        except Exception as e:
            logging.error(f"Error in upload_file function: {e}")
            return False

    def upload_json(self, path: str, json_data: dict) -> bool:
        """
        JSON verisini S3'e yükler
        
        Args:
            path (str): S3'teki dosya yolu
            json_data (dict): Yüklenecek JSON verisi
            
        Returns:
            bool: İşlem başarılı ise True, değilse False
        """
        try:
            import json
            json_str = json.dumps(json_data)
            with self.s3_client.open(f"{BUCKET_NAME}/{path}", 'w') as f:
                f.write(json_str)
            return True
        except Exception as e:
            logging.error(f"Error in upload_json function: {e}")
            return False

    def get_bucket_name(self):
        return BUCKET_NAME

    def get_filesystem(self):
        return self.s3_client
    
    def check_if_file_exists(self, key):
        return self.s3_client.exists(f"{BUCKET_NAME}/{key}")

    def delete_object(self, key):
        try:

            self.s3_client.rm(f"{BUCKET_NAME}/{key}")
        except Exception as e:
            logging.error(f"Error in delete_object function: {e}")

    def s3_get_object(self, key):
        try:
            with self.s3_client.open(f"{BUCKET_NAME}/{key}", 'rb') as f:
                content = f.read()
                return content.decode("utf-8")
        except Exception as e:
            logging.error(f"Error in s3_get_object function: {e}")
            return None

    def s3_get_object_list(self, prefix):
        try:
            return self.s3_client.ls(f"{BUCKET_NAME}/{prefix}")
        except Exception as e:
            logging.error(f"Error in s3_get_object_list function: {e}")
            return []
