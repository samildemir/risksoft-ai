import boto3
import logging
from constants.env_variables import AWS_REGION, S3_ACCESS_KEY, S3_SECRET_ACCESS, BUCKET_NAME

class BotoHandler():
    def __init__(self,service_name="s3",aws_access_key_id=S3_ACCESS_KEY,aws_secret_access_key=S3_SECRET_ACCESS,region_name=AWS_REGION):
        self.client = boto3.client(
            service_name, 
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )



        self.session = boto3.Session(
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_ACCESS,
            region_name=AWS_REGION
        )

    def upload_file(self, path: str, data: str) -> bool:
        try:
            print("Uploading file")
            self.client.put_object(Bucket=BUCKET_NAME, Key=path, Body=data)
            return True
        except Exception as e:
            logging.error(f"Error in upload_file function: {e}")
            return False

    def get_bucket_name(self):
        return BUCKET_NAME

   

    def get_client(self):
        return self.client
    def get_session(self):
        return self.session


    def delete_object(self, key):
        try:
            self.client.delete_object(Bucket=self.get_bucket_name(), Key=key)
        except Exception as e:
            logging.error(f"Error in delete_object function: {e}")
    def check_if_file_exists(self, key):
        try:
            return  self.client.head_object(Bucket=self.get_bucket_name(), Key=key)
        except Exception as e:
            logging.error(f"Error in check_if_file_exists function: {e}")
            return False
    def boto_get_object(self, key):
        try:
            response = self.client.get_object(Bucket=self.get_bucket_name(), Key=key)
            return response['Body'].read()
        except Exception as e:
            logging.error(f"Error in s3_get_object function: {e}")
            return None

    def boto_get_object_list(self, prefix):
        try:
            response = self.client.list_objects_v2(Bucket=self.get_bucket_name(), Prefix=prefix)
            return response.get('Contents', [])
        except Exception as e:
            logging.error(f"Error in s3_get_object_list function: {e}")
            return []
