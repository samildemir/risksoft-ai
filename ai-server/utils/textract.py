import boto3
import time
import requests
from typing import Dict, List, Optional, Union
from botocore.exceptions import ClientError
import logging
from constants.env_variables import AWS_REGION, TEXT_EXTRACT_ACCESS_KEY, TEXT_EXTRACT_SECRET_ACCESS
from utils.s3Handler import S3Handler
import os
logger = logging.getLogger(__name__)

class AWSTextractHandler:
    """
    A comprehensive Handler class for AWS Textract operations.
    Supports text extraction from PDF, Word, Excel, PowerPoint documents stored in S3 or from URLs.
    """
    
    SUPPORTED_FORMATS = {
        'pdf': ['pdf'],
        'word': ['doc', 'docx'],
        'excel': ['xls', 'xlsx'],
        'powerpoint': ['ppt', 'pptx']
    }
    
    def __init__(self):
        """
        Initialize AWS Textract client with credentials.
        Uses environment variables by default, but can be overridden with parameters.
        """
        self.textract_client = boto3.client(
            'textract',
            aws_access_key_id=TEXT_EXTRACT_ACCESS_KEY,
            aws_secret_access_key=TEXT_EXTRACT_SECRET_ACCESS,
            region_name=AWS_REGION
        )
        self.s3_handler = S3Handler()
        self.bucket_name = self.s3_handler.get_bucket_name()
        self.confidence_threshold = max(0.0, min(100.0, 0.0))

    def _download_from_url(self, url: str) -> bytes:
        """Download document from URL."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"Error downloading document from URL: {str(e)}")
            raise

    def _upload_to_s3(self, file_content: bytes, file_name: str) -> str:
        """Upload document to S3 and return the key."""
        try:
            self.s3_handler.upload_file(file_name, file_content.decode('utf-8'))
            return file_name
        except Exception as e:
            logger.error(f"Error uploading to S3: {str(e)}")
            raise

    def _get_file_extension(self, file_path: str) -> str:
        """Get file extension from the file path."""
        return file_path.split('.')[-1].lower()

    def _is_supported_format(self, file_extension: str) -> bool:
        """Check if the file format is supported."""
        return any(file_extension in formats for formats in self.SUPPORTED_FORMATS.values())

    def _extract_text_from_response(self, response: Dict) -> str:
        """
        Extract text from Textract response with confidence threshold.
        """
        text_list = []
        for item in response.get('Blocks', []):
            if (item.get('BlockType') == 'LINE' and 
                item.get('Confidence', 0) >= self.confidence_threshold):
                text_list.append(item.get('Text', ''))
        return '\n'.join(text_list)

    def _wait_for_job_completion(self, job_id: str) -> Dict:
        """Wait for the Textract job to complete and return results."""
        while True:
            response = self.textract_client.get_document_text_detection(JobId=job_id)
            status = response['JobStatus']
            
            if status in ['SUCCEEDED', 'FAILED']:
                if status == 'FAILED':
                    raise Exception(f"Textract job failed: {response.get('StatusMessage')}")
                return response
            
            time.sleep(5)

    def _get_all_job_results(self, job_id: str) -> List[Dict]:
        """Get all pages of results from an async Textract job."""
        pages = []
        next_token = None
        
        while True:
            if next_token:
                response = self.textract_client.get_document_text_detection(
                    JobId=job_id,
                    NextToken=next_token
                )
            else:
                response = self.textract_client.get_document_text_detection(JobId=job_id)
            
            pages.append(response)
            
            next_token = response.get('NextToken')
            if not next_token:
                break
        
        return pages

    def extract_text_from_s3(self, document_key: str) -> Union[str, Dict]:
        """
        Extract text from a document stored in S3.
        
        Args:
            document_key (str): Key of the document in S3
            bucket_name (str, optional): Name of the S3 bucket. If not provided, uses default bucket
            save_raw_response (bool): If True, returns the raw Textract response
        
        Returns:
            Union[str, Dict]: Extracted text or raw Textract response
        """
        try:
            file_extension = self._get_file_extension(document_key)
            
            if not self._is_supported_format(file_extension):
                raise ValueError(f"Unsupported file format: {file_extension}")

            # Start async text detection job
            response = self.textract_client.start_document_text_detection(
                DocumentLocation={
                    'S3Object': {
                        'Bucket':  self.bucket_name,
                        'Name': document_key
                    }
                }
            )
            
            job_id = response['JobId']
            logger.info(f"Started Textract job {job_id} for document {document_key}")
            
            # Wait for job completion
            self._wait_for_job_completion(job_id)
            
            # Get all results
            all_pages = self._get_all_job_results(job_id)
            
           
            # Extract text from all pages
            extracted_text = []
            for page in all_pages:
                extracted_text.append(self._extract_text_from_response(page))


            return '\n\n'.join(extracted_text)

        except ClientError as e:
            logger.error(f"AWS error occurred: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error occurred: {str(e)}")
            raise

    def get_document_analysis(self, document_key: str,
                            bucket_name: Optional[str] = None,
                            features: List[str] = None) -> Dict:
        """
        Perform advanced document analysis using Textract.
        
        Args:
            document_key (str): Key of the document in S3
            bucket_name (str, optional): Name of the S3 bucket. If not provided, uses default bucket
            features (List[str]): List of features to analyze. 
                                Options: TABLES, FORMS, QUERIES
        
        Returns:
            Dict: Analysis results
        """
        if features is None:
            features = ['TABLES', 'FORMS']

        try:
            bucket_name = bucket_name or self.bucket_name
            response = self.textract_client.start_document_analysis(
                DocumentLocation={
                    'S3Object': {
                        'Bucket': bucket_name,
                        'Name': document_key
                    }
                },
                FeatureTypes=features
            )
            
            job_id = response['JobId']
            logger.info(f"Started document analysis job {job_id} for document {document_key}")
            
            # Wait for completion
            while True:
                response = self.textract_client.get_document_analysis(JobId=job_id)
                status = response['JobStatus']
                
                if status in ['SUCCEEDED', 'FAILED']:
                    if status == 'FAILED':
                        raise Exception(f"Document analysis failed: {response.get('StatusMessage')}")
                    break
                
                time.sleep(5)
            
            # Get all results
            all_results = []
            next_token = None
            
            while True:
                if next_token:
                    response = self.textract_client.get_document_analysis(
                        JobId=job_id,
                        NextToken=next_token
                    )
                else:
                    response = self.textract_client.get_document_analysis(JobId=job_id)
                
                all_results.append(response)
                next_token = response.get('NextToken')
                
                if not next_token:
                    break
            
            return {
                'job_id': job_id,
                'results': all_results
            }

        except ClientError as e:
            logger.error(f"AWS error occurred: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error occurred: {str(e)}")
            raise

    def extract_text_from_url(self, document_url: str,
                            save_raw_response: bool = False,
                            delete_after: bool = True) -> Union[str, Dict]:
        """
        Extract text from a document available at a URL.
        
        Args:
            document_url (str): URL of the document
            save_raw_response (bool): If True, returns the raw Textract response
            delete_after (bool): If True, deletes the document from S3 after processing
            
        Returns:
            Union[str, Dict]: Extracted text or raw Textract response
        """
        try:
            # Download document from URL
            file_content = self._download_from_url(document_url)
            
            # Generate a unique filename
            file_name = f"temp/{int(time.time())}_{document_url.split('/')[-1]}"
            
            # Upload to S3
            s3_key = self._upload_to_s3(file_content, file_name)
            
            try:
                # Extract text using existing S3 method
                result = self.extract_text_from_s3(s3_key, save_raw_response=save_raw_response)
                return result
            finally:
                # Clean up the temporary file if requested
                if delete_after:
                    try:
                        self.s3_handler.delete_object(s3_key)
                    except Exception as e:
                        logger.warning(f"Failed to delete temporary file: {str(e)}")

        except Exception as e:
            logger.error(f"Error processing document from URL: {str(e)}")
            raise

    def extract_text_with_confidence(self, source: str,
                                   confidence_threshold: Optional[float] = None,
                                   is_url: bool = False) -> Dict[str, Union[str, float, List[Dict]]]:
        """
        Extract text with confidence scores and detailed information.
        
        Args:
            source (str): S3 key or URL of the document
            confidence_threshold (float, optional): Override default confidence threshold
            is_url (bool): Whether the source is a URL
            
        Returns:
            Dict with keys:
                - text (str): Extracted text
                - average_confidence (float): Average confidence score
                - details (List[Dict]): Detailed information about each text block
        """
        try:
            # Save original threshold and set new one if provided
            original_threshold = self.confidence_threshold
            if confidence_threshold is not None:
                self.confidence_threshold = max(0.0, min(100.0, confidence_threshold))

            try:
                # Process document based on source type
                if is_url:
                    raw_result = self.extract_text_from_url(source, save_raw_response=True)
                else:
                    raw_result = self.extract_text_from_s3(source, save_raw_response=True)

                # Extract detailed information
                details = []
                total_confidence = 0
                block_count = 0

                for page in raw_result['pages']:
                    for block in page.get('Blocks', []):
                        if block.get('BlockType') == 'LINE':
                            confidence = block.get('Confidence', 0)
                            if confidence >= self.confidence_threshold:
                                details.append({
                                    'text': block.get('Text', ''),
                                    'confidence': confidence,
                                    'page': page.get('Page', 1),
                                    'geometry': block.get('Geometry', {})
                                })
                                total_confidence += confidence
                                block_count += 1

                # Calculate average confidence
                avg_confidence = total_confidence / block_count if block_count > 0 else 0

                # Combine text from filtered blocks
                text = '\n'.join(detail['text'] for detail in details)

                return {
                    'text': text,
                    'average_confidence': avg_confidence,
                    'details': details
                }

            finally:
                # Restore original threshold
                self.confidence_threshold = original_threshold

        except Exception as e:
            logger.error(f"Error in text extraction with confidence: {str(e)}")
            raise
