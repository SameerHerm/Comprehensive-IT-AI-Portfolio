"""
Cloud Storage Loader Module
Loads data to cloud storage services
"""

import pandas as pd
import logging
from typing import Dict, Any, Optional
import boto3
from azure.storage.blob import BlobServiceClient
from google.cloud import storage as gcs
import io
import os

logger = logging.getLogger(__name__)

class CloudLoader:
    """Load data to cloud storage services"""
    
    def __init__(self, provider: str, config: Dict[str, Any]):
        """Initialize cloud loader"""
        self.provider = provider.lower()
        self.config = config
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize cloud storage client"""
        if self.provider == 's3' or self.provider == 'aws':
            return self._init_s3_client()
        elif self.provider == 'azure':
            return self._init_azure_client()
        elif self.provider == 'gcs' or self.provider == 'gcp':
            return self._init_gcs_client()
        else:
            raise ValueError(f"Unsupported cloud provider: {self.provider}")
    
    def _init_s3_client(self):
        """Initialize AWS S3 client"""
        return boto3.client(
            's3',
            aws_access_key_id=self.config.get('access_key_id'),
            aws_secret_access_key=self.config.get('secret_access_key'),
            region_name=self.config.get('region', 'us-east-1')
        )
    
    def _init_azure_client(self):
        """Initialize Azure Blob Storage client"""
        return BlobServiceClient(
            account_url=f"https://{self.config['account_name']}.blob.core.windows.net",
            credential=self.config.get('account_key')
        )
    
    def _init_gcs_client(self):
        """Initialize Google Cloud Storage client"""
        if 'credentials_path' in self.config:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.config['credentials_path']
        
        return gcs.Client(project=self.config.get('project_id'))
    
    def load_to_s3(self, data: pd.DataFrame, bucket: str, key: str, format: str = 'csv'):
        """Load data to AWS S3"""
        logger.info(f"Loading data to s3://{bucket}/{key}")
        
        # Convert dataframe to bytes
        if format == 'csv':
            buffer = io.StringIO()
            data.to_csv(buffer, index=False)
            content = buffer.getvalue()
        elif format == 'json':
            content = data.to_json(orient='records')
        elif format == 'parquet':
            buffer = io.BytesIO()
            data.to_parquet(buffer, index=False)
            content = buffer.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Upload to S3
        self.client.put_object(
            Bucket=bucket,
            Key=key,
            Body=content,
            ContentType=self._get_content_type(format)
        )
        
        logger.info(f"Successfully uploaded to s3://{bucket}/{key}")
    
    def load_to_azure(self, data: pd.DataFrame, container: str, blob_name: str, format: str = 'csv'):
        """Load data to Azure Blob Storage"""
        logger.info(f"Loading data to Azure container: {container}/{blob_name}")
        
        # Get container client
        container_client = self.client.get_container_client(container)
        
        # Convert dataframe to bytes
        if format == 'csv':
            content = data.to_csv(index=False)
        elif format == 'json':
            content = data.to_json(orient='records')
        elif format == 'parquet':
            buffer = io.BytesIO()
            data.to_parquet(buffer, index=False)
            content = buffer.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Upload to Azure
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(content, overwrite=True)
        
        logger.info(f"Successfully uploaded to Azure: {container}/{blob_name}")
    
    def load_to_gcs(self, data: pd.DataFrame, bucket_name: str, blob_name: str, format: str = 'csv'):
        """Load data to Google Cloud Storage"""
        logger.info(f"Loading data to gs://{bucket_name}/{blob_name}")
        
        # Get bucket
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        # Convert dataframe to bytes
        if format == 'csv':
            content = data.to_csv(index=False)
        elif format == 'json':
            content = data.to_json(orient='records')
        elif format == 'parquet':
            buffer = io.BytesIO()
            data.to_parquet(buffer, index=False)
            content = buffer.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Upload to GCS
        blob.upload_from_string(content, content_type=self._get_content_type(format))
        
        logger.info(f"Successfully uploaded to gs://{bucket_name}/{blob_name}")
    
    def load(self, data: pd.DataFrame, path: str, format: str = 'csv'):
        """Generic load method"""
        if self.provider in ['s3', 'aws']:
            # Parse S3 path
            parts = path.replace('s3://', '').split('/', 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ''
            self.load_to_s3(data, bucket, key, format)
            
        elif self.provider == 'azure':
            # Parse Azure path
            parts = path.split('/', 1)
            container = parts[0]
            blob_name = parts[1] if len(parts) > 1 else ''
            self.load_to_azure(data, container, blob_name, format)
            
        elif self.provider in ['gcs', 'gcp']:
            # Parse GCS path
            parts = path.replace('gs://', '').split('/', 1)
            bucket = parts[0]
            blob_name = parts[1] if len(parts) > 1 else ''
            self.load_to_gcs(data, bucket, blob_name, format)
    
    def _get_content_type(self, format: str) -> str:
        """Get content type for file format"""
        content_types = {
            'csv': 'text/csv',
            'json': 'application/json',
            'parquet': 'application/octet-stream',
            'txt': 'text/plain',
            'xml': 'application/xml'
        }
        return content_types.get(format, 'application/octet-stream')
    
    def load_partitioned(self, data: pd.DataFrame, base_path: str,
                        partition_cols: list, format: str = 'parquet'):
        """Load partitioned data to cloud storage"""
        logger.info(f"Loading partitioned data to {base_path}")
        
        for partition_values, group_df in data.groupby(partition_cols):
            if not isinstance(partition_values, tuple):
                partition_values = (partition_values,)
            
            # Create partition path
            partition_path = base_path
            for col, val in zip(partition_cols, partition_values):
                partition_path = f"{partition_path}/{col}={val}"
            
            # Save partition
            file_path = f"{partition_path}/data.{format}"
            self.load(group_df, file_path, format)
        
        logger.info(f"Partitioned data loaded to {base_path}")
