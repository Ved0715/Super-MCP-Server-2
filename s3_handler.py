#!/usr/bin/env python3
"""
S3 Handler for Image Storage
Handles all S3 operations for storing and retrieving images from AWS S3
"""

import os
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
import mimetypes

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from config import (
    AWS_ACCESS_KEY_ID, 
    AWS_SECRET_ACCESS_KEY, 
    AWS_REGION, 
    S3_BUCKET_NAME,
    get_s3_url
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class S3Handler:
    """Handles all S3 operations for the multimodal PDF processing system."""
    
    def __init__(self):
        """Initialize S3 client with credentials."""
        try:
            # Validate configuration
            if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET_NAME]):
                raise ValueError("Missing required S3 configuration. Check your .env file.")
            
            # Initialize boto3 S3 client
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                region_name=AWS_REGION
            )
            
            self.bucket_name = S3_BUCKET_NAME
            self.region = AWS_REGION
            
            # Test connection
            self._test_connection()
            
            logger.info(f"S3Handler initialized successfully for bucket: {S3_BUCKET_NAME}")
            
        except Exception as e:
            logger.error(f"Failed to initialize S3Handler: {e}")
            raise
    
    def _test_connection(self):
        """Test S3 connection and bucket access."""
        try:
            # Try to list objects (limited to 1) to test permissions
            self.s3_client.list_objects_v2(Bucket=self.bucket_name, MaxKeys=1)
            logger.info("S3 connection test successful")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchBucket':
                raise ValueError(f"S3 bucket '{self.bucket_name}' does not exist")
            elif error_code == 'AccessDenied':
                raise ValueError(f"Access denied to S3 bucket '{self.bucket_name}'. Check your credentials and bucket permissions.")
            else:
                raise ValueError(f"S3 connection test failed: {e}")
        except NoCredentialsError:
            raise ValueError("AWS credentials not found. Check your .env file.")
    
    def upload_image(self, image_data: bytes, s3_key: str, content_type: str = None) -> str:
        """
        Upload image to S3 and return the public URL.
        
        Args:
            image_data: Raw image bytes
            s3_key: S3 key (path) for the image
            content_type: MIME type (auto-detected if not provided)
            
        Returns:
            Full public S3 URL for the uploaded image
        """
        try:
            # Auto-detect content type if not provided
            if not content_type:
                content_type = self._detect_content_type(s3_key)
            
            # Upload the image
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=image_data,
                ContentType=content_type,
                # Since bucket is public, no need for ACL settings
            )
            
            # Generate and return public URL
            s3_url = get_s3_url(s3_key)
            
            logger.info(f"Successfully uploaded image to S3: {s3_key} ({len(image_data)} bytes)")
            return s3_url
            
        except ClientError as e:
            logger.error(f"Failed to upload image to S3: {s3_key} - {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error uploading image: {s3_key} - {e}")
            raise
    
    def download_image(self, s3_key: str) -> bytes:
        """
        Download image from S3.
        
        Args:
            s3_key: S3 key (path) for the image
            
        Returns:
            Raw image bytes
        """
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            image_data = response['Body'].read()
            
            logger.info(f"Successfully downloaded image from S3: {s3_key} ({len(image_data)} bytes)")
            return image_data
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                logger.error(f"Image not found in S3: {s3_key}")
                raise FileNotFoundError(f"Image not found: {s3_key}")
            else:
                logger.error(f"Failed to download image from S3: {s3_key} - {e}")
                raise
        except Exception as e:
            logger.error(f"Unexpected error downloading image: {s3_key} - {e}")
            raise
    
    def list_objects(self, prefix: str = "", max_keys: int = 1000) -> List[str]:
        """
        List all objects in S3 with given prefix.
        
        Args:
            prefix: S3 prefix to filter objects
            max_keys: Maximum number of keys to return
            
        Returns:
            List of S3 keys
        """
        try:
            objects = []
            paginator = self.s3_client.get_paginator('list_objects_v2')
            
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix, MaxKeys=max_keys):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        objects.append(obj['Key'])
            
            logger.info(f"Found {len(objects)} objects with prefix: {prefix}")
            return objects
            
        except ClientError as e:
            logger.error(f"Failed to list S3 objects with prefix {prefix}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error listing S3 objects: {e}")
            raise
    
    def check_exists(self, s3_key: str) -> bool:
        """
        Check if an object exists in S3.
        
        Args:
            s3_key: S3 key to check
            
        Returns:
            True if object exists, False otherwise
        """
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                return False
            else:
                logger.error(f"Error checking if S3 object exists: {s3_key} - {e}")
                raise
        except Exception as e:
            logger.error(f"Unexpected error checking S3 object: {s3_key} - {e}")
            raise
    
    def delete_object(self, s3_key: str) -> bool:
        """
        Delete an object from S3.
        
        Args:
            s3_key: S3 key to delete
            
        Returns:
            True if deletion successful
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logger.info(f"Successfully deleted S3 object: {s3_key}")
            return True
        except ClientError as e:
            logger.error(f"Failed to delete S3 object: {s3_key} - {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error deleting S3 object: {s3_key} - {e}")
            raise
    
    def get_object_info(self, s3_key: str) -> Dict[str, Any]:
        """
        Get metadata information about an S3 object.
        
        Args:
            s3_key: S3 key to get info for
            
        Returns:
            Dictionary with object metadata
        """
        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            
            return {
                'key': s3_key,
                'size': response.get('ContentLength', 0),
                'content_type': response.get('ContentType', ''),
                'last_modified': response.get('LastModified'),
                'etag': response.get('ETag', '').strip('"'),
                's3_url': get_s3_url(s3_key)
            }
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                raise FileNotFoundError(f"S3 object not found: {s3_key}")
            else:
                logger.error(f"Error getting S3 object info: {s3_key} - {e}")
                raise
        except Exception as e:
            logger.error(f"Unexpected error getting S3 object info: {s3_key} - {e}")
            raise
    
    def _detect_content_type(self, s3_key: str) -> str:
        """Detect MIME type from file extension."""
        content_type, _ = mimetypes.guess_type(s3_key)
        
        # Default to common image types if detection fails
        if not content_type:
            extension = Path(s3_key).suffix.lower()
            content_type_map = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.gif': 'image/gif',
                '.bmp': 'image/bmp',
                '.webp': 'image/webp'
            }
            content_type = content_type_map.get(extension, 'application/octet-stream')
        
        return content_type
    
    def upload_from_local_file(self, local_file_path: str, s3_key: str) -> str:
        """
        Upload a local file to S3.
        
        Args:
            local_file_path: Path to local file
            s3_key: S3 key for the uploaded file
            
        Returns:
            Full public S3 URL
        """
        try:
            with open(local_file_path, 'rb') as f:
                image_data = f.read()
            
            return self.upload_image(image_data, s3_key)
            
        except FileNotFoundError:
            logger.error(f"Local file not found: {local_file_path}")
            raise
        except Exception as e:
            logger.error(f"Error uploading local file to S3: {local_file_path} -> {s3_key} - {e}")
            raise

print("S3Handler class created successfully")