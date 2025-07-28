#!/usr/bin/env python3
"""
Image Processor for LlamaParse Integration
Specialized module for extracting, processing, and storing images with comprehensive metadata
"""

import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
import re
import time
import numpy as np
import tempfile
import base64
from PIL import Image
import io

# Third-party imports
import pinecone
from pinecone import Pinecone
import openai
from openai import OpenAI
import streamlit as st
from llama_parse import LlamaParse

# Local imports
from config import AdvancedConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageProcessor:
    """Specialized processor for LlamaParse image extraction and S3/Pinecone storage."""
    
    def __init__(self, openai_client: OpenAI, pinecone_client: Pinecone, index):
        self.config = AdvancedConfig()
        self.openai_client = openai_client
        self.pinecone_client = pinecone_client
        self.index = index
        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-large')
        
        # Initialize S3 handler
        try:
            from ..s3_handler import S3Handler
            self.s3_handler = S3Handler()
            self.use_s3 = True
            logger.info("S3 storage enabled for image processing")
        except Exception as e:
            logger.warning(f"S3 initialization failed: {e}. Using local storage.")
            self.s3_handler = None
            self.use_s3 = False
        
        # Image processing settings
        self.supported_formats = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
        self.min_ocr_confidence = 0.3
        self.max_keywords = 20
        
    def process_images_from_llamaparse(self, json_objs: List[Dict], pdf_name: str, pdf_id: str) -> Dict[str, Any]:
        """Process all images from LlamaParse JSON output using proven extraction methods."""
        try:
            start_time = time.time()
            
            # Setup storage location (S3 or local)
            if self.use_s3:
                from config import get_pdf_s3_prefix
                storage_prefix = get_pdf_s3_prefix(pdf_name)
                logger.info(f"Using S3 storage: {storage_prefix}")
            else:
                image_dir = self.config.get_pdf_image_dir(pdf_name)
                logger.info(f"Using local storage: {image_dir}")
            
            # Use the proven extraction method with appropriate storage
            storage_location = storage_prefix if self.use_s3 else image_dir
            all_images = self._extract_images_with_llamaparse_get_images(json_objs, pdf_id, pdf_name, storage_location)
            
            if not all_images:
                logger.warning(f"No images found in {pdf_name}")
                return {
                    'processed_images': [],
                    'saved_count': 0,
                    'pinecone_stored': 0,
                    'processing_time': time.time() - start_time
                }
            
            # Process and save images
            processed_images = []
            saved_count = 0
            
            for i, img_data in enumerate(all_images):
                try:
                    processed_img = self._process_single_image(
                        img_data, i, storage_location, pdf_id, pdf_name
                    )
                    
                    if processed_img:
                        processed_images.append(processed_img)
                        if processed_img.get('saved_successfully'):
                            saved_count += 1
                        
                except Exception as e:
                    logger.error(f"Error processing image {i}: {e}")
                    continue
            
            # Store in Pinecone
            pinecone_stored = self._store_images_in_pinecone(processed_images, pdf_id)
            
            processing_time = time.time() - start_time
            
            logger.info(f"Image processing complete: {len(processed_images)} processed, "
                       f"{saved_count} saved, {pinecone_stored} stored in Pinecone")
            
            return {
                'processed_images': processed_images,
                'saved_count': saved_count,
                'pinecone_stored': pinecone_stored,
                'processing_time': processing_time,
                'storage_location': str(storage_location),
                'storage_type': 's3' if self.use_s3 else 'local'
            }
            
        except Exception as e:
            logger.error(f"Error in image processing: {e}")
            raise
    
    def _extract_images_with_llamaparse_get_images(self, json_objs: List[Dict], pdf_id: str, pdf_name: str, storage_location) -> List[Dict]:
        """Extract images using proven LlamaParse get_images method from reference.py."""
        all_images = []
        
        if not json_objs:
            return all_images
        
        try:
            # Initialize LlamaParse parser with same configuration
            api_key = os.getenv("LLAMA_PARSE_API_KEY")
            if not api_key:
                logger.error("LLAMA_PARSE_API_KEY not found")
                return all_images
            
            parser = LlamaParse(
                api_key=api_key,
                result_type="markdown",
                num_workers=4,
                verbose=True,
                language="en"
            )
            
            # Try Method 1: get_json_result() and get_images() (Primary method from reference)
            try:
                logger.info("🔄 Using LlamaParse get_images method for comprehensive extraction...")
                
                # For S3 storage, we need a temporary local path first
                if self.use_s3:
                    import tempfile
                    temp_dir = Path(tempfile.mkdtemp())
                    download_path = temp_dir
                    logger.info(f"Using temporary directory for S3 workflow: {download_path}")
                else:
                    download_path = storage_location
                
                # Extract images using get_images method with download path
                image_dicts = parser.get_images(json_objs, download_path=str(download_path))
                
                for idx, image_dict in enumerate(image_dicts):
                    # Read the downloaded image file - ONLY store actual images
                    image_path = Path(image_dict["path"])
                    if image_path.exists() and image_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']:
                        # Read image data for processing
                        with open(image_path, 'rb') as f:
                            image_data = f.read()
                        
                        # Extract OCR text and metadata from the original JSON
                        ocr_text = self._extract_ocr_from_llamaparse_image(image_dict, json_objs, idx)
                        keywords = self._extract_keywords_from_ocr(ocr_text)
                        
                        # Handle storage (S3 or local)
                        if self.use_s3:
                            # Upload to S3
                            s3_key = f"{storage_location}{pdf_id}_img_{idx+1}{image_path.suffix}"
                            s3_url = self.s3_handler.upload_image(image_data, s3_key)
                            storage_path = s3_url
                            storage_type = 's3'
                        else:
                            # Keep local file (already downloaded)
                            storage_path = str(image_path)
                            storage_type = 'local'
                        
                        # Create enhanced image data structure with OCR
                        enhanced_img_data = {
                            'image_id': f"{pdf_id}_img_{idx+1}",
                            'pdf_id': pdf_id,
                            'storage_path': storage_path,
                            'storage_type': storage_type,
                            's3_key': s3_key if self.use_s3 else None,
                            's3_url': s3_url if self.use_s3 else None,
                            'local_path': str(image_path) if not self.use_s3 else None,
                            'downloaded': True,
                            'extraction_source': 'llamaparse_get_images',
                            'image_data': image_data,
                            'file_size': len(image_data),
                            'ocr_text': ocr_text,
                            'keywords': keywords,
                            'original_dict': image_dict
                        }
                        
                        # Extract page number if available
                        if 'page' in image_dict:
                            enhanced_img_data['page_number'] = image_dict['page']
                        elif 'name' in image_dict:
                            # Try to extract page from filename
                            import re
                            page_match = re.search(r'page[_-]?(\d+)', image_dict['name'], re.IGNORECASE)
                            if page_match:
                                enhanced_img_data['page_number'] = int(page_match.group(1))
                        
                        all_images.append(enhanced_img_data)
                        logger.info(f"Extracted image with OCR: {storage_path} ({len(ocr_text)} chars OCR, {len(keywords)} keywords)")
                    else:
                        logger.debug(f"Skipping non-image file: {image_path}")
                
                # Cleanup temporary directory if using S3
                if self.use_s3 and 'temp_dir' in locals():
                    import shutil
                    try:
                        shutil.rmtree(temp_dir)
                        logger.info("Cleaned up temporary directory")
                    except Exception as e:
                        logger.warning(f"Failed to cleanup temp directory: {e}")
                
                if all_images:
                    logger.info(f"✅ Successfully extracted {len(all_images)} images using get_images method")
                    return all_images
                    
            except Exception as json_error:
                logger.warning(f"get_images method failed: {json_error}, trying fallback methods...")
            
            # Method 2: Fallback to JSON extraction with image saving
            try:
                logger.info("🔄 Falling back to JSON-based image extraction...")
                json_data = json_objs[0]
                
                for page_data in json_data.get('pages', []):
                    page_num = page_data.get('page', 0)
                    images_data = page_data.get('images', [])
                    
                    for i, img_data in enumerate(images_data):
                        image_id = f"{pdf_id}_p{page_num}_img{i}"
                        
                        # Try to save image locally using various data sources - ONLY actual images
                        saved_path = self._save_image_from_json_data(img_data, download_path, image_id, page_num, i)
                        
                        # Only include if we saved an actual image file
                        if saved_path and Path(saved_path).exists() and Path(saved_path).suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']:
                            # Extract OCR and keywords
                            ocr_text = self._extract_ocr_from_json_image_data(img_data)
                            keywords = self._extract_keywords_from_ocr(ocr_text)
                            
                            enhanced_img_data = {
                                **img_data,
                                'image_id': image_id,
                                'page_number': page_num,
                                'pdf_id': pdf_id,
                                'extraction_source': 'json_fallback',
                                'local_path': saved_path,
                                'downloaded': True,
                                'ocr_text': ocr_text,
                                'keywords': keywords
                            }
                            all_images.append(enhanced_img_data)
                            logger.info(f"Saved fallback image with OCR: {saved_path} ({len(ocr_text)} chars OCR)")
                
                logger.info(f"Fallback extraction found {len(all_images)} image references")
                return all_images
                
            except Exception as fallback_error:
                logger.error(f"Fallback extraction failed: {fallback_error}")
            
            return all_images
            
        except Exception as e:
            logger.error(f"Error in image extraction: {e}")
            return all_images
    
    def _save_image_from_json_data(self, img_data: Dict, save_dir: Path, image_id: str, 
                                 page_num: int, img_index: int) -> Optional[str]:
        """Save image from JSON data using base64 or other available sources."""
        try:
            # Check various image data fields that might contain image data
            image_data = None
            file_extension = '.png'  # default
            
            # Look for base64 or direct image data
            for data_field in ['image', 'data', 'base64', 'image_data', 'content']:
                if data_field in img_data and img_data[data_field]:
                    potential_data = img_data[data_field]
                    
                    if isinstance(potential_data, str):
                        if potential_data.startswith('data:image'):
                            # Base64 data URL
                            header, data = potential_data.split(',', 1)
                            image_data = base64.b64decode(data)
                            # Extract format from header
                            if 'jpeg' in header or 'jpg' in header:
                                file_extension = '.jpg'
                            elif 'png' in header:
                                file_extension = '.png'
                            elif 'gif' in header:
                                file_extension = '.gif'
                            break
                        elif len(potential_data) > 100:  # Likely base64 without header
                            try:
                                image_data = base64.b64decode(potential_data)
                                break
                            except:
                                pass
                    elif isinstance(potential_data, bytes):
                        # Direct binary data
                        image_data = potential_data
                        break
            
            # Determine file extension from metadata if available
            if img_data.get('format'):
                file_extension = f".{img_data['format'].lower()}"
            elif img_data.get('name'):
                ext = Path(img_data['name']).suffix
                if ext:
                    file_extension = ext
            
            # Save image ONLY if we have actual image data
            if image_data and len(image_data) > 10:
                image_path = save_dir / f"{image_id}{file_extension}"
                with open(image_path, 'wb') as f:
                    f.write(image_data)
                logger.info(f"Saved image from JSON data: {image_path}")
                return str(image_path)
            
            # No text files - only actual images
            logger.debug(f"No usable image data found for {image_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error saving image from JSON data {image_id}: {e}")
            return None
    
    def _extract_ocr_from_json_image_data(self, img_data: Dict) -> str:
        """Extract OCR text from JSON image data."""
        ocr_texts = []
        
        # Check direct text field
        if 'text' in img_data and img_data['text']:
            ocr_texts.append(img_data['text'])
        
        # Check OCR array with confidence filtering
        if 'ocr' in img_data and isinstance(img_data['ocr'], list):
            for ocr_item in img_data['ocr']:
                if isinstance(ocr_item, dict):
                    text = ocr_item.get('text', '').strip()
                    confidence = ocr_item.get('confidence', 1.0)
                    
                    if text and confidence >= 0.5:  # Filter low confidence text
                        ocr_texts.append(text)
                elif isinstance(ocr_item, str) and ocr_item.strip():
                    ocr_texts.append(ocr_item.strip())
        
        # Check other text fields
        for field in ['extracted_text', 'ocr_text', 'content']:
            if field in img_data and img_data[field]:
                ocr_texts.append(str(img_data[field]))
        
        # Combine and clean
        if ocr_texts:
            combined_text = ' '.join(ocr_texts)
            return ' '.join(combined_text.split())  # Remove excessive whitespace
        
        return ""
    
    def _extract_ocr_from_llamaparse_image(self, image_dict: Dict, json_objs: List[Dict], image_idx: int) -> str:
        """Extract OCR text from LlamaParse image dictionary and JSON objects."""
        ocr_texts = []
        
        # Method 1: Check direct text in image_dict
        if 'text' in image_dict and image_dict['text']:
            ocr_texts.append(image_dict['text'])
        
        # Method 2: Check OCR array in image_dict
        if 'ocr' in image_dict and isinstance(image_dict['ocr'], list):
            for ocr_item in image_dict['ocr']:
                if isinstance(ocr_item, dict):
                    text = ocr_item.get('text', '').strip()
                    confidence = ocr_item.get('confidence', 1.0)
                    if text and confidence >= 0.6:  # Higher confidence for better quality
                        ocr_texts.append(text)
                elif isinstance(ocr_item, str) and ocr_item.strip():
                    ocr_texts.append(ocr_item.strip())
        
        # Method 3: Search in JSON pages for matching image
        if json_objs and len(json_objs) > 0:
            json_data = json_objs[0]
            for page_data in json_data.get('pages', []):
                page_images = page_data.get('images', [])
                if image_idx < len(page_images):
                    img_data = page_images[image_idx]
                    if isinstance(img_data, dict):
                        page_ocr = self._extract_ocr_from_json_image_data(img_data)
                        if page_ocr:
                            ocr_texts.append(page_ocr)
                        break
        
        # Method 4: Check other possible fields
        for field in ['extracted_text', 'ocr_text', 'content', 'caption', 'description']:
            if field in image_dict and image_dict[field]:
                ocr_texts.append(str(image_dict[field]))
        
        # Combine and clean
        if ocr_texts:
            combined_text = ' '.join(ocr_texts)
            # Remove excessive whitespace and duplicates
            words = combined_text.split()
            unique_words = []
            seen = set()
            for word in words:
                if word.lower() not in seen:
                    unique_words.append(word)
                    seen.add(word.lower())
            return ' '.join(unique_words)
        
        return ""
    
    def _extract_keywords_from_ocr(self, ocr_text: str) -> List[str]:
        """Extract meaningful keywords from OCR text."""
        if not ocr_text or len(ocr_text.strip()) < 3:
            return []
        
        # Basic keyword extraction with improved filtering
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_-]*[a-zA-Z0-9]\b|\b[a-zA-Z]\b', ocr_text.lower())
        
        # Enhanced stop words
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 
            'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who',
            'this', 'that', 'with', 'from', 'they', 'she', 'said', 'each', 'which',
            'will', 'been', 'have', 'were', 'what', 'your', 'when', 'him', 'her',
            'than', 'other', 'more', 'very', 'into', 'just', 'only', 'over', 'also'
        }
        
        # Filter and count words
        word_counts = {}
        for word in words:
            if word not in stop_words and len(word) > 2:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Priority boost for technical/academic/visual terms
        priority_terms = {
            'figure', 'table', 'chart', 'graph', 'diagram', 'image', 'photo',
            'analysis', 'data', 'result', 'method', 'system', 'model', 'process',
            'algorithm', 'network', 'structure', 'function', 'performance',
            'research', 'study', 'experiment', 'test', 'evaluation', 'measurement',
            'percentage', 'value', 'score', 'rate', 'level', 'trend', 'pattern',
            'comparison', 'difference', 'correlation', 'significant', 'important'
        }
        
        for word in word_counts:
            if word in priority_terms:
                word_counts[word] *= 3  # Higher boost for important terms
        
        # Sort by frequency and return top keywords
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, count in sorted_words[:self.max_keywords]]
        
        return keywords

    def _download_llamaparse_image(self, img_data: Dict, save_dir: Path, pdf_id: str, 
                                 page_num: int, img_index: int) -> Optional[str]:
        """Download actual image from LlamaParse service."""
        try:
            import requests
            import base64
            import json
            from urllib.parse import urlparse
            
            # Generate unique filename
            image_id = f"{pdf_id}_p{page_num}_img{img_index}"
            
            # Check various image source fields that LlamaParse might use
            image_url = None
            image_data = None
            file_extension = '.png'  # default
            
            # Look for image URL or path
            for url_field in ['url', 'image_url', 'src', 'path', 'image_path']:
                if url_field in img_data and img_data[url_field]:
                    image_url = img_data[url_field]
                    break
            
            # Look for base64 or direct image data
            for data_field in ['image', 'data', 'base64', 'image_data']:
                if data_field in img_data and img_data[data_field]:
                    image_data = img_data[data_field]
                    break
            
            # Determine file extension
            if img_data.get('format'):
                file_extension = f".{img_data['format'].lower()}"
            elif img_data.get('name'):
                ext = Path(img_data['name']).suffix
                if ext:
                    file_extension = ext
            
            save_path = save_dir / f"{image_id}{file_extension}"
            
            # Try downloading from URL
            if image_url:
                if image_url.startswith('http'):
                    # External URL - download it
                    response = requests.get(image_url, timeout=10)
                    if response.status_code == 200:
                        with open(save_path, 'wb') as f:
                            f.write(response.content)
                        logger.info(f"Downloaded image from URL: {save_path}")
                        return str(save_path)
                elif image_url.startswith('/') and Path(image_url).exists():
                    # Local file path - copy it
                    import shutil
                    shutil.copy2(image_url, save_path)
                    logger.info(f"Copied local image: {save_path}")
                    return str(save_path)
            
            # Try processing base64 or direct image data
            if image_data:
                if isinstance(image_data, str):
                    if image_data.startswith('data:image'):
                        # Base64 data URL
                        header, data = image_data.split(',', 1)
                        decoded_data = base64.b64decode(data)
                        with open(save_path, 'wb') as f:
                            f.write(decoded_data)
                        logger.info(f"Saved base64 image: {save_path}")
                        return str(save_path)
                    elif len(image_data) > 100:  # Likely base64 without header
                        try:
                            decoded_data = base64.b64decode(image_data)
                            with open(save_path, 'wb') as f:
                                f.write(decoded_data)
                            logger.info(f"Saved base64 image: {save_path}")
                            return str(save_path)
                        except:
                            pass
                elif isinstance(image_data, bytes):
                    # Direct binary data
                    with open(save_path, 'wb') as f:
                        f.write(image_data)
                    logger.info(f"Saved binary image: {save_path}")
                    return str(save_path)
            
            # No debug files - only save actual images
            
            logger.warning(f"No usable image data found for {image_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error downloading image {pdf_id}_p{page_num}_img{img_index}: {e}")
            return None
    
    def _download_job_images(self, job_id: str, save_dir: Path, pdf_id: str):
        """Download actual images from LlamaParse job results."""
        try:
            import requests
            import os
            
            # Get API key
            api_key = os.getenv('LLAMA_PARSE_API_KEY')
            if not api_key:
                logger.error("LLAMA_PARSE_API_KEY not found")
                return
            
            # LlamaParse API endpoints
            base_url = "https://api.cloud.llamaindex.ai/api/parsing"
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # Get job info to see available result types
            job_url = f"{base_url}/job/{job_id}"
            response = requests.get(job_url, headers=headers)
            
            if response.status_code != 200:
                logger.error(f"Failed to get job info: {response.status_code}")
                return
            
            job_info = response.json()
            logger.info(f"Job info: {job_info.get('status', 'unknown status')}")
            
            # Download images from job results
            # Try different image result endpoints
            image_endpoints = [
                f"{base_url}/job/{job_id}/result/images",
                f"{base_url}/job/{job_id}/result/image",
                f"{base_url}/job/{job_id}/images"
            ]
            
            for endpoint in image_endpoints:
                try:
                    logger.info(f"Trying image endpoint: {endpoint}")
                    img_response = requests.get(endpoint, headers=headers)
                    
                    if img_response.status_code == 200:
                        content_type = img_response.headers.get('content-type', '')
                        
                        if 'application/json' in content_type:
                            # JSON response with image URLs or data
                            images_data = img_response.json()
                            self._process_images_json_response(images_data, save_dir, pdf_id)
                        elif 'application/zip' in content_type:
                            # ZIP file with images
                            self._extract_images_from_zip(img_response.content, save_dir, pdf_id)
                        elif content_type.startswith('image/'):
                            # Single image file
                            image_path = save_dir / f"{pdf_id}_image.png"
                            with open(image_path, 'wb') as f:
                                f.write(img_response.content)
                            logger.info(f"Downloaded single image: {image_path}")
                        
                        return  # Success, stop trying other endpoints
                        
                except Exception as e:
                    logger.warning(f"Failed endpoint {endpoint}: {e}")
                    continue
            
            logger.warning("No image endpoints returned successful results")
            
        except Exception as e:
            logger.error(f"Error downloading job images: {e}")
    
    def _process_images_json_response(self, images_data: Dict, save_dir: Path, pdf_id: str):
        """Process JSON response containing image data or URLs."""
        try:
            import requests
            
            if isinstance(images_data, list):
                # List of image objects
                for i, img_info in enumerate(images_data):
                    self._download_single_job_image(img_info, save_dir, pdf_id, i)
            elif isinstance(images_data, dict):
                # Check for different possible structures
                if 'images' in images_data:
                    for i, img_info in enumerate(images_data['images']):
                        self._download_single_job_image(img_info, save_dir, pdf_id, i)
                elif 'urls' in images_data:
                    for i, url in enumerate(images_data['urls']):
                        self._download_image_from_url(url, save_dir, pdf_id, i)
                else:
                    # Treat entire response as single image info
                    self._download_single_job_image(images_data, save_dir, pdf_id, 0)
                    
        except Exception as e:
            logger.error(f"Error processing images JSON response: {e}")
    
    def _download_single_job_image(self, img_info: Dict, save_dir: Path, pdf_id: str, index: int):
        """Download a single image from job result info."""
        try:
            import requests
            
            # Look for URL in various possible fields
            url = None
            for url_field in ['url', 'download_url', 'image_url', 'src', 'href']:
                if url_field in img_info and img_info[url_field]:
                    url = img_info[url_field]
                    break
            
            if url:
                self._download_image_from_url(url, save_dir, pdf_id, index)
            else:
                logger.warning(f"No URL found in image info: {list(img_info.keys())}")
                
        except Exception as e:
            logger.error(f"Error downloading single job image: {e}")
    
    def _download_image_from_url(self, url: str, save_dir: Path, pdf_id: str, index: int):
        """Download image from URL."""
        try:
            import requests
            import os
            
            headers = {}
            api_key = os.getenv('LLAMA_PARSE_API_KEY')
            if api_key and 'llamaindex.ai' in url:
                headers['Authorization'] = f"Bearer {api_key}"
            
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                # Determine file extension from content type or URL
                content_type = response.headers.get('content-type', '')
                if 'jpeg' in content_type or 'jpg' in content_type:
                    ext = '.jpg'
                elif 'png' in content_type:
                    ext = '.png'
                elif 'gif' in content_type:
                    ext = '.gif'
                else:
                    ext = '.png'  # default
                
                image_path = save_dir / f"{pdf_id}_img_{index}{ext}"
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                
                logger.info(f"Downloaded image from URL: {image_path}")
                return str(image_path)
            else:
                logger.warning(f"Failed to download image: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error downloading image from URL: {e}")
        
        return None
    
    def _extract_images_from_zip(self, zip_content: bytes, save_dir: Path, pdf_id: str):
        """Extract images from ZIP file."""
        try:
            import zipfile
            import io
            
            with zipfile.ZipFile(io.BytesIO(zip_content)) as zip_file:
                image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
                
                for file_info in zip_file.filelist:
                    file_ext = Path(file_info.filename).suffix.lower()
                    
                    if file_ext in image_extensions:
                        # Extract image file
                        image_data = zip_file.read(file_info.filename)
                        
                        # Clean filename
                        clean_name = Path(file_info.filename).name
                        image_path = save_dir / f"{pdf_id}_{clean_name}"
                        
                        with open(image_path, 'wb') as f:
                            f.write(image_data)
                        
                        logger.info(f"Extracted image from ZIP: {image_path}")
                        
        except Exception as e:
            logger.error(f"Error extracting images from ZIP: {e}")
    
    def _find_downloaded_image(self, pdf_image_dir: Path, image_id: str, page_num: int, img_index: int) -> Optional[str]:
        """Find downloaded image file that matches the image metadata."""
        try:
            # Look for various possible filenames (now includes new patterns)
            possible_patterns = [
                f"{image_id}.png",
                f"{image_id}.jpg", 
                f"{image_id}.jpeg",
                f"*_img_{img_index}.*",
                f"*_p{page_num}_*.*",
                f"*img{img_index}.*",
                f"*_p{page_num}_c*_i{img_index}.*",  # New container pattern
                f"*_b64_*_{img_index}.*",  # New base64 pattern
                f"*_p{page_num}_content_{img_index}.*",  # Content placeholders
                f"*_extracted_{img_index}.*"  # Extracted pattern
            ]
            
            # Try to find any image file in the directory first
            all_files = list(pdf_image_dir.glob("*"))
            logger.debug(f"Looking for image in directory with {len(all_files)} files")
            
            for pattern in possible_patterns:
                matches = list(pdf_image_dir.glob(pattern))
                if matches:
                    logger.info(f"Found image match with pattern '{pattern}': {matches[0]}")
                    return str(matches[0])
            
            # If no specific pattern matches, return any image file for this page
            page_files = list(pdf_image_dir.glob(f"*_p{page_num}_*"))
            if page_files:
                logger.info(f"Found page-specific file: {page_files[0]}")
                return str(page_files[0])
            
            # If still nothing, return any image file in the directory
            if all_files:
                for file_path in all_files:
                    if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']:
                        logger.info(f"Using available image file: {file_path}")
                        return str(file_path)
            
            logger.warning(f"No image file found for {image_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error finding downloaded image: {e}")
            return None
    
    def _process_single_image(self, img_data: Dict, index: int, storage_location, 
                            pdf_id: str, pdf_name: str) -> Optional[Dict]:
        """Process a single image with comprehensive metadata extraction."""
        try:
            # Generate unique image ID
            image_id = f"{pdf_id}_img_{index+1}"
            
            # Extract OCR text - use enhanced method if available, otherwise fallback
            if 'ocr_text' in img_data:
                ocr_text = img_data['ocr_text']  # Already extracted
            else:
                ocr_text = self._extract_high_quality_ocr(img_data)
            
            # Generate keywords - use enhanced method if available, otherwise fallback
            if 'keywords' in img_data:
                keywords = img_data['keywords']  # Already extracted
            else:
                keywords = self._extract_keywords_from_text(ocr_text)
            
            # Get image positioning and dimensions
            position_data = self._extract_position_data(img_data)
            
            # Classify image content type
            content_type = self._classify_image_content(ocr_text, img_data)
            
            # Calculate confidence score
            confidence_score = self._calculate_image_confidence(img_data, ocr_text)
            
            # Get storage information from the extraction phase
            storage_path = img_data.get('storage_path')
            storage_type = img_data.get('storage_type', 'local')
            s3_url = img_data.get('s3_url')
            s3_key = img_data.get('s3_key')
            
            # Create comprehensive image metadata
            processed_image = {
                'image_id': image_id,
                'pdf_id': pdf_id,
                'pdf_name': pdf_name,
                'page_number': img_data.get('page_number', 0),
                
                # OCR and semantic content
                'ocr_text': ocr_text,
                'keywords': keywords,
                'content_type': content_type,
                'confidence_score': confidence_score,
                
                # Positioning and layout
                'position': position_data,
                'dimensions': {
                    'width': img_data.get('width', 0),
                    'height': img_data.get('height', 0)
                },
                
                # Storage information
                'original_name': img_data.get('name', f'image_{index+1}'),
                'storage_path': storage_path,
                'storage_type': storage_type,
                's3_url': s3_url,
                's3_key': s3_key,
                'saved_successfully': bool(storage_path),
                
                # Processing metadata
                'extraction_source': img_data.get('extraction_source', 'unknown'),
                'processed_timestamp': datetime.now().isoformat(),
                
                # Vector embedding (will be generated during Pinecone storage)
                'embedding_text': self._create_embedding_text(ocr_text, keywords, content_type)
            }
            
            logger.info(f"Processed image {image_id}: {len(ocr_text)} chars OCR, "
                       f"{len(keywords)} keywords, type: {content_type}")
            
            return processed_image
            
        except Exception as e:
            logger.error(f"Error processing single image {index}: {e}")
            return None
    
    def _extract_high_quality_ocr(self, img_data: Dict) -> str:
        """Extract high-quality OCR text with confidence filtering."""
        ocr_texts = []
        
        # Process OCR data from different possible sources
        ocr_sources = [
            img_data.get('ocr', []),
            img_data.get('text_content', []),
            img_data.get('extracted_text', [])
        ]
        
        for ocr_data in ocr_sources:
            if not ocr_data:
                continue
                
            if isinstance(ocr_data, list):
                for item in ocr_data:
                    if isinstance(item, dict):
                        text = item.get('text', '').strip()
                        confidence = item.get('confidence', 1.0)
                        
                        if text and confidence >= self.min_ocr_confidence:
                            ocr_texts.append(text)
                    elif isinstance(item, str) and item.strip():
                        ocr_texts.append(item.strip())
                        
            elif isinstance(ocr_data, str) and ocr_data.strip():
                ocr_texts.append(ocr_data.strip())
        
        # Combine and clean OCR text
        combined_text = ' '.join(ocr_texts)
        
        # Remove excessive whitespace and normalize
        cleaned_text = ' '.join(combined_text.split())
        
        return cleaned_text
    
    def _extract_keywords_from_text(self, text: str) -> List[str]:
        """Extract meaningful keywords from text using NLP techniques."""
        if not text or len(text.strip()) < 3:
            return []
        
        # Basic keyword extraction
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_-]*[a-zA-Z0-9]\b|\b[a-zA-Z]\b', text.lower())
        
        # Stop words to filter out
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 
            'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who',
            'this', 'that', 'with', 'from', 'they', 'she', 'said', 'each', 'which'
        }
        
        # Filter and count words
        word_counts = {}
        for word in words:
            if word not in stop_words and len(word) > 2:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Priority boost for technical/academic terms
        priority_terms = {
            'figure', 'table', 'chart', 'graph', 'diagram', 'image', 'photo',
            'analysis', 'data', 'result', 'method', 'system', 'model', 'process',
            'algorithm', 'network', 'structure', 'function', 'performance',
            'research', 'study', 'experiment', 'test', 'evaluation'
        }
        
        for word in word_counts:
            if word in priority_terms:
                word_counts[word] *= 2
        
        # Sort by frequency and return top keywords
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, count in sorted_words[:self.max_keywords]]
        
        return keywords
    
    def _extract_position_data(self, img_data: Dict) -> Dict[str, Any]:
        """Extract comprehensive positioning data."""
        return {
            'x': img_data.get('x', 0),
            'y': img_data.get('y', 0),
            'bbox': img_data.get('bbox', []),
            'page_position': self._calculate_relative_position(img_data)
        }
    
    def _calculate_relative_position(self, img_data: Dict) -> str:
        """Calculate relative position on page (top/middle/bottom, left/center/right)."""
        x = img_data.get('x', 0)
        y = img_data.get('y', 0)
        
        # Rough position estimation (would need page dimensions for accuracy)
        vertical = 'top' if y < 200 else 'middle' if y < 600 else 'bottom'
        horizontal = 'left' if x < 200 else 'center' if x < 400 else 'right'
        
        return f"{vertical}_{horizontal}"
    
    def _classify_image_content(self, ocr_text: str, img_data: Dict) -> str:
        """Classify image content based on OCR text and metadata."""
        if not ocr_text:
            return "figure"
        
        text_lower = ocr_text.lower()
        
        # Classification rules
        if any(indicator in text_lower for indicator in ['table', 'column', 'row', 'data']):
            return "table_image"
        elif any(indicator in text_lower for indicator in ['chart', 'graph', 'plot', '%', 'axis']):
            return "chart"
        elif any(indicator in text_lower for indicator in ['diagram', 'flowchart', 'process', 'step']):
            return "diagram"
        elif any(indicator in text_lower for indicator in ['figure', 'fig', 'image', 'photo']):
            return "illustration"
        elif len(ocr_text.split()) > 30:
            return "text_rich_image"
        else:
            return "figure"
    
    def _calculate_image_confidence(self, img_data: Dict, ocr_text: str) -> float:
        """Calculate overall confidence score for the image."""
        confidence = 0.0
        
        # OCR text quality
        if ocr_text:
            text_score = min(0.4, len(ocr_text) / 200)
            confidence += text_score
        
        # Image dimensions
        width = img_data.get('width', 0)
        height = img_data.get('height', 0)
        if width > 100 and height > 100:
            confidence += 0.3
        
        # Position data availability
        if img_data.get('x') is not None and img_data.get('y') is not None:
            confidence += 0.2
        
        # OCR confidence scores
        ocr_data = img_data.get('ocr', [])
        if isinstance(ocr_data, list) and ocr_data:
            avg_confidence = 0.0
            confidence_scores = []
            
            for item in ocr_data:
                if isinstance(item, dict) and 'confidence' in item:
                    confidence_scores.append(item['confidence'])
            
            if confidence_scores:
                avg_confidence = sum(confidence_scores) / len(confidence_scores)
                confidence += min(0.1, avg_confidence)
        
        return min(1.0, confidence)
    
    def _save_image_locally(self, img_data: Dict, image_dir: Path, image_id: str) -> Optional[str]:
        """Attempt to save image locally if image data is available."""
        try:
            # Create PDF-specific image directory
            pdf_id = img_data.get('pdf_id', 'unknown')
            pdf_image_dir = image_dir / pdf_id
            pdf_image_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if image data is available in various formats
            image_sources = [
                img_data.get('image'),  # LlamaParse format
                img_data.get('image_data'),
                img_data.get('base64_data'),
                img_data.get('binary_data'),
                img_data.get('data')  # Alternative field
            ]
            
            for source in image_sources:
                if source:
                    # Determine file format
                    file_extension = '.png'  # default
                    if img_data.get('format'):
                        file_extension = f".{img_data['format'].lower()}"
                    elif img_data.get('name'):
                        file_extension = Path(img_data['name']).suffix or '.png'
                    elif isinstance(source, str) and 'data:image/' in source:
                        # Extract format from base64 data URL
                        if 'jpeg' in source or 'jpg' in source:
                            file_extension = '.jpg'
                        elif 'png' in source:
                            file_extension = '.png'
                        elif 'gif' in source:
                            file_extension = '.gif'
                    
                    # Save image in PDF-specific folder
                    local_path = pdf_image_dir / f"{image_id}{file_extension}"
                    
                    # Handle different data formats
                    if isinstance(source, bytes):
                        with open(local_path, 'wb') as f:
                            f.write(source)
                    elif isinstance(source, str):
                        if source.startswith('data:image'):
                            # Handle base64 encoded images
                            import base64
                            try:
                                header, data = source.split(',', 1)
                                image_data = base64.b64decode(data)
                                with open(local_path, 'wb') as f:
                                    f.write(image_data)
                            except Exception as decode_error:
                                logger.warning(f"Failed to decode base64 image: {decode_error}")
                                continue
                        elif source.startswith('/') or source.startswith('\\'):
                            # Handle file path reference
                            if Path(source).exists():
                                import shutil
                                shutil.copy2(source, local_path)
                    
                    if local_path.exists():
                        logger.info(f"Saved image locally: {local_path}")
                        return str(local_path)
            
            # If no image data available, check if there's a reference to an existing file
            if 'path' in img_data or 'file_path' in img_data:
                existing_path = img_data.get('path') or img_data.get('file_path')
                if existing_path and Path(existing_path).exists():
                    # Copy to our organized structure
                    file_extension = Path(existing_path).suffix or '.png'
                    local_path = pdf_image_dir / f"{image_id}{file_extension}"
                    import shutil
                    shutil.copy2(existing_path, local_path)
                    logger.info(f"Copied existing image: {local_path}")
                    return str(local_path)
            
            logger.warning(f"No usable image data found for {image_id}")
            return None
            
        except Exception as e:
            logger.warning(f"Could not save image {image_id} locally: {e}")
            return None
    
    def _create_embedding_text(self, ocr_text: str, keywords: List[str], content_type: str = "image") -> str:
        """Create text for vector embedding generation optimized for semantic search."""
        embedding_parts = []
        
        # Primary content - OCR text has highest priority for semantic search
        if ocr_text:
            embedding_parts.append(ocr_text)
        
        # Enhanced keywords for better semantic matching
        if keywords:
            # Add keywords multiple times for emphasis in semantic search
            keywords_text = ' '.join(keywords)
            embedding_parts.append(f"Keywords: {keywords_text}")
            embedding_parts.append(keywords_text)  # Duplicate for emphasis
        
        # Content classification for better filtering
        embedding_parts.append(f"Content type: {content_type}")
        
        # Add context descriptors
        embedding_parts.append("Visual content from document image")
        embedding_parts.append("Document figure chart diagram table visual element")
        
        # Combine with separators for better tokenization
        return ' | '.join(embedding_parts)
    
    def _store_images_in_pinecone(self, processed_images: List[Dict], pdf_id: str) -> int:
        """Store processed images in Pinecone with comprehensive metadata."""
        if not processed_images:
            return 0
        
        vectors = []
        
        for img_data in processed_images:
            try:
                # Generate embedding
                embedding_text = img_data['embedding_text']
                embedding = self._get_embedding(embedding_text)
                
                # Create enhanced metadata for Pinecone optimized for semantic search
                ocr_text = img_data.get('ocr_text', '')
                keywords = img_data.get('keywords', [])
                
                metadata = {
                    'content_type': 'image',
                    'image_id': img_data['image_id'],
                    'pdf_id': pdf_id,
                    'pdf_name': img_data['pdf_name'],
                    'page_number': img_data['page_number'],
                    
                    # Enhanced OCR and semantic data for search
                    'ocr_text': ocr_text[:8000],  # Increased limit for better search
                    'keywords': keywords[:20] if isinstance(keywords, list) else [],
                    'has_text': bool(ocr_text and len(ocr_text.strip()) > 0),
                    'keywords_count': len(keywords) if isinstance(keywords, list) else 0,
                    'ocr_length': len(ocr_text) if ocr_text else 0,
                    
                    # Content classification
                    'content_classification': img_data.get('content_type', 'figure'),
                    'confidence_score': img_data.get('confidence_score', 0.0),
                    
                    # Positioning - handle missing position data gracefully
                    'position_x': img_data.get('position', {}).get('x', 0),
                    'position_y': img_data.get('position', {}).get('y', 0),
                    'page_position': img_data.get('position', {}).get('page_position', 'unknown'),
                    
                    # Dimensions - handle missing dimension data gracefully
                    'width': img_data.get('dimensions', {}).get('width', 0),
                    'height': img_data.get('dimensions', {}).get('height', 0),
                    
                    # Storage information (S3 or local)
                    'storage_type': img_data.get('storage_type', 'local'),
                    'storage_path': img_data.get('storage_path', ''),
                    's3_url': img_data.get('s3_url', ''),
                    's3_key': img_data.get('s3_key', ''),
                    'local_path': img_data.get('local_path', ''),  # Legacy field
                    'original_name': img_data.get('original_name', 'unknown'),
                    'saved_successfully': bool(img_data.get('saved_successfully', False)),
                    
                    # Processing metadata
                    'extraction_source': img_data.get('extraction_source', 'unknown'),
                    'processed_timestamp': img_data.get('processed_timestamp', datetime.now().isoformat())
                }
                
                vector = {
                    'id': f"img_{img_data['image_id']}",
                    'values': embedding,
                    'metadata': metadata
                }
                
                vectors.append(vector)
                
            except Exception as e:
                logger.error(f"Error creating vector for image {img_data.get('image_id')}: {e}")
                continue
        
        # Batch upsert to Pinecone
        if vectors:
            try:
                self.index.upsert(vectors=vectors, namespace=pdf_id)
                logger.info(f"Stored {len(vectors)} image vectors in Pinecone namespace: {pdf_id}")
                return len(vectors)
            except Exception as e:
                logger.error(f"Error upserting image vectors to Pinecone: {e}")
                return 0
        
        return 0
    
    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API."""
        try:
            if not text or len(text.strip()) < 3:
                return [0.0] * 3072  # Default embedding size for text-embedding-3-large
            
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text[:8191]  # OpenAI token limit
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return [0.0] * 3072
    
    def extract_images_from_uploaded_file(self, uploaded_file, pdf_id: str = None) -> Dict[str, Any]:
        """Extract images from uploaded PDF file using proven LlamaParse method."""
        try:
            start_time = time.time()
            
            # Generate PDF ID if not provided
            if not pdf_id:
                pdf_content = uploaded_file.getvalue()
                pdf_hash = hashlib.md5(pdf_content).hexdigest()[:10]
                pdf_id = f"pdf_{pdf_hash}"
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Setup download directory using PDF name
                from config import get_pdf_image_dir
                download_path = get_pdf_image_dir(uploaded_file.name)
                
                # Initialize parser
                api_key = os.getenv("LLAMA_PARSE_API_KEY")
                if not api_key:
                    raise ValueError("LLAMA_PARSE_API_KEY not found")
                
                parser = LlamaParse(
                    api_key=api_key,
                    result_type="markdown",
                    num_workers=4,
                    verbose=True,
                    language="en"
                )
                
                images = []
                image_paths = []
                
                # Method 1: Try get_json_result() and get_images()
                try:
                    logger.info("🔄 Using LlamaParse JSON mode for comprehensive extraction...")
                    json_objs = parser.get_json_result(tmp_file_path)
                    
                    if json_objs and len(json_objs) > 0:
                        # Extract images using get_images method
                        image_dicts = parser.get_images(json_objs, download_path=str(download_path))
                        
                        for idx, image_dict in enumerate(image_dicts):
                            # Read the downloaded image file
                            image_path = Path(image_dict["path"])
                            if image_path.exists():
                                with open(image_path, 'rb') as f:
                                    image_data = f.read()
                                
                                images.append(image_data)
                                image_paths.append(str(image_path))
                        
                        if images:
                            processing_time = time.time() - start_time
                            logger.info(f"✅ Successfully extracted {len(images)} images in {processing_time:.2f}s")
                            return {
                                'images': images,
                                'image_paths': image_paths,
                                'pdf_id': pdf_id,
                                'extraction_method': 'llamaparse_get_images',
                                'processing_time': processing_time,
                                'success': True
                            }
                            
                except Exception as json_error:
                    logger.warning(f"get_json_result() method failed: {json_error}, trying standard method...")
                
                # Method 2: Fallback to standard load_data method
                try:
                    logger.info("🔄 Falling back to standard LlamaParse method...")
                    documents = parser.load_data(tmp_file_path)
                    
                    # Try to extract images from document metadata
                    for doc in documents:
                        if hasattr(doc, 'metadata') and 'images' in doc.metadata:
                            for i, image_data in enumerate(doc.metadata['images']):
                                saved_path = self._save_image_locally_from_data(image_data, download_path, pdf_id, len(images))
                                if saved_path:
                                    if isinstance(image_data, str):
                                        image_bytes = base64.b64decode(image_data)
                                    else:
                                        image_bytes = image_data
                                    images.append(image_bytes)
                                    image_paths.append(saved_path)
                    
                    processing_time = time.time() - start_time
                    return {
                        'images': images,
                        'image_paths': image_paths,
                        'pdf_id': pdf_id,
                        'extraction_method': 'llamaparse_standard',
                        'processing_time': processing_time,
                        'success': len(images) > 0
                    }
                    
                except Exception as load_error:
                    logger.error(f"Standard LlamaParse method also failed: {load_error}")
                    return {
                        'images': [],
                        'image_paths': [],
                        'pdf_id': pdf_id,
                        'extraction_method': 'failed',
                        'processing_time': time.time() - start_time,
                        'success': False,
                        'error': str(load_error)
                    }
                
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)
                
        except Exception as e:
            logger.error(f"Error extracting images from uploaded file: {e}")
            return {
                'images': [],
                'image_paths': [],
                'pdf_id': pdf_id or 'unknown',
                'extraction_method': 'failed',
                'processing_time': time.time() - start_time if 'start_time' in locals() else 0,
                'success': False,
                'error': str(e)
            }
    
    def _save_image_locally_from_data(self, image_data, save_dir: Path, pdf_id: str, index: int) -> Optional[str]:
        """Save image locally from raw image data."""
        try:
            # Create PDF-specific image directory
            pdf_image_dir = save_dir
            pdf_image_dir.mkdir(parents=True, exist_ok=True)
            
            # Save image
            image_path = pdf_image_dir / f"image_{index + 1}.png"
            
            # Handle different data formats
            if isinstance(image_data, str):
                # If it's base64, decode it
                image_bytes = base64.b64decode(image_data)
            else:
                image_bytes = image_data
                
            # Save the image
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            
            logger.info(f"Saved image locally: {image_path}")
            return str(image_path)
            
        except Exception as e:
            logger.error(f"Error saving image {index + 1}: {str(e)}")
            return None

    def retrieve_relevant_images(self, query: str, pdf_id: str, top_k: int = 5) -> List[Dict]:
        """Retrieve most relevant images for a query using vector similarity."""
        try:
            # Generate query embedding
            query_embedding = self._get_embedding(query)
            
            # Enhanced query for better semantic search
            results = self.index.query(
                vector=query_embedding,
                top_k=min(top_k * 2, 20),  # Get more candidates for better filtering
                namespace=pdf_id,
                filter={"content_type": "image"},
                include_metadata=True,
                include_values=False
            )
            
            relevant_images = []
            
            for match in results.matches:
                # Enhanced relevance scoring with OCR text consideration
                base_score = match.score
                has_text = match.metadata.get('has_text', False)
                ocr_length = match.metadata.get('ocr_length', 0)
                keywords_count = match.metadata.get('keywords_count', 0)
                
                # Boost score for images with relevant text content
                relevance_boost = 0.0
                if has_text and ocr_length > 20:
                    relevance_boost += 0.1
                if keywords_count > 3:
                    relevance_boost += 0.05
                
                adjusted_score = base_score + relevance_boost
                
                # Dynamic threshold based on content quality
                min_threshold = 0.15 if has_text else 0.25
                
                if adjusted_score > min_threshold:
                    ocr_text = match.metadata.get('ocr_text', '')
                    keywords = match.metadata.get('keywords', [])
                    
                    # Get storage information
                    storage_type = match.metadata.get('storage_type', 'local')
                    storage_path = match.metadata.get('storage_path', '')
                    s3_url = match.metadata.get('s3_url', '')
                    local_path = match.metadata.get('local_path', '')
                    
                    image_data = {
                        'image_id': match.metadata.get('image_id', ''),
                        'relevance_score': adjusted_score,
                        'original_score': base_score,
                        'storage_type': storage_type,
                        'storage_path': storage_path,
                        's3_url': s3_url,
                        'local_path': local_path,
                        'display_url': s3_url if storage_type == 's3' else local_path,  # For easy UI access
                        'page_number': match.metadata.get('page_number', 0),
                        'ocr_text': ocr_text,
                        'keywords': keywords,
                        'has_text': has_text,
                        'content_type': match.metadata.get('content_classification', 'figure'),
                        'confidence_score': match.metadata.get('confidence_score', 0.0),
                        'position': {
                            'x': match.metadata.get('position_x', 0),
                            'y': match.metadata.get('position_y', 0),
                            'page_position': match.metadata.get('page_position', 'unknown')
                        }
                    }
                    relevant_images.append(image_data)
            
            # Sort by adjusted relevance score
            relevant_images.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            # Return top k results
            relevant_images = relevant_images[:top_k]
            
            logger.info(f"Retrieved {len(relevant_images)} relevant images for query")
            return relevant_images
            
        except Exception as e:
            logger.error(f"Error retrieving relevant images: {e}")
            return []

print("ImageProcessor class created successfully")