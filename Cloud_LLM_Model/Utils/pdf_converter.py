"""
Enhanced PDF processor that extracts both text and images from PDFs.
Uses GPT-4 Vision capabilities to analyze images.
"""

import os
import base64
import logging
import tempfile
from typing import Dict, List, Any, Optional, Tuple
import fitz  # PyMuPDF
import requests
import json
import io
from PIL import Image
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PdfToTextConverter")

class PdfToTextConverter:
    """
    Enhanced PDF processor that extracts both text and images from PDFs
    and uses GPT-4 Vision capabilities to understand image content.
    
    Maintains compatibility with the original PdfToTextConverter interface
    while adding image analysis capabilities.
    """
    
    def __init__(self, input_folder=None, output_folder=None, api_key=None):
        """
        Initialize the converter with optional input and output folders.
        
        Args:
            input_folder (str, optional): Directory containing PDF files to convert
            output_folder (str, optional): Directory to save text files
            api_key (str, optional): OpenAI API key for image analysis
        """
        self.input_folder = input_folder
        self.output_folder = output_folder or input_folder
        self.api_key = api_key
        
        # Create output directory if it doesn't exist
        if self.output_folder and not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        logger.info("PdfToTextConverter initialized" + (" with image analysis capabilities" if api_key else ""))
    
    def convert_file(self, pdf_path, output_path=None):
        """
        Convert a single PDF file to text with optional image analysis.
        
        Args:
            pdf_path (str): Path to the PDF file
            output_path (str, optional): Path for the output text file
            
        Returns:
            str: Path to the created text file
        """
        if not output_path:
            # Create output path with same name but .txt extension
            filename = os.path.basename(pdf_path)
            name_without_ext = os.path.splitext(filename)[0]
            output_path = os.path.join(self.output_folder, f"{name_without_ext}.txt")
        
        # Check if we can use the enhanced processing with image analysis
        if self.api_key:
            try:
                logger.info(f"Processing PDF with image analysis: {pdf_path}")
                result = self.process_pdf_with_images(pdf_path, output_path)
                if result and "success" in result and result["success"]:
                    return output_path
                else:
                    logger.warning(f"Enhanced processing failed, falling back to text-only extraction")
            except Exception as e:
                logger.error(f"Error in enhanced PDF processing: {str(e)}")
                logger.info("Falling back to text-only extraction")
        
        # Fall back to basic text extraction (original method)
        try:
            logger.info(f"Extracting text from PDF: {pdf_path}")
            self._extract_text_only(pdf_path, output_path)
            return output_path
        except Exception as e:
            logger.error(f"Error converting {pdf_path}: {str(e)}")
            return None
    
    def _extract_text_only(self, pdf_path, output_path):
        """
        Extract only text from a PDF file (original functionality).
        
        Args:
            pdf_path (str): Path to the PDF file
            output_path (str): Path for the output text file
            
        Returns:
            bool: Success status
        """
        try:
            # Using PyPDF2 for compatibility with your existing implementation
            from PyPDF2 import PdfReader
            
            reader = PdfReader(pdf_path)
            text = ""
            
            # Extract text from each page
            for page in reader.pages:
                text += page.extract_text() + "\n\n"
            
            # Write text to file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
                
            logger.info(f"Successfully extracted text from {pdf_path} to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            return False
    
    def process_pdf_with_images(self, pdf_path, output_path=None):
        """
        Process a PDF file to extract text and analyze images.
        Enhanced to handle various types of visual content in PDFs.
        
        Args:
            pdf_path (str): Path to the PDF file
            output_path (str, optional): Path for the output text file
            
        Returns:
            Dict with extracted text, image analyses, and combined content
        """
        try:
            logger.info(f"Processing PDF with enhanced image extraction: {pdf_path}")
            
            # Output dictionary
            result = {
                "text": "",
                "image_analyses": [],
                "combined_content": "",
                "success": True
            }
            
            # Open the PDF
            pdf_document = fitz.open(pdf_path)
            
            # Process each page
            for page_num, page in enumerate(pdf_document):
                logger.info(f"Processing page {page_num+1}/{len(pdf_document)}")
                
                # Extract text from the page
                page_text = page.get_text()
                result["text"] += page_text + "\n\n"
                
                # Extract images from the page using enhanced method
                image_list = self._extract_images_from_page(page, page_num)
                
                # If no images were found with regular extraction, render the whole page
                if not image_list and self.api_key:
                    logger.info(f"No images found on page {page_num+1}, rendering full page")
                    full_page_image = self._extract_page_as_image(page, page_num)
                    if full_page_image:
                        image_list.append(full_page_image)
                
                # Analyze images if any were found
                if image_list:
                    logger.info(f"Found {len(image_list)} images on page {page_num+1}")
                    for img_info in image_list:
                        # Only analyze if we have image data
                        if "image_data" not in img_info or not img_info["image_data"]:
                            continue
                            
                        # Analyze image with GPT-4 Vision
                        img_context = f"This image is from page {page_num + 1} of a PDF document."
                        if "is_full_page" in img_info and img_info["is_full_page"]:
                            img_context = f"This is a rendering of the entire page {page_num + 1} of a PDF document."
                        
                        analysis = self._analyze_image_with_gpt(img_info["image_data"], context=img_context)
                        
                        # Save analysis result
                        img_info["analysis"] = analysis
                        result["image_analyses"].append(img_info)
                        
                        # Add image description to the combined content
                        image_desc = f"\n\n[Image on page {page_num + 1}"
                        if "is_full_page" in img_info and img_info["is_full_page"]:
                            image_desc = f"\n\n[Full page rendering of page {page_num + 1}"
                        result["combined_content"] += f"{image_desc}: {analysis}]\n\n"
            
            # Add extracted text to combined content
            result["combined_content"] += result["text"]
            
            # Create text file if output_path is provided
            if output_path:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(result["combined_content"])
                
                result["output_file"] = output_path
                logger.info(f"Successfully processed PDF with images to {output_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path} with images: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "text": "",
                "image_analyses": [],
                "combined_content": "",
                "success": False,
                "error": str(e)
            }
    
    def _extract_images_from_page(self, page, page_num):
        """
        Enhanced method to extract images from a page.
        Handles multiple types of image data in PDFs.
        
        Args:
            page: PyMuPDF page object
            page_num: Page number (0-based)
            
        Returns:
            List of dictionaries with image data and metadata
        """
        image_list = []
        
        # METHOD 1: Extract standard images using get_images
        try:
            img_list = page.get_images(full=True)
            logger.debug(f"Found {len(img_list)} standard images on page {page_num+1}")
            
            for img_index, img in enumerate(img_list):
                try:
                    xref = img[0]
                    base_image = page.parent.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Skip very small images (likely icons or bullet points)
                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)
                    if width < 50 or height < 50:
                        logger.debug(f"Skipping small image {width}x{height} on page {page_num+1}")
                        continue
                    
                    # Convert to base64 for API
                    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                    
                    # Save image info
                    image_info = {
                        "page_num": page_num,
                        "image_index": img_index,
                        "image_data": image_b64,
                        "width": width,
                        "height": height,
                        "colorspace": base_image.get("colorspace", "unknown"),
                        "extraction_method": "standard"
                    }
                    
                    # Save image to file if output folder is specified
                    if self.output_folder:
                        img_filename = f"page{page_num+1}_img{img_index+1}.png"
                        img_path = os.path.join(self.output_folder, img_filename)
                        
                        with open(img_path, "wb") as img_file:
                            img_file.write(image_bytes)
                        
                        image_info["file_path"] = img_path
                    
                    image_list.append(image_info)
                    
                except Exception as e:
                    logger.error(f"Error extracting standard image {img_index} from page {page_num}: {str(e)}")
                    continue
        except Exception as e:
            logger.error(f"Error in standard image extraction on page {page_num}: {str(e)}")
        
        # METHOD 2: Extract images from XObjects that might be missed by get_images
        try:
            # Get all XObjects on the page
            xobjects = page.get_xobjects()
            
            if xobjects:
                logger.debug(f"Found {len(xobjects)} XObjects on page {page_num+1}")
                
                for xobj_index, xobj in enumerate(xobjects):
                    try:
                        # Try to extract image from this XObject
                        xref = xobj[0]  # Get reference number
                        
                        # Skip if already processed (would be in the standard images)
                        already_processed = False
                        for img in img_list:
                            if img[0] == xref:
                                already_processed = True
                                break
                        
                        if already_processed:
                            continue
                        
                        # Try to extract as image
                        try:
                            base_image = page.parent.extract_image(xref)
                            if not base_image:
                                continue  # Not an image XObject
                                
                            image_bytes = base_image["image"]
                            
                            # Skip very small images
                            width = base_image.get("width", 0)
                            height = base_image.get("height", 0)
                            if width < 50 or height < 50:
                                continue
                                
                            # Convert to base64 for API
                            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                            
                            # Save image info
                            image_info = {
                                "page_num": page_num,
                                "image_index": f"xobj_{xobj_index}",
                                "image_data": image_b64,
                                "width": width,
                                "height": height,
                                "colorspace": base_image.get("colorspace", "unknown"),
                                "extraction_method": "xobject"
                            }
                            
                            # Save image to file if output folder is specified
                            if self.output_folder:
                                img_filename = f"page{page_num+1}_xobj{xobj_index+1}.png"
                                img_path = os.path.join(self.output_folder, img_filename)
                                
                                with open(img_path, "wb") as img_file:
                                    img_file.write(image_bytes)
                                
                                image_info["file_path"] = img_path
                            
                            image_list.append(image_info)
                            
                        except Exception as e:
                            logger.debug(f"XObject {xref} on page {page_num+1} is not an image or couldn't be extracted: {str(e)}")
                            continue
                            
                    except Exception as e:
                        logger.error(f"Error processing XObject {xobj_index} on page {page_num+1}: {str(e)}")
                        continue
        except Exception as e:
            logger.error(f"Error extracting XObject images from page {page_num}: {str(e)}")
        
        # METHOD 3: Look for image data in Form XObjects (nested content)
        try:
            for form_xref in page._getContents():
                try:
                    # This is a more advanced technique that tries to find images in form XObjects
                    # Implementation would be complex and require parsing PDF operators
                    # This is a placeholder for a more advanced implementation
                    pass
                except Exception as e:
                    logger.debug(f"Error processing form XObject {form_xref} on page {page_num+1}: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing form XObjects on page {page_num}: {str(e)}")
        
        # Return all found images
        return image_list
    
    def _extract_page_as_image(self, page, page_num):
        """
        Extract the entire page as an image.
        Useful when standard image extraction methods don't find images,
        or for pages that use vector graphics or other non-standard image content.
        
        Args:
            page: PyMuPDF page object
            page_num: Page number (0-based)
            
        Returns:
            Dictionary with image data and metadata
        """
        try:
            # Set resolution for rendering (higher values = better quality but larger files)
            zoom_factor = 2.0  # Adjust based on your needs
            
            # Create a matrix for zooming
            matrix = fitz.Matrix(zoom_factor, zoom_factor)
            
            # Render page to pixmap
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            
            # Convert to PNG format
            img_bytes = pix.tobytes("png")
            
            # Convert to base64 for API
            image_b64 = base64.b64encode(img_bytes).decode('utf-8')
            
            # Save image info
            image_info = {
                "page_num": page_num,
                "image_index": "full_page",
                "image_data": image_b64,
                "width": pix.width,
                "height": pix.height,
                "is_full_page": True,
                "extraction_method": "page_render"
            }
            
            # Save image to file if output folder is specified
            if self.output_folder:
                img_filename = f"page{page_num+1}_full.png"
                img_path = os.path.join(self.output_folder, img_filename)
                
                with open(img_path, "wb") as img_file:
                    img_file.write(img_bytes)
                
                image_info["file_path"] = img_path
            
            return image_info
            
        except Exception as e:
            logger.error(f"Error rendering page {page_num} as image: {str(e)}")
            return None
    
    def _analyze_image_with_gpt(self, image_b64, context=""):
        """
        Analyze an image using OpenAI's Vision capabilities.
        
        Args:
            image_b64: Base64-encoded image data
            context: Optional context about the image
            
        Returns:
            String containing the analysis result
        """
        try:
            if not self.api_key:
                return "Image analysis not available (no API key provided)"
                
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # Create a more specific prompt based on image context
            system_prompt = "You are a helpful assistant that describes and analyzes images from PDFs."
            if "full page" in context.lower():
                system_prompt += " This is a full page rendering. Focus on describing the overall layout, main visual elements, and any visible text. Transcribe important text that might not be captured by OCR."
            else:
                system_prompt += " Focus on the most important content and details. If the image contains text, transcribe it accurately."
            
            # Updated to use the current model name for GPT-4 with vision capabilities
            payload = {
                "model": "gpt-4o",  # Updated to use the current recommended model
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Please describe and analyze this image. {context}"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 500  # Increased token limit for more detailed descriptions
            }
            
            # Add unique request ID for debugging
            request_id = str(uuid.uuid4())
            logger.debug(f"Sending image analysis request {request_id}")
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                analysis = result["choices"][0]["message"]["content"]
                logger.debug(f"Image analysis successful for request {request_id}")
                return analysis
            else:
                logger.error(f"Error from OpenAI API ({request_id}): {response.status_code} - {response.text}")
                return f"Error analyzing image: API returned status {response.status_code}"
                
        except Exception as e:
            logger.error(f"Error analyzing image with GPT: {str(e)}")
            return "Error analyzing image: " + str(e)    
        
    def convert_folder(self):
        """
        Convert all PDF files in the input folder to text files.
        
        Returns:
            list: Paths to all created text files
        """
        if not self.input_folder:
            raise ValueError("Input folder not specified")
        
        created_files = []
        
        # Process all PDF files in the folder
        for filename in os.listdir(self.input_folder):
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(self.input_folder, filename)
                output_file = self.convert_file(pdf_path)
                if output_file:
                    created_files.append(output_file)
        
        return created_files