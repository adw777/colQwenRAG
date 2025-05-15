import os
import json
import requests
import base64
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
import pdf2image
import io
from PIL import Image

class DocumentParsingTool:
    """Tool for extracting and parsing text and images from documents using an API."""

    name: str = "Document Parsing Tool"
    description: str = "Extract text from PDF by converting to images and using OCR"
    pdf_to_image_api_url: str = "https://sweeping-moth-probably.ngrok-free.app/parse_image/image" #https://sweeping-moth-probably.ngrok-free.app/parse_image/image
    
    def _validate_input(self, file_path: str) -> Dict:
        """Validate the input document file."""
        result = {
            "valid": False,
            "file_path": "",
            "error": ""
        }
        
        try:
            # Convert to Path object for better path handling
            path = Path(file_path)
            
            # Verify file exists
            if not path.exists():
                result["error"] = f"File not found: {file_path}"
                return result
            
            # Verify it's a PDF file
            if path.suffix.lower() != '.pdf':
                result["error"] = f"File must be a PDF: {path.suffix}"
                return result
            
            result["file_path"] = str(path.absolute())
            result["valid"] = True
            return result
            
        except Exception as e:
            result["error"] = f"Error validating input: {str(e)}"
            return result
    
    def _convert_pdf_to_images(self, pdf_path: str) -> List[Dict]:
        """Convert PDF to a list of images."""
        try:
            print(f"Converting PDF to images: {pdf_path}")
            
            # Convert PDF to list of PIL images
            images = pdf2image.convert_from_path(
                pdf_path, 
                dpi=300,  # Higher DPI for better quality
                fmt="png"
            )
            
            print(f"Successfully converted PDF to {len(images)} images")
            
            # Convert PIL images to the format needed for API
            image_list = []
            for i, img in enumerate(images):
                # Save image to bytes buffer
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                
                # Add to image list
                image_list.append({
                    "index": i,
                    "image": img_byte_arr,
                    "format": "png"
                })
            
            return image_list
            
        except Exception as e:
            print(f"Error converting PDF to images: {str(e)}")
            raise
    
    def _extract_text_from_image(self, image_data: Dict) -> Dict:
        """Send image to API and extract text."""
        result = {
            "success": False,
            "content": {},
            "page_number": image_data["index"] + 1,
            "error": ""
        }
        
        try:
            # Prepare the file for upload
            files = {
                'file': (f'page_{image_data["index"] + 1}.{image_data["format"]}', 
                         image_data["image"], 
                         f'image/{image_data["format"]}')
            }
            
            # Send POST request
            print(f"Sending Page {result['page_number']} to OCR API...")
            response = requests.post(self.pdf_to_image_api_url, files=files)
            response.raise_for_status()  # Raise exception for bad status codes
            
            # Parse JSON response
            content = response.json()
            
            result["success"] = True
            result["content"] = content
            return result
                
        except requests.RequestException as e:
            result["error"] = f"Error sending request to OCR API: {str(e)}"
            return result
        except json.JSONDecodeError as e:
            result["error"] = f"Error parsing API response: {str(e)}"
            return result
        except Exception as e:
            result["error"] = f"Unexpected error during image OCR: {str(e)}"
            return result
    
    def _save_to_markdown(self, extraction_results: List[Dict], output_file: str) -> None:
        """Save the extracted content to a markdown file."""
        try:
            with open(output_file, 'w', encoding='utf-8') as md_file:
                # Add timestamp and header
                md_file.write(f"# PDF OCR Extraction Results\n\n")
                md_file.write(f"_Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n\n")
                
                # Add content from each page
                for result in extraction_results:
                    md_file.write(f"## Page {result['page_number']}\n\n")
                    
                    if not result["success"]:
                        md_file.write(f"*Error processing this page: {result['error']}*\n\n")
                        continue
                    
                    content = result["content"]
                    
                    # Add extracted text
                    if "text" in content:
                        if isinstance(content["text"], str):
                            md_file.write(content["text"])
                        elif isinstance(content["text"], list):
                            for section in content["text"]:
                                md_file.write(f"{section}\n\n")
                    else:
                        md_file.write("*No text extracted for this page*\n\n")
                    
                    md_file.write("\n---\n\n")
                
                # Add summary information
                successful_pages = sum(1 for r in extraction_results if r["success"])
                md_file.write(f"## Summary\n\n")
                md_file.write(f"* Total Pages: {len(extraction_results)}\n")
                md_file.write(f"* Successfully Processed: {successful_pages}\n")
                md_file.write(f"* Failed: {len(extraction_results) - successful_pages}\n\n")
        
        except Exception as e:
            print(f"Error saving markdown file: {str(e)}")
            raise
    
    def parse_pdf_as_images(self, file_path: str) -> str:
        """Process the PDF by converting to images and extracting text via OCR."""
        # Validate the input file
        input_data = self._validate_input(file_path)
        
        if not input_data["valid"]:
            return f"Error: {input_data['error']}"
        
        try:
            # Convert PDF to images
            images = self._convert_pdf_to_images(input_data["file_path"])
            
            if not images:
                return f"Error: Failed to convert PDF to images"
            
            # Process each image with the OCR API
            extraction_results = []
            for image_data in images:
                result = self._extract_text_from_image(image_data)
                extraction_results.append(result)
            
            # Generate output filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = Path(file_path).stem
            output_file = f"{base_name}_ocr_{timestamp}.md"
            
            # Save extracted content to markdown file
            self._save_to_markdown(extraction_results, output_file)
            
            # Return success message
            successful_pages = sum(1 for r in extraction_results if r["success"])
            output = f"PDF OCR extraction completed!\n\n"
            output += f"Source: {input_data['file_path']}\n"
            output += f"Results saved to: {output_file}\n"
            output += f"Successfully processed {successful_pages} of {len(extraction_results)} pages\n"
            
            # Add error info for pages that failed
            failed_pages = [r for r in extraction_results if not r["success"]]
            if failed_pages:
                output += f"\nWarning: {len(failed_pages)} pages failed processing:\n"
                for page in failed_pages[:3]:  # Show first 3 errors at most
                    output += f"  - Page {page['page_number']}: {page['error']}\n"
                if len(failed_pages) > 3:
                    output += f"  - ... and {len(failed_pages) - 3} more errors\n"
            
            return output
            
        except Exception as e:
            return f"Error processing PDF: {str(e)}"


def main():
    """Main function to run the document parsing tool."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert PDF to images and extract text using OCR API')
    parser.add_argument('file_path', help='Path to the PDF file to be processed')
    parser.add_argument('--api_url', 
                        default='https://sweeping-moth-probably.ngrok-free.app/parse_image/image',
                        help='URL of the image OCR API')
    
    args = parser.parse_args()
    
    # Create the document parsing tool
    doc_parser = DocumentParsingTool()
    
    # Override the API URL if provided
    if args.api_url:
        doc_parser.pdf_to_image_api_url = args.api_url
    
    # Parse the PDF
    try:
        result = doc_parser.parse_pdf_as_images(args.file_path)
        print(result)
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()