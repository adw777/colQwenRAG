import json
import requests
import fitz
import os
import tempfile
import shutil
from pathlib import Path
from huggingface_hub import HfApi
from datasets import Dataset, Features, Image
import io
from PIL import Image as PILImage
import time

def download_pdf(url, temp_dir):
    """Download PDF from URL and save to temporary directory"""
    try:
        # Add headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Ensure URL is for PDF download
        url = url.replace('abs', 'pdf')  # Convert arXiv abstract URL to PDF URL
        if not url.endswith('.pdf'):
            url = url + '.pdf'
            
        print(f"Downloading from: {url}")
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        filename = url.split('/')[-1]
        filepath = os.path.join(temp_dir, filename)
        
        with open(filepath, 'wb') as f:
            f.write(response.content)
        print(f"Successfully downloaded: {filename}")
        return filepath
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None

def process_pdf(pdf_path):
    """Process single PDF and convert to images"""
    try:
        doc = fitz.open(pdf_path)
        images = []
        total_pages = len(doc)
        print(f"Processing PDF: {pdf_path} ({total_pages} pages)")

        for page_num in range(total_pages):
            print(f"Converting page {page_num + 1}/{total_pages}")
            page = doc[page_num]
            try:
                # Modify the pixmap creation process
                pix = page.get_pixmap(
                    matrix=fitz.Matrix(2, 2),  # Reduced resolution but more stable
                    alpha=False  # Disable alpha channel
                )
                
                # Convert directly to PIL Image using RGB mode
                img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Resize if needed (optional)
                # img = img.resize((800, int(800 * img.size[1] / img.size[0])), PILImage.LANCZOS)
                
                # Save as JPEG with high quality
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG', quality=95)
                img_byte_arr = img_byte_arr.getvalue()
                
                images.append({'image': img_byte_arr})
                print(f"Successfully converted page {page_num + 1}")
                
            except Exception as e:
                print(f"Error on page {page_num + 1}: {str(e)}")
                continue  # Skip this page and continue with next
        
        doc.close()
        
        if not images:
            print("No pages were successfully converted")
            return None
            
        print(f"Successfully converted {len(images)} pages")
        return images
        
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return None

def process_and_upload_to_hf(json_file, dataset_name, batch_size=10, private=False):
    """Process PDFs and upload all images to Hugging Face dataset in a single 'train' split"""
    
    # Load JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Initialize Hugging Face API
    try:
        api = HfApi()
        # Test authentication
        api.whoami()
        print("Successfully authenticated with Hugging Face")
    except Exception as e:
        print(f"Error authenticating with Hugging Face: {e}")
        print("Please run 'huggingface-cli login' first")
        return False

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")
    
    try:
        all_images = []
        
        # Process PDFs sequentially
        for i, item in enumerate(data):
            print(f"\nProcessing item {i+1}/{len(data)}")
            pdf_url = item['pdf_link']
            arxiv_id = item['arxiv_id']
            
            # Download PDF
            pdf_path = download_pdf(pdf_url, temp_dir)
            if not pdf_path:
                print(f"Skipping {arxiv_id} due to download error")
                continue
            
            # Process PDF
            images = process_pdf(pdf_path)
            if not images:
                print(f"Skipping {arxiv_id} due to processing error")
                continue
            
            # # Add metadata to images if needed
            # for img in images:
            #     img['arxiv_id'] = arxiv_id  # Add metadata if desired
            
            # Add images to collection
            all_images.extend(images)
            print(f"Added {len(images)} images from {arxiv_id}")
            
            # Clean up PDF file
            os.remove(pdf_path)
            
            # Print progress
            print(f"Total images collected so far: {len(all_images)}")
            
            # Optional: Upload in batches to avoid memory issues but still use 'train' split
            if len(all_images) >= batch_size:
                print(f"\nUploading batch of {len(all_images)} images to 'train' split")
                # Create dataset from current batch
                dataset_dict = {'image': [img['image'] for img in all_images]}
                
                # # Add metadata if needed
                # if 'arxiv_id' in all_images[0]:
                #     dataset_dict['arxiv_id'] = [img['arxiv_id'] for img in all_images]
                
                features = Features({'image': Image()})
                dataset = Dataset.from_dict(dataset_dict, features=features)
                
                # Push to hub with train split (will be concatenated automatically)
                dataset.push_to_hub(
                    dataset_name,
                    split="train",
                    private=private
                )
                
                all_images = []
                print(f"Successfully uploaded batch to 'train' split")
                time.sleep(2)  # Rate limiting
        
        # Upload remaining images
        if all_images:
            print(f"\nUploading final batch of {len(all_images)} images to 'train' split")
            dataset_dict = {'image': [img['image'] for img in all_images]}
            
            # Add metadata if needed
            if 'arxiv_id' in all_images[0]:
                dataset_dict['arxiv_id'] = [img['arxiv_id'] for img in all_images]
                
            features = Features({'image': Image()})
            dataset = Dataset.from_dict(dataset_dict, features=features)
            
            dataset.push_to_hub(
                dataset_name,
                split="train",
                private=private
            )
            print("Successfully uploaded final batch to 'train' split")

        print(f"\nAll processing complete! All images uploaded to the 'train' split.")
        return True

    except Exception as e:
        print(f"An error occurred: {e}")
        return False
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print("Cleaned up temporary directory")

def main():
    json_file = "scrape_arxiv/arxiv_extracted_data.json"
    dataset_name = "axondendriteplus/Legal-LLM-K-Hub"
    batch_size = 10  # Increased batch size, still using 'train' split
    private = False
    
    process_and_upload_to_hf(
        json_file,
        dataset_name,
        batch_size=batch_size,
        private=private
    )

if __name__ == "__main__":
    main()