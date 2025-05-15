import json
import requests
import os
from pathlib import Path
from tqdm import tqdm
import time

def download_pdfs(json_file_path, output_folder="Legal_AI_K_Hub"):
    # Create output directory if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Read JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Setup progress bar
    pbar = tqdm(total=len(data), desc="Downloading PDFs")
    
    # Track successful and failed downloads
    successful = 0
    failed = []
    
    for item in data:
        pdf_url = item['pdf_link']
        arxiv_id = item['arxiv_id']
        
        # Create filename from arxiv_id
        filename = f"{arxiv_id.replace(':', '_')}.pdf"
        output_path = os.path.join(output_folder, filename)
        
        # Skip if file already exists
        if os.path.exists(output_path):
            pbar.update(1)
            successful += 1
            continue
        
        try:
            # Download PDF
            response = requests.get(pdf_url, stream=True)
            response.raise_for_status()  # Raise exception for bad status codes
            
            # Save PDF
            with open(output_path, 'wb') as pdf_file:
                pdf_file.write(response.content)
            
            successful += 1
            
            # Add delay to avoid overwhelming the server
            time.sleep(1)
            
        except Exception as e:
            failed.append((arxiv_id, str(e)))
        
        pbar.update(1)
    
    pbar.close()
    
    # Print summary
    print(f"\nDownload Summary:")
    print(f"Successfully downloaded: {successful}")
    print(f"Failed downloads: {len(failed)}")
    
    if failed:
        print("\nFailed downloads details:")
        for arxiv_id, error in failed:
            print(f"- {arxiv_id}: {error}")

if __name__ == "__main__":
    # Specify your JSON file path
    json_file = "scrape_arxiv/arxiv_extracted_data.json"
    
    # Run the download
    download_pdfs(json_file)