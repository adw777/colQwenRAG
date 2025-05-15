#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import required libraries
import os
import torch
import time
from qdrant_client import QdrantClient
from qdrant_client.http import models
from datasets import load_dataset
from colpali_engine.models import ColPali, ColPaliProcessor
from PIL import Image
from dotenv import load_dotenv

output_dir = "search_results_images"
os.makedirs(output_dir, exist_ok=True)

load_dotenv()

# Set up Qdrant client
# Replace with your own Qdrant URL and API key
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# Alternatively, use in-memory instance for testing
# qdrant_client = QdrantClient(":memory:")

# Collection name
collection_name = "ufo-binary"

# Initialize ColPali model and processor
model_name = "vidore/colpaligemma-3b-pt-448-base" #davanstrien/finetune_colpali_v1_2-ufo-4bit
colpali_model = ColPali.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",  # Use "cuda:0" for GPU, "cpu" for CPU, or "mps" for Apple Silicon
)
colpali_processor = ColPaliProcessor.from_pretrained(
    "vidore/colpaligemma-3b-pt-448-base"
)

# Load the dataset (if you want to display results)
dataset = load_dataset("davanstrien/ufo-ColPali", split="train")

def search_documents(query_text, limit=10):
    """
    Search for documents using a text query
    
    Args:
        query_text (str): The text query
        limit (int): Maximum number of results to return
        
    Returns:
        search_result: Qdrant search results
        elapsed_time: Time taken for the search in seconds
    """
    # Process the query
    with torch.no_grad():
        batch_query = colpali_processor.process_queries([query_text]).to(
            colpali_model.device
        )
        query_embedding = colpali_model(**batch_query)
    
    # Convert to multivector format
    multivector_query = query_embedding[0].cpu().float().numpy().tolist()
    
    # Search in Qdrant with timing
    start_time = time.time()
    search_result = qdrant_client.query_points(
        collection_name=collection_name,
        query=multivector_query,
        limit=limit,
        timeout=100,
        search_params=models.SearchParams(
            quantization=models.QuantizationSearchParams(
                ignore=False,
                rescore=True,
                oversampling=2.0,
            )
        )
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    return search_result, elapsed_time

# Example usage
# if __name__ == "__main__":
#     query = "top secret"
#     results, search_time = search_documents(query)
    
#     print(f"Search completed in {search_time:.4f} seconds")
#     print(f"Found {len(results.points)} results")
    
#     # Display top result IDs
#     for i, point in enumerate(results.points[:5]):
#         print(f"Result {i+1}: Document ID {point.id}, Score: {point.score}")
        
#         # If you want to display the actual images (requires dataset)
#         # Uncomment the following line if you have a display environment
#         display(dataset[point.id]["image"])

if __name__ == "__main__":
    query = "top secret"
    results, search_time = search_documents(query)
    
    print(f"Search completed in {search_time:.4f} seconds")
    print(f"Found {len(results.points)} results")
    
    # Display and save top result images
    for i, point in enumerate(results.points[:5]):
        print(f"Result {i+1}: Document ID {point.id}, Score: {point.score}")
        
        image_data = dataset[point.id]["image"]

        # Load image from path or use directly
        if isinstance(image_data, str):
            image = Image.open(image_data)
        else:
            image = image_data

        # Save image as PNG (you can use .jpeg too)
        save_path = os.path.join(output_dir, f"result_{i+1}_doc_{point.id}.png")
        image.save(save_path)
        print(f"Image saved to: {save_path}")