import os
import torch
import time
from qdrant_client import QdrantClient
from qdrant_client.http import models
from datasets import load_dataset
from colpali_engine.models import ColPali, ColPaliProcessor, ColQwen2, ColQwen2Processor
from PIL import Image
from dotenv import load_dotenv

output_dir = "search_results_LEGAL_AI"
os.makedirs(output_dir, exist_ok=True)

load_dotenv()

# Set up Qdrant client
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# Alternatively, use in-memory instance for testing
# qdrant_client = QdrantClient(":memory:")

# Collection name
collection_name = "testColPali"

model = ColQwen2.from_pretrained(
    "vidore/colqwen2-v0.1",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",  
)
processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v0.1")

# Load the dataset (if you want to display results)
dataset = load_dataset("axondendriteplus/Legal-AI-K-Hub", split="train")

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
        batch_query = processor.process_queries([query_text]).to(
            model.device
        )
        query_embedding = model(**batch_query)
    
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

if __name__ == "__main__":
    query = "Ethical challenges of using AI in judiciary!"
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

        save_path = os.path.join(output_dir, f"result_{i+1}_doc_{point.id}.png")
        image.save(save_path)
        print(f"Image saved to: {save_path}")