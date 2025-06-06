import os
import torch
import time
from qdrant_client import QdrantClient
from qdrant_client.http import models
from datasets import load_dataset
from colpali_engine.models import ColQwen2, ColQwen2Processor
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI
from parsing import DocumentParsingTool

output_dir = "search_results_LEGAL_AI"
os.makedirs(output_dir, exist_ok=True)

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize document parser
doc_parser = DocumentParsingTool()

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

collection_name = "testColPali"

model = ColQwen2.from_pretrained(
    "vidore/colqwen2-v0.1",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0"
)
processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v0.1")

# Load the dataset
dataset = load_dataset("axondendriteplus/Legal-AI-K-Hub", split="train")

def extract_text_from_images(images):
    """
    Extract text from a list of images using the parsing tool
    """
    all_text = []
    for img in images:
        # Save image temporarily
        temp_path = f"temp_image_{time.time()}.png"
        img.save(temp_path)
        
        # Extract text
        result = doc_parser._extract_text_from_image({
            "index": 0,
            "image": open(temp_path, "rb"),
            "format": "png"
        })
        
        if result["success"] and "text" in result["content"]:
            all_text.append(result["content"]["text"])
        
        # Clean up temp file
        os.remove(temp_path)
    
    return "\n".join(all_text)

def get_ai_response(query, context):
    """
    Get response from OpenAI based on the query and context
    """
    prompt = f"""
    Context from images:
    {context}
    
    User Query: {query}
    
    Please provide a detailed answer based on the context above.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # or any other available model
        messages=[
            {"role": "system", "content": "You are a helpful assistant analyzing images from AI papers."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

def search_documents(query_text, limit=10):
    """
    Search for documents using a text query and process results with AI
    """
    search_result, elapsed_time = perform_search(query_text, limit)
    
    images = []
    for point in search_result.points[:limit]:
        image_data = dataset[point.id]["image"]
        if isinstance(image_data, str):
            image = Image.open(image_data)
        else:
            image = image_data
        images.append(image)
    
    # Extract text from images
    extracted_text = extract_text_from_images(images)
    
    ai_response = get_ai_response(query_text, extracted_text)
    
    return {
        "search_time": elapsed_time,
        "results": search_result,
        "extracted_text": extracted_text,
        "ai_response": ai_response
    }

def perform_search(query_text, limit):
    """
    Perform the actual vector search
    """
    with torch.no_grad():
        batch_query = processor.process_queries([query_text]).to(model.device)
        query_embedding = model(**batch_query)
    
    multivector_query = query_embedding[0].cpu().float().numpy().tolist()
    
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
    elapsed_time = time.time() - start_time
    
    return search_result, elapsed_time

if __name__ == "__main__":
    query = "Ethical challenges of using AI in judiciary!"
    results = search_documents(query)
    
    print(f"Search completed in {results['search_time']:.4f} seconds")
    print(f"Found {len(results['results'].points)} results")
    print("\nAI Response:")
    print(results['ai_response'])