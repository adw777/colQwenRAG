import os
import torch
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm
from datasets import load_dataset
import stamina
from colpali_engine.models import ColPali, ColPaliProcessor, ColQwen2, ColQwen2Processor
from dotenv import load_dotenv

load_dotenv()

# Set environment variable for faster downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Set up Qdrant client
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    https=True,
    port=None,
    prefer_grpc=True,
    timeout=60,
)

# Create Qdrant collection with binary quantization - run it once
collection_name = "testColPali"
qdrant_client.create_collection(
    collection_name=collection_name,
    on_disk_payload=True,  # store the payload on disk
    vectors_config=models.VectorParams(
        size=128,
        distance=models.Distance.COSINE,
        on_disk=True,  # move original vectors to disk
        multivector_config=models.MultiVectorConfig(
            comparator=models.MultiVectorComparator.MAX_SIM
        ),
        quantization_config=models.BinaryQuantization(
            binary=models.BinaryQuantizationConfig(
                always_ram=True
            ),
        ),
    ),
)

# Initialize model and processor
model = ColQwen2.from_pretrained(
    "vidore/colqwen2-v0.1",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",  
)
processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v0.1")

# Load the dataset
dataset = load_dataset("axondendriteplus/Legal-AI-K-Hub", split="train") 

# Define retry mechanism for upsert operations
# @stamina.retry(on=Exception, attempts=3)
def upsert_to_qdrant(points):
    try:
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )
    except Exception as e:
        print(f"Error during upsert: {e}")
        return False
    return True

# Process and upload vectors in batches
batch_size = 4  # Adjust based on your GPU memory constraints

# Use tqdm to create a progress bar
with tqdm(total=len(dataset), desc="Indexing Progress") as pbar:
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]

        # The images are already PIL Image objects, so we can use them directly
        images = batch["image"]

        # Process and encode images
        with torch.no_grad():
            batch_images = processor.process_images(images).to(
                model.device
            )
            image_embeddings = model(**batch_images)

        # Prepare points for Qdrant
        points = []
        for j, embedding in enumerate(image_embeddings):
            # Convert the embedding to a list of vectors
            multivector = embedding.cpu().float().numpy().tolist()
            points.append(
                models.PointStruct(
                    id=i + j,  # we just use the index as the ID
                    vector=multivector  # This is now a list of vectors
                )
            )

        # Upload points to Qdrant
        try:
            upsert_to_qdrant(points)
        except Exception as e:
            print(f"Error during upsert: {e}")
            continue

        # Update the progress bar
        pbar.update(batch_size)

print("Indexing complete!")

# optional
# Update collection settings
# qdrant_client.update_collection(
#     collection_name=collection_name,
#     optimizer_config=models.OptimizersConfigDiff(indexing_threshold=10),
# )
