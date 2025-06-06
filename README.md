# ColQwen RAG Pipeline

A high-performance Retrieval-Augmented Generation system leveraging ColQwen2 vision-language model for multimodal document search and analysis. The pipeline processes legal-AI research papers through visual embeddings and provides contextual responses via OpenAI integration.

## Architecture

### Core Components

**Vector Database**: Qdrant with binary quantization and multi-vector configuration
- Collection: `testColPali` with COSINE distance metric
- Quantization: Binary with always-RAM configuration
- Multi-vector comparator: MAX_SIM for optimal retrieval

**Vision Model**: ColQwen2-v0.1 (vidore/colqwen2-v0.1)
- Precision: bfloat16 for memory efficiency
- Device mapping: CUDA:0
- Embedding dimension: 128 per vector

**Dataset**: Legal-AI-K-Hub (axondendriteplus/Legal-AI-K-Hub)
- Format: HuggingFace dataset with PIL images
- Content: Legal AI research papers and documents

**OCR Engine**: External API-based text extraction
- Endpoint: ngrok-tunneled OCR service
- Input: PNG images at 300 DPI
- Output: Structured JSON with extracted text

## System Requirements

### Hardware
- NVIDIA GPU with CUDA support (minimum 8-12GB VRAM)
- 16GB+ system RAM
- 50GB+ storage for dataset and models

## Environment Configuration

Create `.env` file with required credentials:
```
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key
OPENAI_API_KEY=your_openai_api_key
HF_HUB_ENABLE_HF_TRANSFER=1
```

## Installation

```bash
# Clone repository
git clone https://github.com/adw777/colQwenRAG<repository_url>
cd colpaliRAG

# Install PyTorch with CUDA support (if CUDA available)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 # according to your CUDA version

# Install dependencies
pip install -r requirements.txt
```

## Pipeline Execution

### 1. Data Ingestion (`Ingestion.py`)

Processes and indexes documents into Qdrant vector database:

```python
# Initialize ColQwen2 model and processor
model = ColQwen2.from_pretrained("vidore/colqwen2-v0.1", torch_dtype=torch.bfloat16, device_map="cuda:0")
processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v0.1")

# Load dataset
dataset = load_dataset("axondendriteplus/Legal-AI-K-Hub", split="train")

# Process in batches (configurable batch_size=4)
# Generate multi-vector embeddings
# Upsert to Qdrant with retry mechanism
```

**Key Features**:
- Batch processing with progress tracking
- Retry mechanism with stamina decorator
- Memory-efficient processing with torch.no_grad()
- Binary quantization for storage optimization

### 2. Query Processing (`query_with_response.py`)

Handles search queries and generates AI responses:

```python
def search_documents(query_text, limit=5):
    # 1. Encode query using ColQwen2
    # 2. Perform vector search in Qdrant
    # 3. Extract text from retrieved images via OCR
    # 4. Generate contextual response using OpenAI GPT-4o-mini
    # 5. Return structured results with timing metrics
```

**Search Parameters**:
- Quantization search with rescore=True
- Oversampling factor: 2.0
- Timeout: 100 seconds
- Binary quantization ignore=False

### 3. API Service (`app.py`)

FastAPI-based REST service for query processing:

```python
POST /query
{
    "query": "string",
    "limit": 10  # optional, default 10
}

Response:
{
    "search_time": float,
    "ai_response": "string",
    "extracted_text": "string"
}
```

### 4. Streamlit Interface (`streamlit_app.py`)

Interactive Streamlit frontend for user queries:
- Chat-based interface with session state management
- Real-time API communication
- Error handling and timeout management
- Response formatting and display

## Document Processing Pipeline

### OCR Integration (`parsing.py`)

**DocumentParsingTool** class provides:
- PDF to image conversion (300 DPI PNG)
- API-based OCR text extraction
- Markdown output generation
- Error handling and retry logic

**Process Flow**:
1. Validate PDF input
2. Convert PDF pages to PIL images
3. Send images to OCR API endpoint
4. Parse JSON responses
5. Generate structured markdown output

### Data Acquisition (`dataset/download_pdfs.py`)

Automated PDF download from ArXiv:
- JSON metadata parsing
- Concurrent download with rate limiting
- Progress tracking and error reporting
- Duplicate detection and skipping

## Performance Optimizations

### Vector Search
- Binary quantization reduces memory footprint by 32x
- Multi-vector configuration with MAX_SIM comparator
- Quantization search parameters tuned for accuracy/speed balance

### Model Inference
- bfloat16 precision for 2x memory reduction
- Batch processing for throughput optimization
- CUDA device mapping for GPU acceleration
- torch.no_grad() context for inference efficiency

### Storage
- On-disk payload storage for large datasets
- On-disk vector storage with quantization
- Optimized collection settings for indexing

## API Endpoints

### Query Endpoint
```
POST /query
Content-Type: application/json

{
    "query": "Ethical challenges of using AI in judiciary",
    "limit": 10
}
```

### Response Format
```json
{
    "search_time": 0.1234,
    "ai_response": "Detailed analysis based on retrieved documents...",
    "extracted_text": "Combined OCR text from retrieved images..."
}
```

### Development
```bash
# Start API server
python app.py

# Launch Streamlit interface
streamlit run streamlit_app.py
```