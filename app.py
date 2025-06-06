from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from query_with_response import search_documents
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ColPali Search API")

class QueryRequest(BaseModel):
    query: str
    limit: int = 10

class QueryResponse(BaseModel):
    search_time: float
    ai_response: str
    extracted_text: str

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    try:
        # Call search_documents from query_response.py
        results = search_documents(
            query_text=request.query,
            limit=request.limit
        )
        
        return QueryResponse(
            search_time=results["search_time"],
            ai_response=results["ai_response"],
            extracted_text=results["extracted_text"]
        )
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error processing your query"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)