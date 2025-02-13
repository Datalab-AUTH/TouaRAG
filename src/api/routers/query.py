from fastapi import APIRouter, Request, HTTPException
from enum import Enum
from pydantic import BaseModel
from typing import List
import time


class MethodName(str, Enum):
    baseline = "baseline"
    hybrid = "hybrid"
    automerge = "automerge"
    hyde = "hyde"
    graphrag = "graphrag"

class QueryRequest(BaseModel):
    query: str
    method: MethodName

class QueryResponse(BaseModel):
    response: str
    context: List[str]
    time: float

router = APIRouter()
@router.post("/query", summary="Query processed documents")
async def root(request_body: QueryRequest, request: Request = None):
    """
    Retrieve the appropriate query engine based on the method provided,
    use it to process the incoming query, and return the response along with context.

    The application state (app.state) is assumed to have each query engine stored separately:
    - app.state.baseline_query_engine
    - app.state.hybrid_query_engine
    - app.state.automerge_query_engine
    - app.state.hyde_query_engine
    - app.state.graphrag_query_engine
    """
    # Determine which query engine to use based on the method.
    method = request_body.method
    if method == MethodName.baseline:
        engine = getattr(request.app.state, "baseline_query_engine", None)
    elif method == MethodName.hybrid:
        engine = getattr(request.app.state, "hybrid_query_engine", None)
    elif method == MethodName.automerge:
        engine = getattr(request.app.state, "automerge_query_engine", None)
    elif method == MethodName.hyde:
        engine = getattr(request.app.state, "hyde_query_engine", None)
    elif method == MethodName.graphrag:
        engine = getattr(request.app.state, "graphrag_query_engine", None)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported method: {method}")

    if engine is None:
        raise HTTPException(status_code=500, detail=f"{method} query engine is not loaded in application state.")

    # Execute the query using the appropriate engine.
    # It is assumed that each engine exposes a query() method that accepts a query string
    # and returns an object (or dict) with attributes `response` and `context`.
    start = time.perf_counter() 
    result = await engine.aquery(request_body.query)
    end = time.perf_counter() 
    print(f"Query execution time: {end-start:.2f} seconds")

    # Extract from response object, the source nodes (context)
    retrieved_context = result.source_nodes[:]
    retrieved_texts = [context.text if hasattr(context, 'text') else context for context in retrieved_context]
    # print(f"Retrieved context: {retrieved_texts}")

    return QueryResponse(
        response=result.response,
        context=retrieved_texts,
        time=end-start
    )