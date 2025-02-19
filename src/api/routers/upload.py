from fastapi import APIRouter, UploadFile, File, Request
from llama_index.readers.file import PyMuPDFReader
from typing import List
import os


from llama_index.core.node_parser import SentenceSplitter, HierarchicalNodeParser
from llama_index.extractors.entity import EntityExtractor
from llama_index.core import Settings
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine, TransformQueryEngine
from llama_index.core.storage.docstore import SimpleDocumentStore
from touarag.prompts.template import TourAgentPromptManager
import llama_index.core
import Stemmer
import nltk
import torch
import gc


from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.retrievers import QueryFusionRetriever, AutoMergingRetriever, VectorIndexRetriever
from llama_index.core.node_parser import get_leaf_nodes, get_root_nodes
from touarag.common.stores import setup_postgresql_database


from llama_index.core.ingestion import IngestionPipeline


from touarag.configs.configurator import Config
import asyncio
import time
import shutil

CONFIG_PATH = "/home/marios/projects/RAG-App-DWS/src/api/config.yaml"
SETTINGS_GRAPHRAG_PATH = "/home/marios/projects/RAG-App-DWS/src/api/"
# Setup configuration
config = Config(CONFIG_PATH)

# DB config
DB_NAME = config.get_section(section="db")["DB_NAME"]
DB_USER = config.get_section(section="db")["USER"]
DB_PASS = config.get_section(section="db")["PASSWORD"]
DB_HOST = config.get_section(section="db")["HOST"]
DB_PORT = config.get_section(section="db")["PORT"]


LOAD_DIR = "temp_data"

def parse_docs(directory):
    # Get a list of all PDF files in the directory
    pdf_files = [os.path.join(directory, f)\
                for f in os.listdir(directory) if f.endswith('.pdf')]

    documents = []
    for pdf_file in pdf_files:
        loader = PyMuPDFReader()
        documents.extend(loader.load(pdf_file))
    return documents


async def preprocess_nodes(docs, vector_store, transformations):
    # Create the ingestion pipeline with the provided transformations and vector store
    pipeline = IngestionPipeline(
        transformations=transformations,
        vector_store=vector_store
    )

    # Execute the pipeline asynchronously on the provided documents
    nodes = await pipeline.arun(documents=docs)
    return nodes

def clear_memory():
    """
    Clears the CUDA memory cache and performs garbage collection.

    This function empties the CUDA memory cache using `torch.cuda.empty_cache()`
    and then triggers Python's garbage collector using `gc.collect()`. After
    clearing the memory, it prints a summary of the CUDA memory usage.

    Note:
        This function is useful for freeing up GPU memory in between different
        stages of a program or after certain operations to ensure efficient
        memory usage.

    Returns:
        None
    """
    torch.cuda.empty_cache()
    gc.collect()
    print(torch.cuda.memory_summary(device=None, abbreviated=False))

router = APIRouter()

@router.post("/properties", summary="Send personalization profile")
async def fetch_properties(properties: dict = None, request: Request = None):
    """
    Fetch the personalization profile and store it in the application state.
    """
    request.app.state.properties = properties
    return {"message": "Personalization profile received."}
 
@router.post("/upload/{top_k}", summary="Upload and process a document")
async def upload_documents(files: List[UploadFile] = File(...),top_k: int =  5, request: Request = None):
    """
    Upload one or more PDF documents, save them to disk under the 'temp_data' folder, 
    process each for all architecture approaches, and parse all PDF files in the directory.

    For each file:
    - The file is saved to the 'temp_data' folder.
    - A unique document ID is generated using a hash of its contents.
    - If the document is already processed, the stored document ID is returned.
    - Otherwise, the file is processed and stored.
    
    Finally, parse_docs() is called on the 'temp_data' folder to convert all PDFs to document objects.
    """
    # Ensure the temporary data directory exists.
    os.makedirs(LOAD_DIR, exist_ok=True)

    # Clear all files and subdirectories in LOAD_DIR
    for filename in os.listdir(LOAD_DIR):
        file_path = os.path.join(LOAD_DIR, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

    clear_memory()

    # Check if request.app.state.processed_doc_ids exists, otherwise initialize it
    if not hasattr(request.app.state, 'properties'):
        request.app.state.properties = None
    
    for file in files:
        # Construct the file path where the PDF will be stored.
        file_path = os.path.join(LOAD_DIR, file.filename)
        
        # Save the file to disk.
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Read the file bytes from disk (or reuse the already-read content).
        with open(file_path, "rb") as f:
            file_bytes = f.read()
    
    # After processing, parse the PDF directory to generate document objects.
    parsed_docs = parse_docs(LOAD_DIR)
    request.app.state.parsed_docs = parsed_docs
    
    # Start timing
    start = time.perf_counter() 

    # Run the query engine creation tasks in parallel using asyncio.gather directly
    baseline_query_engine, hyde_query_engine, hybrid_query_engine, automerge_query_engine = await asyncio.gather(
        baseline(top_k, request),
        hyde(top_k, request),
        hybrid(top_k, request),
        automerge(top_k, request)
    )

    # Calculate execution time
    end = time.perf_counter() 
    print(f"Total execution time: {end-start:.2f} seconds")

    # Store the results in the request state
    request.app.state.baseline_query_engine = baseline_query_engine
    request.app.state.hyde_query_engine = hyde_query_engine
    request.app.state.hybrid_query_engine = hybrid_query_engine
    request.app.state.automerge_query_engine = automerge_query_engine
    clear_memory()
    
    # Return both the per-file processing responses and the parsed documents.
    return {"message": "Documents uploaded and processed successfully."}


async def baseline(top_k, request: Request = None):
    Settings.llm = request.app.state.llm

    nltk.download("punkt_tab") # Used in transformations

    docs = request.app.state.parsed_docs.copy()

    # Setup Transformations
    transformations = [
        SentenceSplitter(chunk_size=400,# Split sentences into chunks of size 400 with an overlap of 50
                        chunk_overlap=50
        ),
        EntityExtractor(
            prediction_threshold=0.5,
            label_entities=False,  # include the entity label in the metadata (can be erroneous)
            device="cuda",
        )
    ]

    nodes = await asyncio.to_thread(asyncio.run, preprocess_nodes(docs, None, transformations))

    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)

    # Create a BM25Retriever with default parameters
    bm25_retriever = BM25Retriever.from_defaults(
        docstore=docstore,
        similarity_top_k=top_k,
        # Optional: We can pass in the stemmer and set the language for stopwords
        # This is important for removing stopwords and stemming the query + text
        # The default is english for both
        stemmer=Stemmer.Stemmer("english"),
        language="english",
    )

    # Create query engine
    ret_query_engine = RetrieverQueryEngine(bm25_retriever)

    # Add personalized prompt template to the query engine
    query_engine = TourAgentPromptManager("personalized", request.app.state.properties).\
        update_query_engine_template(ret_query_engine)
    
    print("Baseline query engine created.")
    
    return query_engine


async def hyde(top_k, request: Request = None):
    Settings.llm = request.app.state.llm
    Settings.embed_model = request.app.state.embed_model

    docs = request.app.state.parsed_docs.copy()
    embed_model = request.app.state.embed_model

    # Setup Transformations
    transformations = [
        SentenceSplitter(chunk_size=400,# Split sentences into chunks of size 400 with an overlap of 50
                        chunk_overlap=50
        ),
        EntityExtractor(
            prediction_threshold=0.5,
            label_entities=False,  # include the entity label in the metadata (can be erroneous)
            device="cuda",
        ),
        embed_model
    ]

    # Determine embed_dim based on provider
    embed_dim = 1536 if request.app.state.provider == "openai" else 1024

    # Setup PostgreSQL PGVector database
    pg_vector_store = setup_postgresql_database(DB_USER, DB_PASS, DB_HOST, DB_PORT, DB_NAME+"_hyde", embed_dim=embed_dim)

    # Preprocess Documents (Passed directly on vector store)
    # _ = await preprocess_nodes(docs, pg_vector_store, transformations=transformations)
    # _ = await asyncio.to_thread(asyncio.run, preprocess_nodes(docs, pg_vector_store, transformations))
    nodes = await asyncio.to_thread(asyncio.run, preprocess_nodes(docs, None, transformations))

    pg_vector_store.add(nodes)

    # Create query engine with vector store
    index = VectorStoreIndex.from_vector_store(vector_store=pg_vector_store)

    # Configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
    )

    pg_query_engine = RetrieverQueryEngine(
        retriever=retriever
    )

    # Add personalized prompt template to the query engine
    pers_query_engine = TourAgentPromptManager("personalized", request.app.state.properties)\
        .update_query_engine_template(pg_query_engine)

    # Create a HyDE query engine
    hyde_transform = HyDEQueryTransform(include_original=True)
    hyde_query_engine = TransformQueryEngine(pers_query_engine, hyde_transform)
    print("HyDE query engine created.")
    return hyde_query_engine

async def hybrid(top_k, request: Request = None):
    FUSION_MODE = "reciprocal_rerank" # TODO: Add it in request parameters
    Settings.llm = request.app.state.llm
    Settings.embed_model = request.app.state.embed_model

    docs = request.app.state.parsed_docs.copy()
    embed_model = request.app.state.embed_model

    # Setup Transformations
    transformations = [
        SentenceSplitter(chunk_size=400,# Split sentences into chunks of size 400 with an overlap of 50
                        chunk_overlap=50
        ),
        EntityExtractor(
            prediction_threshold=0.5,
            label_entities=False,  # include the entity label in the metadata (can be erroneous)
            device="cuda",
        ),
        embed_model
    ]

    # Determine embed_dim based on provider
    embed_dim = 1536 if request.app.state.provider == "openai" else 1024

    # Setup PostgreSQL PGVector database
    pg_vector_store = setup_postgresql_database(DB_USER, DB_PASS, DB_HOST, DB_PORT, DB_NAME+"_hybrid", embed_dim=embed_dim)

    # Preprocess Documents
    # nodes = await preprocess_nodes(docs, pg_vector_store, transformations=transformations)
    nodes = await asyncio.to_thread(asyncio.run, preprocess_nodes(docs, None, transformations))
    pg_vector_store.add(nodes)

    index = VectorStoreIndex.from_vector_store(vector_store=pg_vector_store)

    # Create a BM25Retriever with default parameters
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=top_k,
        stemmer=Stemmer.Stemmer("english"),
        language="english",
    )

    # Hybrid retriever with both sparse and dense retrieval
    retriever = QueryFusionRetriever(
        [
            index.as_retriever(similarity_top_k=top_k),
            bm25_retriever,
        ],
        num_queries=4,
        mode=FUSION_MODE,
        use_async=True,
    )

    # Create the new hybrid query engine
    hyb_query_engine = RetrieverQueryEngine(retriever)

    # Add personalized prompt template to the query engine
    query_engine = TourAgentPromptManager("personalized", request.app.state.properties)\
        .update_query_engine_template(hyb_query_engine)
    
    print("Hybrid query engine created.")
    
    return query_engine


async def automerge(top_k, request: Request = None):
    Settings.llm = request.app.state.llm
    Settings.embed_model = request.app.state.embed_model

    docs = request.app.state.parsed_docs.copy()
    embed_model = request.app.state.embed_model

    hier_node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[2048, 512, 128]
    )

    # Setup Transformations
    transformations = [
        hier_node_parser,
        SentenceSplitter(chunk_size=400,# Split sentences into chunks of size 400 with an overlap of 50
                        chunk_overlap=50
        ),
        EntityExtractor(
            prediction_threshold=0.5,
            label_entities=False,  # include the entity label in the metadata (can be erroneous)
            device="cuda",
        ),
        embed_model
    ]

    # Determine embed_dim based on provider
    embed_dim = 1536 if request.app.state.provider == "openai" else 1024

    # Setup PostgreSQL PGVector database
    pg_vector_store = setup_postgresql_database(DB_USER, DB_PASS, DB_HOST, DB_PORT, DB_NAME+"_automerge", embed_dim=embed_dim)

    # Preprocess Documents
    # nodes = await preprocess_nodes(docs, pg_vector_store, transformations=transformations)
    nodes = await asyncio.to_thread(asyncio.run, preprocess_nodes(docs, pg_vector_store, transformations))

    leaf_nodes = get_leaf_nodes(nodes)

    # Add nodes to the vector store
    pg_vector_store.add(nodes)

    # Setup Storage Context
    storage_context = StorageContext.from_defaults(vector_store=pg_vector_store)

    # Uses embeddings model predefined along with Vector Store defined (pgvector)
    index = VectorStoreIndex(
        leaf_nodes, storage_context=storage_context, show_progress=True
    )

    base_retriever = index.as_retriever(similarity_top_k=top_k)
    retriever = AutoMergingRetriever(base_retriever, storage_context, verbose=True)
    automerge_query_engine = RetrieverQueryEngine.from_args(retriever)

    # Add personalized prompt template to the query engine
    query_engine = TourAgentPromptManager("personalized", request.app.state.properties)\
        .update_query_engine_template(automerge_query_engine)
    
    print("Automerge query engine created.")
    return query_engine


@router.get("/loaded_files", summary="Get loaded files")
async def get_loaded_files():
    """
    Get a list of all files currently loaded in the 'temp_data' directory.
    """
    files = [{"name": f} for f in os.listdir(LOAD_DIR) if os.path.isfile(os.path.join(LOAD_DIR, f))]
    return {"loaded_files": files}