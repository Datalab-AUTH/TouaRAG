from fastapi import FastAPI
from api.routers import upload, query
from contextlib import asynccontextmanager
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from touarag.llm.huggingface import load_hf_llm, load_hf_embed
from touarag.configs.configurator import Config
import time
import torch
import gc
import os



def load_base_openai_llm(llm_name=None):
    """
    Loads an OpenAI language model using the specified model name.

    Args:
        llm_name (str, optional): The name of the language model to load. 
                                  Defaults to "gpt-4o-mini" if not provided.

    Returns:
        ChatOpenAI: An instance of ChatOpenAI initialized with 
                    the specified or default model.
    """
    if llm_name is None:
        llm_name = "gpt-4o-mini"
    openai_llm = OpenAI(model=llm_name, max_retries=15, reuse_client=False)

    return openai_llm

def load_base_openai_embed():
    """
    Load the base OpenAI embeddings model.
    This function initializes and returns an instance of the OpenAIEmbeddings
    class with the specified model and maximum number of retries.
    Returns:
        OpenAIEmbeddings: An instance of the OpenAIEmbeddings class initialized
        with the 'text-embedding-3-small' model and a maximum of 15 retries.
    """
    model="text-embedding-3-small"
    openai_embed = OpenAIEmbedding(model=model, max_retries=15, reuse_client=False)
    return openai_embed

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


# http://127.0.0.1:8000/docs

CONFIG_PATH = "/home/marios/projects/RAG-App-DWS/src/api/config.yaml"
# Setup configuration
config = Config(CONFIG_PATH)
HF_TOKEN = config.get_section(section="api")["HF_TOKEN"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.provider = "na"
    yield

app = FastAPI(title="Multi-Architecture RAG API", lifespan=lifespan)

app.include_router(upload.router, prefix="/api")
app.include_router(query.router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Welcome to the Multi-Architecture RAG API!"} 


@app.post("/model/{provider}", summary="Change the provider of the LLM and Embedding models.")
async def model_change(provider: str, token: str):
    clear_memory()

    global llm, embed_model

    if provider == "openai":
        # Set OpenAI API key
        os.environ["OPENAI_API_KEY"] = token

    # Start timing
    start = time.perf_counter()
    print("Loading LLM and embed models...")

    if provider == "openai":
        if app.state.provider == "openai":
            return {"message": "Provider is already OpenAI."}
        # Load OpenAI models
        llm = load_base_openai_llm()
        embed_model = load_base_openai_embed()
        app.state.provider = "openai"
    elif provider == "local":
        if app.state.provider == "local":
            return {"message": "Provider is already Local."}
        # Load Hugging Face models
        llm = load_hf_llm(hf_token=token)
        embed_model = load_hf_embed(hf_token=token)
        app.state.provider = "local"

    # Setup Default LLM and Embed Model
    app.state.llm = llm
    app.state.embed_model = embed_model

    print("LLM and embed model loaded successfully.")
    # Calculate execution time
    end = time.perf_counter()
    print(f"LLM loading time: {end-start:.2f} seconds")
    del llm
    del embed_model
    clear_memory()
    return {"message": f"Provider changed to {provider}"}
