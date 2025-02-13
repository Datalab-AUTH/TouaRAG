"""
This module implements the HyDE (Hybrid Document Embedding) scenario for the Tourist 
Guides Evaluation project.

The script performs the following steps:
1. Parses command-line arguments to get the configuration file path.
2. Sets up logging to both a file and the console.
3. Loads the configuration from the specified YAML file.
4. Configures database connection parameters and API keys.
5. Loads Hugging Face models for language and embedding tasks.
6. Sets up document transformations including sentence splitting and entity extraction.
7. Initializes a PostgreSQL PGVector database for storing vector embeddings.
8. Parses and preprocesses documents from a specified directory.
9. Creates a query engine using the vector store and applies the HyDE query transformation.
10. Adds a personalized prompt template to the query engine.
11. Initializes an evaluator to generate and save samples, and to perform evaluations.
12. Clears memory after processing.

Dependencies:
- touarag library for configuration, utilities, preprocessing, LLM loading, 
    database setup, evaluation, and prompt management.
- llama_index library for document parsing, entity extraction, and query transformations.
- nltk for sentence tokenization.
- argparse, os, datetime, logging, and sys for standard operations.

Usage:
    python hyde.py --config_path /path/to/config.yaml
"""
import os
import datetime
import argparse
import logging
import sys

from touarag.configs.configurator import Config
from touarag.common.utils import parse_docs, clear_memory, ExecutionTimer
from touarag.common.preprocess import preprocess_nodes
from touarag.llm.huggingface import load_hf_llm, load_hf_embed
from touarag.common.stores import setup_postgresql_database
from touarag.evaluation.evaluator import Evaluator
from touarag.prompts.template import TourAgentPromptManager

from llama_index.core.node_parser import SentenceSplitter
from llama_index.extractors.entity import EntityExtractor
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine, RetrieverQueryEngine
import llama_index.core
from llama_index.core.retrievers import VectorIndexRetriever

import nltk
nltk.download("punkt_tab") # Used in transformations

SIM_TOP_K = 5

# Timestamp constant
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Instantiate the timer
timer = ExecutionTimer()

logging.basicConfig(filename=f'/home/marios/projects/RAG-App-DWS/logs/output_hyde_{TIMESTAMP}.log',
                    level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
llama_index.core.set_global_handler("simple")


parser = argparse.ArgumentParser(description="Generate data for \
                                 the Tourist Guides Evaluation scenario")
parser.add_argument("--config_path",
                    type=str,
                    default="/home/marios/projects/RAG-App-DWS/src/scenarios/config.yaml",
                    help="Path to the configuration file")

args = parser.parse_args()

# Setup configuration
config = Config(args.config_path)

# DB config
DB_NAME = config.get_section(section="db")["DB_NAME"]
DB_USER = config.get_section(section="db")["USER"]
DB_PASS = config.get_section(section="db")["PASSWORD"]
DB_HOST = config.get_section(section="db")["HOST"]
DB_PORT = config.get_section(section="db")["PORT"]

# OpenAI
os.environ["OPENAI_API_KEY"] = config.get_section(section="api")["OPEN_AI_API_KEY_2"]
HF_TOKEN = config.get_section(section="api")["HF_TOKEN"]

# Data Directory
PDF_DIR = "/home/marios/projects/RAG-App-DWS/data/Tourist Guides Eval/"
EVAL_DIR = "/home/marios/projects/RAG-App-DWS/data/Tourist Guides Eval/testset_persona_100.csv"

@timer.measure
def hyde():
    # Load Hugging Face models
    llm = load_hf_llm(hf_token=HF_TOKEN)
    embed_model = load_hf_embed(hf_token=HF_TOKEN)

    # Setup Default LLM and Embed Model
    Settings.llm = llm
    Settings.embed_model = embed_model

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

    # Setup PostgreSQL PGVector database
    pg_vector_store = setup_postgresql_database(DB_USER, DB_PASS, DB_HOST, DB_PORT, DB_NAME)

    # Parse documents
    docs = parse_docs(PDF_DIR)

    # Preprocess Documents (Passed directly on vector store)
    _ = preprocess_nodes(docs, pg_vector_store, transformations=transformations)

    # Create query engine with vector store
    index = VectorStoreIndex.from_vector_store(vector_store=pg_vector_store)

    # Configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=SIM_TOP_K,
    )

    pg_query_engine = RetrieverQueryEngine(
        retriever=retriever
    )

    # Add personalized prompt template to the query engine
    pers_query_engine = TourAgentPromptManager("personalized")\
        .update_query_engine_template(pg_query_engine)

    # Create a HyDE query engine
    hyde_transform = HyDEQueryTransform(include_original=True)
    hyde_query_engine = TransformQueryEngine(pers_query_engine, hyde_transform)

    # Create an evaluator
    evaluator = Evaluator(eval_dir=EVAL_DIR,
                        query_engine=hyde_query_engine,
                        sampling=False, # Testing purposes
                        scenario_label="hyde"
    )
    evaluator.generate_samples()
    evaluator.save_samples(output_dir="/home/marios/projects/RAG-App-DWS/data/Tourist Guides Eval")
    evaluator.evaluate(output_dir="/home/marios/projects/RAG-App-DWS/data/Tourist Guides Eval")

hyde()

timer.save_to_file(f'/home/marios/projects/RAG-App-DWS/logs/exectime_hyde_{TIMESTAMP}.log')
# Clear memory
clear_memory()
