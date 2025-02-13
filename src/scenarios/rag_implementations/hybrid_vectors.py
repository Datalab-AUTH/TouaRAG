"""
This module implements a hybrid vector-based retrieval system for evaluating tourist guides.
It utilizes both dense and sparse retrieval methods to enhance the accuracy of document retrieval.
The module includes configurations for database connections, API keys, and various processing steps.

Main functionalities:
- Load configurations from a YAML file.
- Set up PostgreSQL database with PGVector.
- Load and preprocess documents using Hugging Face models.
- Create a hybrid retriever combining dense and sparse retrieval methods.
- Evaluate the retrieval system using a custom evaluator.

Dependencies:
- touarag library for configurations, utilities, preprocessing, and evaluation.
- llama_index library for node parsing, entity extraction, and vector store indexing.
- Hugging Face models for language model and embedding.
- PostgreSQL for database storage.
- nltk for natural language processing tasks.
- Stemmer for stemming operations.

Usage:
Run the script with the appropriate configuration file to generate and evaluate samples
for the Tourist Guides Evaluation scenario.
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
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.retrievers.bm25 import BM25Retriever
import llama_index.core
import Stemmer

import nltk
nltk.download("punkt_tab") # Used in transformations

# Fusion mode for hybrid retriever
# Available modes: "reciprocal_rerank", "relative_score", "dist_based_score", "simple"
FUSION_MODE = "reciprocal_rerank"
SIM_TOP_K = 5

# Timestamp constant
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Instantiate the timer
timer = ExecutionTimer()

logging.basicConfig(filename=f'/home/marios/projects/RAG-App-DWS/logs/output_hybvec_{TIMESTAMP}.log', level=logging.DEBUG)
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
def hybvec():
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

    # Preprocess Documents
    nodes = preprocess_nodes(docs, pg_vector_store, transformations=transformations)

    index = VectorStoreIndex.from_vector_store(vector_store=pg_vector_store)

    # Create a BM25Retriever with default parameters
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=SIM_TOP_K,
        # Optional: We can pass in the stemmer and set the language for stopwords
        # This is important for removing stopwords and stemming the query + text
        # The default is english for both
        stemmer=Stemmer.Stemmer("english"),
        language="english",
    )

    # Hybrid retriever with both sparse and dense retrieval
    retriever = QueryFusionRetriever(
        [
            index.as_retriever(similarity_top_k=SIM_TOP_K),
            bm25_retriever,
        ],
        num_queries=4,
        mode=FUSION_MODE,
        use_async=True,
    )

    # Create the new hybrid query engine
    hyb_query_engine = RetrieverQueryEngine(retriever)

    # Add personalized prompt template to the query engine
    query_engine = TourAgentPromptManager("personalized")\
        .update_query_engine_template(hyb_query_engine)

    # Create an evaluator
    evaluator = Evaluator(eval_dir=EVAL_DIR,
                        query_engine=query_engine,
                        sampling=False, # Testing purposes
                        scenario_label="hybridvec"
    )
    evaluator.generate_samples()
    evaluator.save_samples(output_dir="/home/marios/projects/RAG-App-DWS/data/Tourist Guides Eval")
    evaluator.evaluate(output_dir="/home/marios/projects/RAG-App-DWS/data/Tourist Guides Eval")

hybvec()

timer.save_to_file(f'/home/marios/projects/RAG-App-DWS/logs/exectime_hybvec_{TIMESTAMP}.log')
# Clear memory
clear_memory()
