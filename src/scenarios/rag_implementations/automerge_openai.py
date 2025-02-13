"""
This module implements the data generation process for the Tourist Guides Evaluation scenario.
It sets up the necessary configurations, loads models, preprocesses documents, and evaluates
the generated data using a personalized query engine.

Modules and functionalities used:
- Configuration setup using `touarag.configs.configurator.Config`
- PostgreSQL database setup using `touarag.common.stores.setup_postgresql_database`
- Document parsing and preprocessing using `touarag.common.utils` and `touarag.common.preprocess`
- Hugging Face models loading using `touarag.llm.huggingface`
- Node parsing and transformations using `llama_index.core.node_parser` and `llama_index.extractors.entity`
- Vector store index and query engine setup using `llama_index.core`
- Evaluation using `touarag.evaluation.evaluator.Evaluator`
- Prompt management using `touarag.prompts.template.TourAgentPromptManager`

Constants:
- SIM_TOP_K: Number of top similar documents to retrieve
- TIMESTAMP: Current timestamp for logging and output file naming

Arguments:
- --config_path: Path to the configuration file

Steps:
1. Parse command-line arguments and load configuration.
2. Set up PostgreSQL database connection.
3. Load Hugging Face models for language and embedding.
4. Define transformations for document preprocessing.
5. Parse and preprocess documents.
6. Add preprocessed nodes to the vector store.
7. Set up storage context and vector store index.
8. Create a query engine with personalized prompt template.
9. Evaluate the generated samples and save the results.
10. Clear memory to free up resources.
"""
import os
import datetime
import argparse
import logging
import sys

from touarag.configs.configurator import Config
from touarag.common.utils import parse_docs, clear_memory, ExecutionTimer
from touarag.common.preprocess import preprocess_nodes
from touarag.llm.openai import load_base_openai_llm, load_base_openai_embed
from touarag.common.stores import setup_postgresql_database
from touarag.evaluation.evaluator import Evaluator
from touarag.prompts.template import TourAgentPromptManager

from llama_index.core.node_parser import SentenceSplitter, HierarchicalNodeParser
from llama_index.extractors.entity import EntityExtractor
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import get_leaf_nodes, get_root_nodes
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import AutoMergingRetriever
import llama_index.core

import nltk
nltk.download("punkt_tab") # Used in transformations

SIM_TOP_K = 5

# Timestamp constant
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Instantiate the timer
timer = ExecutionTimer()

logging.basicConfig(filename=f'/home/marios/projects/RAG-App-DWS/logs/output_automerge_openai_{TIMESTAMP}.log', level=logging.DEBUG)
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
def automerge():
# Load Hugging Face models
    llm = load_base_openai_llm()
    embed_model = load_base_openai_embed()

    # Setup Default LLM and Embed Model
    Settings.llm = llm
    Settings.embed_model = embed_model

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

    # Setup PostgreSQL PGVector database
    pg_vector_store = setup_postgresql_database(DB_USER, DB_PASS, DB_HOST, DB_PORT, DB_NAME, embed_dim=1536)

    # Parse documents
    docs = parse_docs(PDF_DIR)

    # Preprocess Documents
    nodes = preprocess_nodes(docs, None, transformations=transformations)

    leaf_nodes = get_leaf_nodes(nodes)

    # Add nodes to the vector store
    pg_vector_store.add(nodes)

    # Setup Storage Context
    storage_context = StorageContext.from_defaults(vector_store=pg_vector_store)

    # Uses embeddings model predefined along with Vector Store defined (pgvector)
    index = VectorStoreIndex(
        leaf_nodes, storage_context=storage_context, show_progress=True
    )

    base_retriever = index.as_retriever(similarity_top_k=SIM_TOP_K)
    retriever = AutoMergingRetriever(base_retriever, storage_context, verbose=True)
    automerge_query_engine = RetrieverQueryEngine.from_args(retriever)

    # Add personalized prompt template to the query engine
    query_engine = TourAgentPromptManager("personalized")\
        .update_query_engine_template(automerge_query_engine)

    # Create an evaluator
    evaluator = Evaluator(eval_dir=EVAL_DIR,
                        query_engine=query_engine,
                        sampling=False, # Testing purposes
                        scenario_label="automerge_openai"
    )
    evaluator.generate_samples()
    evaluator.save_samples(output_dir="/home/marios/projects/RAG-App-DWS/data/Tourist Guides Eval")
    evaluator.evaluate(output_dir="/home/marios/projects/RAG-App-DWS/data/Tourist Guides Eval")

automerge()

timer.save_to_file(f'/home/marios/projects/RAG-App-DWS/logs/exectime_automerge_openai_{TIMESTAMP}.log')
# Clear memory
clear_memory()
