"""
This script generates data for the Tourist Guides Evaluation scenario using a 
retrieval-augmented generation (RAG) approach. It performs the following steps:

1. Parses command-line arguments to get the path to the configuration file.
2. Loads the configuration settings from the specified file.
3. Sets up API keys for OpenAI and Hugging Face from the configuration.
4. Defines directories for PDF data and evaluation data.
5. Loads Hugging Face models for language generation and embedding.
6. Configures the default language model and embedding model.
7. Sets up document transformations, including sentence splitting.
8. Parses documents from the specified PDF directory.
9. Preprocesses the parsed documents into nodes with the specified transformations.
10. Stores the preprocessed nodes in a simple document store.
11. Creates a BM25 retriever with default parameters and optional stemming and stopword removal.
12. Initializes a query engine using the BM25 retriever.
13. Creates an evaluator to generate and save samples, and to perform evaluations.
14. Clears memory after the evaluation is complete.

Usage:
    python baseline.py --config_path /path/to/config.yaml

Arguments:
    --config_path: Path to the configuration file 
                  (default: /home/marios/projects/RAG-App-DWS/src/scenarios/config.yaml)
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
from touarag.evaluation.evaluator import Evaluator
from touarag.prompts.template import TourAgentPromptManager

from llama_index.core.node_parser import SentenceSplitter
from llama_index.extractors.entity import EntityExtractor
from llama_index.core import Settings
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.storage.docstore import SimpleDocumentStore
import llama_index.core

import Stemmer
import nltk

nltk.download("punkt_tab") # Used in transformations
SIM_TOP_K = 5

# Timestamp constant
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Instantiate the timer
timer = ExecutionTimer()

logging.basicConfig(filename=f'/home/marios/projects/RAG-App-DWS/logs/output_baseline_{TIMESTAMP}.log',
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

# OpenAI
os.environ["OPENAI_API_KEY"] = config.get_section(section="api")["OPEN_AI_API_KEY_2"]
HF_TOKEN = config.get_section(section="api")["HF_TOKEN"]

# Data Directory
PDF_DIR = "/home/marios/projects/RAG-App-DWS/data/Tourist Guides Eval/"
EVAL_DIR = "/home/marios/projects/RAG-App-DWS/data/Tourist Guides Eval/testset_persona_100.csv"

@timer.measure
def baseline():
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
        )
    ]

    # Parse documents
    docs = parse_docs(PDF_DIR)

    # Preprocess Documents
    nodes = preprocess_nodes(docs, vector_store=None, transformations=transformations)

    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)

    # Create a BM25Retriever with default parameters
    bm25_retriever = BM25Retriever.from_defaults(
        docstore=docstore,
        similarity_top_k=SIM_TOP_K,
        # Optional: We can pass in the stemmer and set the language for stopwords
        # This is important for removing stopwords and stemming the query + text
        # The default is english for both
        stemmer=Stemmer.Stemmer("english"),
        language="english",
    )

    # Create query engine
    ret_query_engine = RetrieverQueryEngine(bm25_retriever)

    # Add personalized prompt template to the query engine
    query_engine = TourAgentPromptManager("personalized").\
        update_query_engine_template(ret_query_engine)

    # Create an evaluator
    evaluator = Evaluator(eval_dir=EVAL_DIR,
                        query_engine=query_engine,
                        sampling=False, # Testing purposes
                        scenario_label="baseline"
    )
    evaluator.generate_samples()
    evaluator.save_samples(output_dir="/home/marios/projects/RAG-App-DWS/data/Tourist Guides Eval")
    evaluator.evaluate(output_dir="/home/marios/projects/RAG-App-DWS/data/Tourist Guides Eval")

baseline()

timer.save_to_file(f'/home/marios/projects/RAG-App-DWS/logs/exectime_baseline_{TIMESTAMP}.log')
# Clear memory
clear_memory()
