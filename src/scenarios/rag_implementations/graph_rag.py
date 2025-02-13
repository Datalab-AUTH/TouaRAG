"""
This script is designed to generate data for the Tourist Guides Evaluation 
scenario using a Graph-based Retrieval-Augmented Generation (RAG) approach. 
It performs the following steps:

1. Imports necessary libraries and modules.
2. Defines constants and patterns for entity and relationship extraction.
3. Sets up logging to capture the script's output.
4. Parses command-line arguments to get the configuration file path.
5. Loads configuration settings from the specified file.
6. Sets up database and API configurations.
7. Loads Hugging Face and OpenAI language models.
8. Defines transformations for sentence splitting and entity extraction.
9. Parses and preprocesses documents from a specified directory.
10. Initializes a GraphRAGExtractor for extracting knowledge triplets.
11. Sets up a GraphRAGStore for storing the extracted knowledge graph.
12. Creates a PropertyGraphIndex to index the preprocessed nodes.
13. Adds a personalized prompt template to the query engine.
14. Initializes an Evaluator to generate and evaluate samples.
15. Clears memory after processing is complete.

Functions:
- parse_fn(response_str: str): Parses the given response string to extract 
                               entities and relationships.

Constants:
- SIM_TOP_K: Number of top similar items to consider.
- TIMESTAMP: Current timestamp for logging.
- ENT_PATTERN: Regex pattern for extracting entities.
- REL_PATTERN: Regex pattern for extracting relationships.
- KG_TRIPLET_EXTRACT_TMPL: Template for extracting knowledge triplets.

Command-line Arguments:
- --config_path: Path to the configuration file 
                 (default: "/home/marios/projects/RAG-App-DWS/src/scenarios/config.yaml").

Configuration:
- DB_NAME: Database name.
- DB_USER: Database user.
- DB_PASS: Database password.
- DB_HOST: Database host.
- DB_PORT: Database port.
- OPENAI_API_KEY: OpenAI API key.
- HF_TOKEN: Hugging Face token.
- PDF_DIR: Directory containing PDF documents.
- EVAL_DIR: Directory containing evaluation data.

Logging:
- Logs output to a file with the current timestamp and also streams to stdout.

Usage:
Run the script with the appropriate configuration file path to generate and evaluate 
data for the Tourist Guides Evaluation scenario.
"""
import os
import re
import datetime
import argparse
import logging
import sys
import random

from touarag.configs.configurator import Config
from touarag.common.utils import parse_docs, clear_memory, ExecutionTimer
from touarag.common.preprocess import preprocess_nodes
from touarag.llm.huggingface import load_hf_llm, load_hf_embed
from touarag.llm.openai import load_base_openai_llm, load_base_openai_embed
from touarag.evaluation.evaluator import Evaluator
from touarag.prompts.template import TourAgentPromptManager
from touarag.custom.graphrag import GraphRAGExtractor, GraphRAGStore, GraphRAGQueryEngine

from llama_index.core.node_parser import SentenceSplitter
from llama_index.extractors.entity import EntityExtractor
from llama_index.core import Settings
from llama_index.core import PropertyGraphIndex
import llama_index.core

import nltk

nltk.download("punkt_tab") # Used in transformations
SIM_TOP_K = 5

# Timestamp constant
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

DB_RESET = True # Resets the database
SAMPLING = True # Uses only small portion of data for testing purposes
KG_EXTRACT = True # Executes the full KG extraction process
MODE = "hf" # Mode for loading models

# Instantiate the timer
timer = ExecutionTimer()

ENT_PATTERN = r'\("entity"\$\$\$\$(.+?)\$\$\$\$(.+?)\$\$\$\$(.+?)\)'
REL_PATTERN = r'\("relationship"\$\$\$\$(.+?)\$\$\$\$(.+?)\$\$\$\$(.+?)\$\$\$\$(.+?)\)'

KG_TRIPLET_EXTRACT_TMPL = """
-Goal-
Given a text document, identify all entities and their entity types from the text and all relationships among the identified entities.
Given the text, extract up to {max_knowledge_triplets} entity-relation triplets.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: Type of the entity
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"$$$$<entity_name>$$$$<entity_type>$$$$<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relation: relationship between source_entity and target_entity
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other

Format each relationship as ("relationship"$$$$<source_entity>$$$$<target_entity>$$$$<relation>$$$$<relationship_description>)

3. When finished, output. Include ONLY thet parenthesis with the entity and relationship information.
Example Output:
("entity"$$$$Paris$$$$City$$$$Paris is the capital city in France, known for its rich history, vibrant culture, and significant landmarks. It serves as a major economic and cultural hub in the region, attracting tourists and residents alike.)

-Real Data-
######################
text: {text}
######################
output:"""


def parse_fn(response_str: str):
    """
    Parses the given response string to extract entities and relationships.

    Args:
        response_str (str): The response string to be parsed.

    Returns:
        tuple: A tuple containing two lists:
            - entities (list): A list of entities found in the response string.
            - relationships (list): A list of relationships found in the response string.
    """
    entities = re.findall(ENT_PATTERN, response_str)
    relationships = re.findall(REL_PATTERN, response_str)

    print("Entities found:", entities)
    print("Relationships found:", relationships)

    return entities, relationships


# Setup logging
logging.basicConfig(
    filename=f'/home/marios/projects/RAG-App-DWS/logs/output_graph_rag_{TIMESTAMP}.log',
    level=logging.DEBUG
)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
llama_index.core.set_global_handler("simple")

# Setup parser
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
DB_NAME = config.get_section(section="test_graph_db")["DB_NAME"]
DB_USER = config.get_section(section="test_graph_db")["USER"]
DB_PASS = config.get_section(section="test_graph_db")["PASSWORD"]
DB_HOST = config.get_section(section="test_graph_db")["HOST"]
DB_PORT = config.get_section(section="test_graph_db")["PORT"]

# OpenAI
os.environ["OPENAI_API_KEY"] = config.get_section(section="api")["OPEN_AI_API_KEY_2"]
HF_TOKEN = config.get_section(section="api")["HF_TOKEN"]

# Data Directory
PDF_DIR = "/home/marios/projects/RAG-App-DWS/data/Tourist Guides Eval/"
EVAL_DIR = "/home/marios/projects/RAG-App-DWS/data/Tourist Guides Eval/testset_persona_100.csv"

@timer.measure
def graphrag():
    if MODE == "hf":
        # Load Hugging Face models
        llm = load_hf_llm(hf_token=HF_TOKEN)
        openai_llm = load_base_openai_llm()
        embed_model = load_hf_embed(hf_token=HF_TOKEN)
    elif MODE == "openai":
        # Load OpenAI models
        llm = load_base_openai_llm()
        openai_llm = load_base_openai_llm()
        embed_model = load_base_openai_embed()
    else:
        raise ValueError("Invalid mode specified. Choose either 'hf' or 'openai'.")

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

    # Define KG Extractor object
    kg_extractor = GraphRAGExtractor(
        llm=openai_llm,
        extract_prompt=KG_TRIPLET_EXTRACT_TMPL,
        max_paths_per_chunk=2,
        parse_fn=parse_fn,
    )

    # Define Graph Store object for storing the extracted knowledge graph
    graph_store = GraphRAGStore(
        username=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        url=f"bolt://{DB_HOST}:{DB_PORT}",
        reset=DB_RESET
    )

    # Prepare nodes for indexing and sample if needed
    pgi_nodes = random.sample(nodes, 50) if SAMPLING else nodes

    if KG_EXTRACT:
        # Extract knowledge graph when creating the index 
        index = PropertyGraphIndex(
            nodes=pgi_nodes,
            kg_extractors=[kg_extractor],
            property_graph_store=graph_store,
            embed_kg_nodes=True,
            show_progress=True,
        )
    else:
        # When not recreating the graph store run this
        index = PropertyGraphIndex(
            nodes=pgi_nodes,
            property_graph_store=graph_store,
            embed_kg_nodes=True,
            show_progress=True,
        )

    # Generate community summaries in graph_store object
    graph_store.community_summaries = graph_store.get_community_summaries()

    # Add personalized prompt template to the query engine
    prompt_mngr = TourAgentPromptManager("personalized_kg")

    kg_tmplt = prompt_mngr.get_prompt_text()

    query_engine = GraphRAGQueryEngine(
        graph_store=index.property_graph_store,
        llm=llm,
        index=index,
        similarity_top_k=SIM_TOP_K,
        prompt=kg_tmplt
    )

    # Create an evaluator
    evaluator = Evaluator(eval_dir=EVAL_DIR,
                        query_engine=query_engine,
                        sampling=SAMPLING, # Testing purposes
                        scenario_label="graph_rag"
    )
    evaluator.generate_samples(async_enabled=True)
    evaluator.save_samples(output_dir="/home/marios/projects/RAG-App-DWS/data/Tourist Guides Eval")
    evaluator.evaluate(output_dir="/home/marios/projects/RAG-App-DWS/data/Tourist Guides Eval")

graphrag()

timer.save_to_file(f'/home/marios/projects/RAG-App-DWS/logs/exectime_graphrag_{TIMESTAMP}.log')
# Clear memory
clear_memory()
