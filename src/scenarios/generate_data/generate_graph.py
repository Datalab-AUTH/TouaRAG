
import os
import datetime
import argparse
import sys
import logging
from touarag.configs.configurator import Config
from touarag.datageneration import graph_generator
from touarag.common.utils import clear_memory
from touarag.llm.openai import load_openai_llm, load_openai_embed
from touarag.common.utils import parse_docs, clear_memory

# Timestamp constant
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Setup logging
logging.basicConfig(
    filename=f'/home/marios/projects/RAG-App-DWS/logs/output_generate_graph_obj_{TIMESTAMP}.log',
    level=logging.DEBUG
)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

parser = argparse.ArgumentParser(description="Generate data for \
                                 the Tourist Guides Evaluation scenario")
parser.add_argument("--config_path",
                    type=str,
                    default="/home/marios/projects/RAG-App-DWS/src/scenarios/config.yaml",
                    help="Path to the configuration file")
parser.add_argument("--graph_dir",
                    type=str,
                    default=f"/home/marios/projects/RAG-App-DWS/data/Tourist Guides Eval/knowledge_graph_thess_{TIMESTAMP}.pkl",
                    help="Path to the knowledge graph file")

args = parser.parse_args()

# Setup configuration
config = Config(args.config_path)
os.environ["OPENAI_API_KEY"] = config.get_section(section="api")["OPEN_AI_API_KEY_2"]
GRAPH_DIR = args.graph_dir
# Data Directory
PDF_DIR = "/home/marios/projects/RAG-App-DWS/data/Tourist Guides Eval/"
EVAL_DIR = "/home/marios/projects/RAG-App-DWS/data/Tourist Guides Eval/testset_concat.csv"

llm_model = load_openai_llm()
embed_model = load_openai_embed()

# Parse documents
docs = parse_docs(PDF_DIR, module="langchain")

# Load graph
kg = graph_generator.generate_graph(docs, llm_model, embed_model, transforms=None)

graph_generator.dump_graph(kg, GRAPH_DIR)

clear_memory()
