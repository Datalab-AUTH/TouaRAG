"""
This module generates data for the Tourist Guides Evaluation scenario.

It uses the following components:
- Configuration file to set up the environment and API keys.
- Knowledge graph file to load the graph data.
- OpenAI models for language and embedding tasks.
- Test set generation based on the provided samples.

The generated test set is saved as a CSV file with a timestamp.

Example usage:
    python generate_data.py --config_path /path/to/config.yaml \
                            --graph_dir /path/to/knowledge_graph.pkl \
                            --output_csv /path/to/output.csv \
                            --samples 10
"""
import os
import datetime
import argparse
from touarag.configs.configurator import Config
from touarag.datageneration import graph_generator, testset_generator
from touarag.common.utils import clear_memory
from touarag.llm.openai import load_openai_llm, load_openai_embed
from ragas.testset.persona import Persona

# Timestamp constant
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

parser = argparse.ArgumentParser(description="Generate data for \
                                 the Tourist Guides Evaluation scenario")
parser.add_argument("--config_path",
                    type=str,
                    default="/home/marios/projects/RAG-App-DWS/src/scenarios/config.yaml",
                    help="Path to the configuration file")
parser.add_argument("--graph_dir",
                    type=str,
                    default="/home/marios/projects/RAG-App-DWS/data/Tourist Guides Eval/knowledge_graph_thess_2024-12-06_23-44-52.pkl",
                    help="Path to the knowledge graph file")
parser.add_argument("--output_csv",
                    type=str,
                    default=f"/home/marios/projects/RAG-App-DWS/data/Tourist Guides Eval/testset{TIMESTAMP}.csv",
                    help="Path to the output CSV file")
parser.add_argument("--samples",
                    type=int,
                    default=100,
                    help="Number of samples to generate")

args = parser.parse_args()

# Setup configuration
config = Config(args.config_path)
os.environ["OPENAI_API_KEY"] = config.get_section(section="api")["OPEN_AI_API_KEY_2"]
GRAPH_DIR = args.graph_dir

llm_model = load_openai_llm()
emeb_model = load_openai_embed()

# Add Persona for the scenario
persona_traveler = Persona(
    name="Traveler",
    role_description="Don't know much about what places to visit, where they are and their history. Wants to learn more information to improve the trip with as many details as possible."
)

personas = [persona_traveler]

# Load graph
graph = graph_generator.load_graph(GRAPH_DIR)
test_set = testset_generator.generate_testset(llm_model, emeb_model, graph, samples=args.samples, persona_list=personas)
test_set.to_csv(args.output_csv)

clear_memory()
