import os
import argparse
from pathlib import Path
from functools import partial
from graphrag.cli.query import run_global_search, run_drift_search, run_local_search
from touarag.custom.graphrag_ms import IndexBuilder
from touarag.llm.openai import load_openai_llm, load_openai_embed
from touarag.evaluation.evaluator import Evaluator
from touarag.configs.configurator import Config

from ragas.metrics import FactualCorrectness, SemanticSimilarity, ResponseRelevancy

BUILD_INDEX = False
SAMPLING = True

CONFIG_PATH = Path(os.path.dirname(__file__)) / "settings.yaml"
ROOT_PATH = Path(os.path.dirname(__file__))
EVAL_DIR = "/home/marios/projects/RAG-App-DWS/data/Tourist Guides Eval/testset_persona_100.csv"

parser = argparse.ArgumentParser(description="")
parser.add_argument("--config_path",
                    type=str,
                    default="/home/marios/projects/RAG-App-DWS/src/scenarios/config.yaml",
                    help="Path to the configuration file")

args = parser.parse_args()

# Setup configuration
config = Config(args.config_path)

# OpenAI
os.environ["OPENAI_API_KEY"] = config.get_section(section="api")["OPEN_AI_API_KEY_2"]

llm_judge = load_openai_llm()
embed_judge = load_openai_embed()

# Define the function as a variable with some arguments
global_func = partial(run_global_search,
    config_filepath=CONFIG_PATH,
    data_dir=None,  # Retrieved automatically from root_dir
    root_dir=ROOT_PATH,
    community_level=2,  # The community level in the Leiden community hierarchy from which to load community reports. Higher values represent reports from smaller communities.
    dynamic_community_selection=False,
    response_type="Multiple Paragraphs",
    streaming=False
)

metrics = [
    FactualCorrectness(llm=llm_judge),
    SemanticSimilarity(embeddings=embed_judge),
    ResponseRelevancy(llm=llm_judge, embeddings=embed_judge)
]

def graphrag_ms_global():
    if BUILD_INDEX:
        # Build index
        runner = IndexBuilder(config_path=os.path.dirname(__file__))
        runner.run()
    
    # Create an evaluator
    evaluator = Evaluator(eval_dir=EVAL_DIR,
                        query_func=global_func,
                        sampling=SAMPLING, # Testing purposes
                        scenario_label="graph_ms_global"
    )
    evaluator.generate_samples_with_func()
    evaluator.save_samples(output_dir="/home/marios/projects/RAG-App-DWS/data/Tourist Guides Eval")
    evaluator.evaluate(output_dir="/home/marios/projects/RAG-App-DWS/data/Tourist Guides Eval")

graphrag_ms_global()