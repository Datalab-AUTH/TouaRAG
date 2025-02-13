import os
import argparse
import datetime
from pathlib import Path
from functools import partial
from graphrag.cli.query import run_drift_search
from touarag.custom.graphrag_ms import IndexBuilder
from touarag.llm.openai import load_openai_llm, load_openai_embed
from touarag.evaluation.evaluator import Evaluator
from touarag.configs.configurator import Config
from touarag.common.utils import ExecutionTimer

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

# Instantiate the timer
timer = ExecutionTimer()

# Timestamp constant
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# OpenAI
os.environ["OPENAI_API_KEY"] = config.get_section(section="api")["OPEN_AI_API_KEY_2"]

llm_judge = load_openai_llm()
embed_judge = load_openai_embed()

# Define the function as a variable with some arguments
drift_func = partial(run_drift_search,
    config_filepath=CONFIG_PATH,
    data_dir=None,  # Retrieved automatically from root_dir
    root_dir=ROOT_PATH,
    community_level=2,  # The community level in the Leiden community hierarchy from which to load community reports. Higher values represent reports from smaller communities.
    streaming=False
)

metrics = [
    FactualCorrectness(llm=llm_judge),
    SemanticSimilarity(embeddings=embed_judge),
    ResponseRelevancy(llm=llm_judge, embeddings=embed_judge)
]

@timer.measure
def graphrag_ms_drift():
    if BUILD_INDEX:
        # Build index
        runner = IndexBuilder(config_path=os.path.dirname(__file__))
        runner.run()
    
    # Create an evaluator
    evaluator = Evaluator(eval_dir=EVAL_DIR,
                        query_func=drift_func,
                        sampling=SAMPLING, # Testing purposes
                        scenario_label="graph_ms_drift"
    )
    evaluator.generate_samples_with_func(eval_context=True)
    evaluator.save_samples(output_dir="/home/marios/projects/RAG-App-DWS/data/Tourist Guides Eval")
    evaluator.evaluate(output_dir="/home/marios/projects/RAG-App-DWS/data/Tourist Guides Eval",
                       #custom_metrics=metrics
                       )

graphrag_ms_drift()
timer.save_to_file(f'/home/marios/projects/RAG-App-DWS/logs/exectime_graphrag_ms_drift_{TIMESTAMP}.log')
