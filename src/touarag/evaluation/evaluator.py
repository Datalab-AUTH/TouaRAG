"""
This module provides the `Evaluator` class for evaluating the performance 
of a model using various metrics.

Classes:
    Evaluator: A class for generating samples, saving samples, and 
    evaluating the performance of a model.

Functions:
    generate_samples: Generates samples for evaluation by iterating over the evaluation DataFrame.
    save_samples: Saves the current samples to a pickle file in the specified output directory.
    evaluate: Evaluates the performance of the model using various metrics and saves the results.

Constants:
    COST_DICT: A dictionary containing the prices per 1 million tokens for different models.

Dependencies:
    datetime
    pickle
    json
    ast
    pandas as pd
    tqdm
    touarag.llm.openai (load_openai_llm, load_openai_embed)
    ragas.metrics (LLMContextRecall, FactualCorrectness, Faithfulness, SemanticSimilarity, 
                   LLMContextPrecisionWithoutReference, LLMContextPrecisionWithReference, 
                   NonLLMContextRecall, ContextEntityRecall, NoiseSensitivity, ResponseRelevancy)
    ragas (SingleTurnSample, EvaluationDataset)
    ragas.evaluation (evaluate)
    ragas.run_config (RunConfig)
    ragas.cost (get_token_usage_for_openai)
"""
from datetime import datetime
import pickle
import json
import ast
import pandas as pd
from tqdm import tqdm

from touarag.llm.openai import load_openai_llm, load_openai_embed

from ragas.metrics import LLMContextRecall, FactualCorrectness, Faithfulness, \
    SemanticSimilarity, LLMContextPrecisionWithoutReference, LLMContextPrecisionWithReference, \
    NonLLMContextRecall, ContextEntityRecall, NoiseSensitivity, ResponseRelevancy
from ragas import SingleTurnSample, EvaluationDataset
from ragas.evaluation import evaluate
from ragas.run_config import RunConfig
from ragas.cost import get_token_usage_for_openai

# Prices indicated per 1 million tokens
COST_DICT = {
    "gpt-4o-mini": {
        "input_tokens_price": 0.15,
        "output_tokens_price": 0.600,
    },
    "gpt-4o": {
        "input_tokens_price": 2.5,
        "output_tokens_price": 10,
    }
}

class Evaluator:
    """    
    Evaluator class for evaluating the performance of a model using various metrics.

    Attributes:
        eval_dir (str): Directory containing the evaluation data.
        eval_df (pd.DataFrame): DataFrame containing the evaluation data.
        query_engine (object): Query engine for generating responses.
        query_func (object): Query function for generating responses.
        llm_judge_loader (object): Loader for the default OpenAI LLM judge.
        embed_judge_loader (object): Loader for the default OpenAI embed judge.
        samples (list): List of generated samples for evaluation.

    Methods:
        generate_samples(): Generates samples for evaluation by iterating over 
                            the evaluation DataFrame.
        save_samples(output_dir): Saves the current samples to a pickle file in 
                                  the specified output directory.
        evaluate(output_dir): Evaluates the performance of the model using various 
                              metrics and saves the results.
    """
    def __init__(self, eval_dir, query_engine=None, query_func=None, llm_judge=None, \
                 embed_judge=None, sampling = False, scenario_label="undefined_scenario"):
        self.eval_dir = eval_dir
        self.eval_df = pd.read_csv(self.eval_dir, index_col=0).reset_index(drop=True)

        if sampling:
            self.eval_df = self.eval_df.sample(n=2, random_state=42)

        self.query_engine = query_engine
        self.query_func = query_func

        self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        if (llm_judge is None) or (embed_judge is None):
            print("Loading default OpenAI judges for evaluation")
            self.llm_judge_loader = load_openai_llm()
            self.embed_judge_loader = load_openai_embed()
        self.samples = []
        self.scenario_label = scenario_label

    def generate_samples(self, async_enabled=False):
        """
        Generates samples for evaluation by iterating over the evaluation DataFrame.

        For each row in the evaluation DataFrame, this method performs the following steps:
        1. Prints the user input question.
        2. Retrieves contexts using the retriever based on the user input.
        3. Prints the retrieved contexts.
        4. Queries the query engine to get a response for the user input.
        5. Prints the generated response.
        6. Creates a SingleTurnSample object with the user input, retrieved contexts, 
           reference contexts, generated response, and reference answer.
        7. Appends the created sample to the samples list.

        The progress of sample generation is displayed using a progress bar.

        Returns:
            None
        """
        for _, row in tqdm(self.eval_df.iterrows(), total=self.eval_df.shape[0], \
                           desc="Generating samples"):
            print(f"Generating answer for question : {row['user_input']}")

            if not async_enabled:
                response = self.query_engine.query(row['user_input'])
            else:
                response = self.query_engine.aquery(row['user_input'])

            # Extract from response object, the source nodes (context)
            retrieved_context = response.source_nodes[:]

            retrieved_texts = [context.text if hasattr(context, 'text') else context for context in retrieved_context]
            print(f"Retrieved contexts: {retrieved_texts}")

            print(f"Answer: {response.response}")

            sample = SingleTurnSample(
                user_input=row['user_input'],
                retrieved_contexts=retrieved_texts,
                reference_contexts=ast.literal_eval(row['reference_contexts']),
                response=response.response,
                reference=row['reference']
            )
            self.samples.append(sample)
    
    def generate_samples_with_func(self, eval_context=False):
        """
        Generates samples for evaluation by iterating over the evaluation DataFrame.

        For each row in the evaluation DataFrame, this method performs the following steps:
        1. Prints the user input question.
        2. Retrieves contexts using the retriever based on the user input.
        3. Prints the retrieved contexts.
        4. Queries the query engine to get a response for the user input.
        5. Prints the generated response.
        6. Creates a SingleTurnSample object with the user input, retrieved contexts, 
           reference contexts, generated response, and reference answer.
        7. Appends the created sample to the samples list.

        The progress of sample generation is displayed using a progress bar.

        Returns:
            None
        """
        for _, row in tqdm(self.eval_df.iterrows(), total=self.eval_df.shape[0], \
                           desc="Generating samples"):
            print(f"Generating answer for question : {row['user_input']}")

            self.query_func.keywords["query"] = row['user_input']

            # Execute query and retrieve response and retrieved context
            response, context_output = self.query_func()

            if eval_context:
                # Take only the reports section
                retrieved_context = context_output['sources']

                # Extract only the content of the reports
                retrieved_texts = [context['text'] for context in retrieved_context]
                print(f"Retrieved contexts: {retrieved_texts}")
                
                sample = SingleTurnSample(
                                user_input=row['user_input'],
                                retrieved_contexts=retrieved_texts,
                                reference_contexts=ast.literal_eval(row['reference_contexts']),
                                response=response,
                                reference=row['reference']
                        )
            else:
                sample = SingleTurnSample(
                                    user_input=row['user_input'],
                                    response=response,
                                    reference=row['reference']
                )
            print(f"Answer: {response}")
            self.samples.append(sample)

    def save_samples(self, output_dir):
        """
        Save the current samples to a pickle file in the specified output directory.

        Args:
            output_dir (str): The directory where the pickle file will be saved.

        The filename will be in the format 'samples_baseline_<timestamp>_<number_of_samples>.pkl'.
        """
        with open(f"{output_dir}/samples_{self.scenario_label}_{self.timestamp}_{len(self.samples)}.pkl",\
                   "wb") as f:
            pickle.dump(self.samples, f)

    def evaluate(self, output_dir, custom_metrics=None, run_config=None):
        """
        Evaluates the performance of the model using various metrics and saves the results.
        Args:
            output_dir (str): The directory where the evaluation results will be saved.
            custom_metrics (list, optional): A list of custom metrics to use for evaluation. 
                             If None, default metrics will be used.
            run_config (RunConfig, optional): The configuration for running the evaluation.
        Returns:
            None
        This method performs the following steps:
        1. Loads the necessary judges for evaluation.
        2. Defines a list of metrics to evaluate the model. The default metrics are:
            - LLMContextRecall
            - FactualCorrectness
            - Faithfulness
            - SemanticSimilarity
            - LLMContextPrecisionWithoutReference
            - LLMContextPrecisionWithReference
            - NonLLMContextRecall
            - ContextEntityRecall
            - NoiseSensitivity
            - ResponseRelevancy
        3. Creates an evaluation dataset from the provided samples.
        4. Runs the evaluation using the specified metrics and configuration.
        5. Prints the calculated token consumption and estimated cost.
        6. Converts the evaluation results to a pandas DataFrame and saves it as a CSV file.
        7. Saves the evaluation results as a JSON file.
        """
        llm_judge = self.llm_judge_loader
        embed_judge = self.embed_judge_loader

        if custom_metrics is None:
            metrics = [
                LLMContextRecall(llm=llm_judge),
                FactualCorrectness(llm=llm_judge),
                Faithfulness(llm=llm_judge),
                SemanticSimilarity(embeddings=embed_judge),
                LLMContextPrecisionWithoutReference(llm=llm_judge),
                LLMContextPrecisionWithReference(llm=llm_judge),
                NonLLMContextRecall(),
                ContextEntityRecall(llm=llm_judge),
                NoiseSensitivity(llm=llm_judge),
                ResponseRelevancy(llm=llm_judge, embeddings=embed_judge)
            ]
        else:
            metrics = custom_metrics

        if run_config is None:
            run_config = RunConfig(
                timeout=60,
                max_retries=10,
                max_workers=8
            )

        ragas_dataset = EvaluationDataset(samples=self.samples)
        results = evaluate(
            dataset=ragas_dataset,
            metrics=metrics,
            run_config=run_config,
            token_usage_parser=get_token_usage_for_openai
        )

        print(f"Calculated token consumption : {results.total_tokens()}")

        # Identify selected LLM
        model_name = self.llm_judge_loader.langchain_llm.model_name
        if model_name in COST_DICT:
            input_cost = COST_DICT[model_name]['input_tokens_price'] / 1e6
            output_cost = COST_DICT[model_name]['output_tokens_price'] / 1e6
            print(f"Estimated eavaluation cost : {round(results.total_cost(input_cost, output_cost), 2)}$")
        else:
            print(f"No cost reported for model: {model_name}")

        results_df = results.to_pandas()
        results_df.to_csv(f"{output_dir}/results_{self.scenario_label}_{self.timestamp}_{len(self.samples)}.csv")

        res_dict = results._repr_dict
        with open(f"{output_dir}/results_{self.scenario_label}_{self.timestamp}_{len(self.samples)}.json",
                  "w", 
                  encoding="utf-8") as json_file:
            json.dump(res_dict, json_file, indent=4)

# Usage example:
# evaluator = Evaluator(eval_dir=EVAL_DIR, bm25_retriever=bm25_retriever, query_engine=query_engine,
#                       llm_judge_loader=load_openai_llm, embed_judge_loader=load_openai_embed)
#
# evaluator.load_data()
# evaluator.generate_samples()
# evaluator.save_samples(output_dir="/home/marios/projects/RAG-App-DWS/data/Tourist Guides Eval")
# evaluator.evaluate(output_dir="/home/marios/projects/RAG-App-DWS/data/Tourist Guides Eval")
