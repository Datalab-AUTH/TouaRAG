"""
This module provides functionality to generate a test dataset using a specified language 
model and knowledge graph.

Functions:
    generate_testset(llm, knowledge_graph, samples=60, run_config=None):
"""
from ragas.testset import TestsetGenerator
def generate_testset(llm, embed, knowledge_graph, samples=60, run_config=None, persona_list=None):
    """
    Generates a test dataset using the provided language model and knowledge graph.
    Args:
        llm: The language model to be used for generating the test set.
        embed: The embedding model to be used for generating the test set.
        knowledge_graph: The knowledge graph to be used for generating the test set.
        samples (int, optional): The number of samples to generate. Defaults to 60.
        run_config (dict, optional): Configuration settings for the test set generation process.
    Returns:
        pandas.DataFrame: A DataFrame containing the generated test set.
    """

    generator = TestsetGenerator(llm=llm, embedding_model=embed, knowledge_graph=knowledge_graph, persona_list=persona_list)
    dataset_obj = generator.generate(samples, run_config=run_config)

    test_df = dataset_obj.to_pandas()

    return test_df
