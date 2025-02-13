"""
This module provides functionality for preprocessing documents using a pipeline of transformations
and a vector store. It includes an asynchronous function to preprocess a list of documents by 
applying specified transformations and storing the results in a vector store.

Functions:
    preprocess_nodes(docs, vector_store, transformations=None, embed_model=None):
        Asynchronously preprocesses a list of documents using a specified set of transformations 
        and a vector store, and returns the processed nodes.
"""

from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter

def preprocess_nodes(docs, vector_store, transformations=None, embed_model=None):

    """
    Preprocesses a list of documents using a specified set of transformations and a vector store.

    Args:
        docs (list): A list of documents to be processed.
        transformations (list, optional): A list of transformation functions to be applied to the 
                                          documents. Defaults to None.
        embed_model (object, optional): An embedding model to be added to the transformations. 
                                        Defaults to None.
        vector_store (object): An instance of a vector store to be used in the pipeline.
    Returns:
        list: A list of processed nodes resulting from the pipeline execution.

    """

    # If no transformations are provided, use default transformations
    if transformations is None:
        transformations = [
            # Split sentences into chunks of size 400 with an overlap of 50
            SentenceSplitter(chunk_size=400, chunk_overlap=50),
            # Add the embedding model to the transformations
            embed_model
        ]

    # Create the ingestion pipeline with the provided transformations and vector store
    pipeline = IngestionPipeline(
        transformations=transformations,
        vector_store=vector_store
    )

    # Persist the pipeline configuration to disk
    pipeline.persist("./pipeline_storage")

    # Execute the pipeline asynchronously on the provided documents
    nodes = pipeline.run(documents=docs)
    return nodes
