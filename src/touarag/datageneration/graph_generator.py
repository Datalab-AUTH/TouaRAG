"""

This module provides functionality for lgenerating 
knowledge graphs from document batches.

Functions:
    load_openai_llm(llm_name=None):
    dump_graph(knowledge_graph, path): Dumps a knowledge graph to a file.
    load_graph(path): Loads a knowledge graph from a file.
    generate_graph(documents_batch, transforms=None):
"""
import os
import pickle
from ragas.testset.transforms import default_transforms
from ragas.testset.transforms.engine import apply_transforms
from ragas.testset.graph import KnowledgeGraph, Node, NodeType



def generate_graph(documents_batch, llm_model, embed_model, transforms=None):
    """
    Processes a batch of documents to generate a knowledge graph.

    Args:
        documents_batch (list): A list of document objects, where each document 
                                has `page_content` and `metadata` attributes.
        llm_model: The language model to use for generating the knowledge graph.
        embed_model: The embedding model to use for generating the knowledge graph.
        transforms (list, optional): A list of transformation functions to apply to the knowledge 
                                     graph. If None, default transformations will be applied.

    Returns:
        KnowledgeGraph: A knowledge graph generated from the batch of documents with applied 
                        transformations.
    """
    nodes = []
    for doc in documents_batch:
        node = Node(
            type=NodeType.DOCUMENT,
            properties={
                "page_content": doc.page_content,
                "document_metadata": doc.metadata,
            },
        )
        nodes.append(node)

    kg = KnowledgeGraph(nodes=nodes)

    if transforms is None:
        transforms = default_transforms(documents=documents_batch, llm=llm_model, embedding_model=embed_model)

    apply_transforms(kg, transforms)
    return kg


def dump_graph(knowledge_graph, path):
    """
    Loads a knowledge graph from a file.

    Returns:
        KnowledgeGraph: A knowledge graph loaded from the specified file.
    """
    filehandler = open(path, 'wb')
    pickle.dump(knowledge_graph, filehandler)
    filehandler.close()


def load_graph(path):
    """
    Loads a knowledge graph from a file.

    Returns:
        KnowledgeGraph: A knowledge graph loaded from the specified file.
    """
    filehandler = open(path, 'rb')
    knowledge_graph = pickle.load(filehandler)
    filehandler.close()
    return knowledge_graph
