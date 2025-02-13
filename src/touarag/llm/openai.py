"""This module provides functionality to load and initialize an OpenAI language model
using the Langchain framework. It includes a function to load the model with a 
specified or default name and returns an instance of LangchainLLMWrapper.

Functions:
    load_openai_llm(llm_name=None):
        Loads an OpenAI language model using the specified model name and returns
        an instance of LangchainLLMWrapper.

Dependencies:
    - langchain_openai.ChatOpenAI
    - ragas.llms.LangchainLLMWrapper
    - ..common.utils.load_config
"""
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

def load_openai_llm(llm_name=None):
    """
    Loads an OpenAI language model using the specified model name.

    Args:
        llm_name (str, optional): The name of the language model to load. 
                                  Defaults to "gpt-4o-mini" if not provided.

    Returns:
        LangchainLLMWrapper: An instance of LangchainLLMWrapper initialized with 
                             the specified or default model.
    """
    if llm_name is None:
        llm_name = "gpt-4o-mini"
    ragas_openai_llm = LangchainLLMWrapper(ChatOpenAI(model=llm_name))

    return ragas_openai_llm

def load_openai_embed():
    """
    Loads an OpenAI language model for embedding generation.

    Returns:
        LangchainLLMWrapper: An instance of LangchainLLMWrapper initialized with 
                             the "text-embedding-3-small" model.
    """
    embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    # With the `text-embedding-3` class
    # of models, you can specify the size
    # of the embeddings you want returned.
    # dimensions=1024
    )
    ragas_embed = LangchainEmbeddingsWrapper(embeddings)
    return ragas_embed

def load_base_openai_llm(llm_name=None):
    """
    Loads an OpenAI language model using the specified model name.

    Args:
        llm_name (str, optional): The name of the language model to load. 
                                  Defaults to "gpt-4o-mini" if not provided.

    Returns:
        ChatOpenAI: An instance of ChatOpenAI initialized with 
                    the specified or default model.
    """
    if llm_name is None:
        llm_name = "gpt-4o-mini"
    openai_llm = OpenAI(model=llm_name, max_retries=15, reuse_client=False)

    return openai_llm

def load_base_openai_embed():
    """
    Load the base OpenAI embeddings model.
    This function initializes and returns an instance of the OpenAIEmbeddings
    class with the specified model and maximum number of retries.
    Returns:
        OpenAIEmbeddings: An instance of the OpenAIEmbeddings class initialized
        with the 'text-embedding-3-small' model and a maximum of 15 retries.
    """
    model="text-embedding-3-small"
    openai_embed = OpenAIEmbedding(model=model, max_retries=15, reuse_client=False)
    return openai_embed
