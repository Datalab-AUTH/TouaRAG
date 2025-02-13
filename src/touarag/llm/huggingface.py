"""
This module provides functionality for loading and configuring Hugging Face language models (LLMs) 
with optional quantization and device mapping. It includes utilities for logging into the Hugging 
Face API using credentials from a configuration file and loading specific LLMs and embedding models.

Constants:
    LLAMA3_1_8B_INSTRUCT (str): Identifier for the Meta-Llama-3.1-8B-Instruct model.
    NEMOTRON_MINI_4B_INSTRUCT (str): Identifier for the Nvidia-Nemotron-Mini-4B-Instruct model.
    LLAMA3_2_3B_INSTRUCT (str): Identifier for the Meta-Llama-3.2-3B-Instruct model.
    UNSLOTH_3_2_3B_INSTRUCT_4BIT (str): Identifier for the Unsloth-Llama-3.2-3B-Instruct-bnb-4bit 
                                        model.
    NOMIC_EMBED_1_5 (str): Identifier for the Nomic-Embed-Text-v1.5 embedding model.
    BGE_M3 (str): Identifier for the BAAI-bge-m3 embedding model.
    DEFAULT_MODEL (str): Default language model identifier.
    DEFAULT_EMBED (str): Default embedding model identifier.
    DEFAULT_QUAN_CONFIG (BitsAndBytesConfig): Default configuration for model quantization.

Functions:
    hf_login_wrap():

    load_hf_llm(llm_name=None, quantization_config=DEFAULT_QUAN_CONFIG, device_map="auto"):
"""
import os
from transformers import AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# LLM Models
LLAMA3_1_8B_INSTRUCT = "meta-llama/Meta-Llama-3.1-8B-Instruct"
NEMOTRON_MINI_4B_INSTRUCT = "nvidia/Nemotron-Mini-4B-Instruct"
LLAMA3_2_3B_INSTRUCT = "meta-llama/Llama-3.2-3B-Instruct"
UNSLOTH_3_2_3B_INSTRUCT_4BIT= "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"

# Embedding Models
NOMIC_EMBED_1_5 = "nomic-ai/nomic-embed-text-v1.5"
BGE_M3 = "BAAI/bge-m3"

# Default Selections
DEFAULT_MODEL = LLAMA3_2_3B_INSTRUCT
DEFAULT_EMBED = BGE_M3


DEFAULT_QUAN_CONFIG = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )

def hf_login_wrap(hf_token):
    """    
    Logs into the Hugging Face API using the provided token.

    This function sets the "HF" environment variable with the provided API token 
    and then logs into the Hugging Face API using this token.

    Args:
        hf_token (str): The API token for Hugging Face.

    Returns:
        str: The API token that was set in the environment variable.

        KeyError: If the "HF" key is not found in the environment variables.
    """

    os.environ["HF"] = hf_token
    login(os.environ["HF"])
    return os.environ["HF"]

def load_hf_llm(llm_name=None,
                quantization_config=DEFAULT_QUAN_CONFIG,
                device_map="auto",
                hf_token=None):
    """
    Load a Hugging Face language model (LLM) with optional quantization and device mapping.

    Args:
        llm_name (str, optional): The name of the language model to load. 
                                  If None, a default model is used.
        quantization_config (dict, optional): Configuration for model quantization. 
                                              Defaults to DEFAULT_QUAN_CONFIG.
        device_map (str or dict, optional): Device mapping for model deployment. Defaults to "auto".
        hf_token (str): The API token for Hugging Face.

    Returns:
        HuggingFaceLLM: An instance of the HuggingFaceLLM class with the specified configuration.
    """

    # Login with Hugging Face API
    hf_token = hf_login_wrap(hf_token)

    if llm_name is None:
        llm_name = DEFAULT_MODEL

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm_name)

    llm_model = HuggingFaceLLM(model_name=llm_name,
                               tokenizer=tokenizer,
                               max_new_tokens=700,
                               model_kwargs={"quantization_config": quantization_config,
                                             "token": hf_token},
                               device_map=device_map
    )

    return llm_model

def load_hf_embed(embed_name=None, device="cuda", hf_token=None):
    """
    Loads a HuggingFace embedding model.

    This function logs into HuggingFace, loads the specified embedding model, and returns it.
    If no embedding model name is provided, a default model is used.

    Args:
        embed_name (str, optional): The name of the embedding model to load. Defaults to None.
        device (str, optional): The device to load the model on (e.g., "cuda" or "cpu"). 
                                Defaults to "cuda".
        hf_token (str): The API token for Hugging Face.

    Returns:
        HuggingFaceEmbedding: The loaded HuggingFace embedding model.
    """
    # Login with Hugging Face API
    hf_token = hf_login_wrap(hf_token)

    if embed_name is None:
        embed_name = DEFAULT_EMBED

    # Load the embedding model
    embed_model = HuggingFaceEmbedding(model_name=embed_name,
                                       trust_remote_code=True,
                                       device=device,
                                       token=hf_token)

    return embed_model
