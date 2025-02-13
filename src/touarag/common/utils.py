"""This module provides utility general functions.

Functions:
    parse_docs(directory): 
"""
import os
import gc
import time
from functools import wraps
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import PyMuPDFReader
from langchain.document_loaders import PyMuPDFLoader
import torch
import nest_asyncio
nest_asyncio.apply()


def parse_docs(directory, module=None):
    """
    Parses all PDF documents in the specified directory.

    This function searches the given directory for all files with a '.pdf' extension,
    loads them using the PyMuPDFLoader from Llamaindex or LlamaParse, and returns a list 
    of the loaded PDF documents.

    Args:
        directory (str): The path to the directory containing PDF files.

    Returns:
        list: A list of loaded PDF documents.
    """

    # If no module defined use PyMuPDFReader by default

    # Get a list of all PDF files in the directory
    pdf_files = [os.path.join(directory, f)\
                for f in os.listdir(directory) if f.endswith('.pdf')]
    if module is None:
        # Load and process PDF documents
        documents = []
        for pdf_file in pdf_files:
            loader = PyMuPDFReader()
            documents.extend(loader.load(pdf_file))
    elif module == "llamaparse":
        # Set up parser
        parser = LlamaParse(
            result_type="markdown",  # "markdown" and "text" are available
            # api_key=LLAMA_CLOUD_API_KEY
        )

        file_extractor = {".pdf": parser}
        documents = []
        for pdf_file in pdf_files:
            documents.extend(
                SimpleDirectoryReader(input_files=[pdf_file],
                                      file_extractor=file_extractor).load_data())
    elif module == "langchain":
        # Load and process PDF documents
        documents = []
        for pdf_file in pdf_files:
            loader = PyMuPDFLoader(pdf_file)
            pdf_documents = loader.load()  # Load the PDFs using PyMuPDFLoader from LangChain
            documents.extend(pdf_documents)  # Add the loaded documents to the list

    return documents

def clear_memory():
    """
    Clears the CUDA memory cache and performs garbage collection.

    This function empties the CUDA memory cache using `torch.cuda.empty_cache()`
    and then triggers Python's garbage collector using `gc.collect()`. After
    clearing the memory, it prints a summary of the CUDA memory usage.

    Note:
        This function is useful for freeing up GPU memory in between different
        stages of a program or after certain operations to ensure efficient
        memory usage.

    Returns:
        None
    """
    torch.cuda.empty_cache()
    gc.collect()
    print(torch.cuda.memory_summary(device=None, abbreviated=False))

class ExecutionTimer:
    """    
    A class used to measure and store the execution times of functions.
    Methods
    -------
    measure(func)
    save_to_file(file_path)
    """
    def __init__(self):
        self.execution_times = {}  # Dictionary to store execution times

    def measure(self, func):
        """
        Decorator to measure the execution time of a function and store it.
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            # Store the execution time with function name as the key
            self.execution_times[func.__name__] = elapsed_time
            print(f"{func.__name__} executed in {elapsed_time:.4f} seconds")
            return result
        return wrapper

    def save_to_file(self, file_path):
        """
        Save the recorded execution times to a file.
        """
        with open(file_path, 'w', encoding='utf-8') as file:
            for func_name, exec_time in self.execution_times.items():
                file.write(f"{func_name}: {exec_time:.4f} seconds\n")
        print(f"Execution times saved to {file_path}")
