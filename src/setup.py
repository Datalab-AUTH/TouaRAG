"""
This module sets up the package configuration for the 'touarag' project using setuptools.

Attributes:
    name (str): The name of the package.
    version (str): The version of the package.
    packages (list): A list of all Python import packages that should be 
                     included in the distribution package.
    install_requires (list): A list of dependencies required for the project.
    entry_points (dict): A dictionary of entry points for command line scripts.
    author (str): The name of the author of the package.
    author_email (str): The email address of the author.
    description (str): A brief description of the project.
    long_description_content_type (str): The format of the long description (e.g., 'text/markdown').
    url (str): The URL for the project's homepage.
    classifiers (list): A list of classifiers that provide some additional 
                        metadata about the package.
    python_requires (str): The Python version requirement for the package.
"""

from setuptools import setup, find_packages

setup(
    name='touarag',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # Add your project's dependencies here
        # e.g., 'numpy', 'pandas', 'requests'
    ],
    entry_points={
        'console_scripts': [
            # Add command line scripts here
            # e.g., 'mycommand=mypackage.module:function'
        ],
    },
    author='Datalab@AUTH',
    author_email='',
    description='Enhancing Personalized Travel Assistance with Retrieval-Augmented Generation: A Context-Aware Approach',
    long_description_content_type='text/markdown',
    url='https://github.com/Datalab-AUTH/TouaRAG',
    classifiers=[
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Ubuntu 24.04',
    ],
    python_requires='>=3.11',
)