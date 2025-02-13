<!-- # <img style="vertical-align:middle" height="200" src="./docs/_static/imgs/touarag_logo.png" alt="TouaRAG Logo">   -->
# TouaRAG
*Enhance Your Travel Experience with Personalized RAG Assistance 🌍✈️*

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg?color=purple)](https://www.python.org/)  
[![License](https://img.shields.io/github/license/Datalab-AUTH/TouaRAG.svg?color=green)](https://github.com/Datalab-AUTH/TouaRAG/blob/master/LICENSE)
---

## Empowering Personalized Travel Assistance with Retrieval-Augmented Generation

TouaRAG is your comprehensive framework for building and evaluating context-aware travel chatbots. Leverage our modular library **touarag**—packed with implementations for testing various RAG architectures and evaluating their performance—to deliver tailored travel recommendations. In addition, our solution offers an integrated API backend and UI frontend to effortlessly run a RAG chatbot with profile personalization, model selection, and architecture configuration.

> **Prerequisites:**  
> This framework is designed for Linux environments (due to bash script usage). Make sure to install the required libraries using `requirements.txt` before proceeding.

## Key Features

- **Personalized Travel Assistance:** Deliver customized travel itineraries based on user profiles.
- **RAG Evaluation:** Test and evaluate multiple retrieval-augmented generation architectures.
- **Integrated Chatbot:** Seamlessly run a travel assistant with both API backend and UI frontend.
- **Easy Deployment:** Manage installation and uninstallation with simple wrapper scripts.
- **Modular Design:** Swap out models, tweak architectures, and refine evaluation metrics effortlessly.

## Installation

### Prerequisites

- **Operating System:** Linux (for bash script compatibility)
- **Dependencies:** Install required libraries with:
  ```bash
  pip install -r requirements.txt

## Install the TouaRAG Library

```bash
./install.sh
```

## Uninstall the TouaRAG Library

```bash
./uninstall.sh
```

## Quickstart

### Launch Your Personalized Travel Chatbot

Get started quickly by running our integrated API backend and UI frontend with a single command:

```bash
./start_app.sh
```

This wrapper script launches the complete TouaRAG travel assistant application, allowing you to:

- Interact with a dynamic travel chatbot
- Personalize user profiles for tailored recommendations
- Choose from various RAG architectures and models

### Evaluate a RAG Architecture with TouaRAG

Below is a brief example showcasing how to evaluate a travel query dataset using the touarag library:

```python
query_engine = TransformQueryEngine(...)

    # Create an evaluator
    evaluator = Evaluator(eval_dir=EVAL_DIR,
                        query_engine=query_engine,
                        sampling=False, # Use for testing purposes
                        scenario_label="test_scenario_1"
    )
    evaluator.generate_samples()
    evaluator.save_samples(output_dir="...")
    evaluator.evaluate(output_dir="...")

```
