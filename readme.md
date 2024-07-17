# Hybrid Retrieval and Question-Answering System Using MILVUS vector datastore and LLMs

## Overview

This project implements a hybrid retrieval and question-answering system that combines traditional text-based retrieval (BM25) and modern embedding-based retrieval (DPR). The system expands user queries, retrieves relevant documents from a Milvus vector database, and generates answers using a pre-trained language model. A Streamlit interface allows users to interact with the system by entering queries and viewing the results in a user-friendly manner.

## Setup Instructions

### Prerequisites

To run the code successfully in the system ensure the following requirements:

- Python 3.7+
- Pip (Python package installer)
- Docker
- Milvus (Vector database)

### Dependencies

- Create the virtual environment by following command:

    `python3 -m venv env_name`

- Activate the virtaul environment and then run the below command to install the dependencies

    `pip install -r requirements.txt`

### Milvus Setup

1. Follow the detailed instructions from [Milvus installation guide](https://milvus.io/docs/v2.0.x/install_standalone-docker.md) and [How to Get Started with Milvus](https://milvus.io/blog/how-to-get-started-with-milvus.md) to set up Milvus.
- To run Milvus succesfully in the system ensure Docker is properly running in the system. 
- To run Docker in windows, install _Docker Desktop_  and ensure that _Virtualization_ is enabled in the system. If it is not enabled in the system, then enable it from the BIOS settings.
2. Ensure Milvus is running on `localhost` and the default port `19530`.


## Running the System

To run the entire system, I have created streamlit UI.

Run the application using below command:
`streamlit run app.py`