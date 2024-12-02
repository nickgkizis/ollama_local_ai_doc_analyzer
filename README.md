# Document Assistant App

This Document Assistant is an intelligent application built to process PDF documents, extract relevant sections, and provide insightful responses to user queries using advanced vector-based search and AI-powered language modeling.

## Table of Contents
- Features
- Prerequisites
- Installation
- How to Run

## Features
- Load and parse PDF documents.
- Extract specific sections such as Contact Information, Education, and Work Experience.
- Use vector-based storage for efficient document retrieval.
- Generate dynamic suggestions for similar questions.
- Offer contextual responses powered by Llama 3.2.
- Interactive and user-friendly interface built with Streamlit.

## Prerequisites
Before running the app, ensure you have the following installed:

- **Python 3.12.4** Download and install from the official [Python website](https://www.python.org/).
- **Llama 3.2** installed through the [Ollama website](https://ollama.com/library/llama3.2).

Installation details are provided in the steps below.

## Installation
Follow these steps to set up and run the app:

1. **Clone the Repository**
    ```bash
    git clone (https://github.com/nickgkizis/ollama_local_ai_doc_analyzer.git)
    cd ollama_local_ai_doc_analyzer
    ```

2. **Set Up a Virtual Environment**
    Create and activate a virtual environment:
    ### On Windows
    ```
    python -m venv venv
    venv\Scripts\activate
    ```
    ### On macOS/Linux
    ```
    python -m venv venv
    source venv/bin/activate
    ```

4. **Install Dependencies**
    Ensure you have a `requirements.txt` file in the project directory, then run:
    ```
    pip install -r requirements.txt
    ```

5. **Install Llama 3.2**
    Install Llama 3.2 using the ollama package manager:
    ```
    ollama pull llama3.2
    ```

## Run the Streamlit app:
```
streamlit run pdf-rag.py
```


#Open the link provided in your terminal (e.g., http://localhost:8501) to access the app in your web browser.
