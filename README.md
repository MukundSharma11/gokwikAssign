# How to run
## Download the repo -> python3.11 -m venv myenv -> source myenv/bin/activate -> pip install -r req.txt -> streamlit run streamlit_app.py

-----------------------------------------------------------------------------------------------------------------------------------------------------------

# Gokwik- Project Option 1: AI Agent for Document-Based Q&A
## Problem
Our company has hundreds of internal Word documents containing important product,
business, and tech information. We want an AI agent that can answer user queries solely
using these documents, and not hallucinate beyond them.

Your Task
Build a working prototype that:
● Takes a question as input
● Searches through the corpus of documents based on question
● Returns a grounded, relevant answer using only the content in those documents
● Avoids answering when relevant information is missing
● Show sources (file name + excerpt) for any answer
● Use streamlit as UI


## Indexing 
Done using pypdfloader, RecursiveCharacterTextSplitter and Faiss vector db

## Retrieval
[Read Retrieval.txt](./Retrieval.txt)

## Augmentation
Formed a system prompt for the llm

## Generation
Done using Gemma2-9b-it model by providing the context docs and the query


