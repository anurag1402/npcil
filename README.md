# NPCIL HR Bot

## Overview
NPCIL HR Bot is an AI-powered chatbot for retrieving and understanding HR policies using advanced RAG (Retrieval-Augmented Generation) techniques.

## Setup Instructions
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Place HR documents in `HR_DOCUMENTS` directory
4. Run document indexing: `python document_processor.py`
5. Launch web app: `streamlit run app.py`

## Features
- Open-source RAG implementation
- PDF document processing
- Semantic search capabilities
- User-friendly web interface

## Technologies
- LangChain
- HuggingFace Transformers
- FAISS Vector Store
- Streamlit
