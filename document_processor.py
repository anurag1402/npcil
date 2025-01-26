import os
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader

@st.cache_resource
def load_documents(directory='HR_DOCUMENTS'):
    """
    Load HR documents from specified directory
    """
    documents = []
    
    # Handle both local and Streamlit Cloud scenarios
    if os.path.exists(directory):
        file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pdf')]
    else:
        file_paths = [f for f in os.listdir() if f.endswith('.pdf')]
    
    for file_path in file_paths:
        try:
            pdf_reader = PdfReader(file_path)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            documents.append(text)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return documents

@st.cache_resource
def create_vector_store(documents):
    """
    Create FAISS vector store from documents
    """
    # Split documents into chunks
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text("\n".join(documents))
    
    # Create embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create vector store
    vector_store = FAISS.from_texts(chunks, embedding_model)
    
    return vector_store

def initialize_vector_store():
    """
    Load documents and create vector store
    """
    documents = load_documents()
    
    if not documents:
        raise ValueError("No documents found. Please upload PDF files.")
    
    return create_vector_store(documents)
