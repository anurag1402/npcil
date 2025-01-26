import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader

def load_documents(directory):
    """
    Load HR documents from a specified directory
    """
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(directory, filename)
            pdf_reader = PdfReader(pdf_path)
            
            # Extract text from all pages
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            documents.append(text)
    
    return documents

def create_vector_store(documents):
    """
    Create FAISS vector store from documents
    """
    # Split documents into chunks
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(documents)
    
    # Create embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create and save vector store
    vector_store = FAISS.from_texts(chunks, embedding_model)
    vector_store.save_local("hr_document_index")
    
    return vector_store

def main():
    # Load documents from HR_DOCUMENTS directory
    documents = load_documents("HR_DOCUMENTS")
    vector_store = create_vector_store(documents)
    print("Vector store created successfully!")

if __name__ == "__main__":
    main()
