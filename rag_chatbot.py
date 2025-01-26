from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

class NPCILHRBot:
    def __init__(self, vector_store):
        """
        Initialize HR Bot with vector store
        """
        # Load embeddings
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Store vector store
        self.vector_store = vector_store
        
        # Initialize language model
        self.generator = pipeline(
            "text-generation", 
            model="bigscience/bloomz-560m",
            max_length=300
        )
        
        # Configure retriever
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
    
    def generate_response(self, query):
        """
        Generate response using RAG approach
        """
        # Retrieve relevant documents
        docs = self.retriever.get_relevant_documents(query)
        context = " ".join([doc.page_content for doc in docs])
        
        # Generate response
        prompt = f"Context: {context}\nQuestion: {query}\nComprehensive Answer:"
        result = self.generator(prompt)
        
        return result[0]['generated_text']
