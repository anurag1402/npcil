from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline

class NPCILHRBot:
    def __init__(self, index_path="hr_document_index"):
        """
        Initialize HR Bot with pre-embedded documents
        """
        # Load embeddings
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Load vector store
        self.vector_store = FAISS.load_local(index_path, self.embedding_model)
        
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

# Example usage
def main():
    hr_bot = NPCILHRBot()
    while True:
        query = input("Ask a question about HR policies (or 'quit'): ")
        if query.lower() == 'quit':
            break
        response = hr_bot.generate_response(query)
        print("Bot Response:", response)

if __name__ == "__main__":
    main()
