import streamlit as st
from rag_chatbot import NPCILHRBot
from document_processor import initialize_vector_store

def main():
    st.set_page_config(page_title="NPCIL HR Bot", page_icon="🤖")
    
    st.title("NPCIL HR Bot 🤖")
    st.subheader("Intelligent HR Policy Assistant")
    
    # Initialize vector store
    try:
        vector_store = initialize_vector_store()
    except ValueError as e:
        st.error(str(e))
        st.info("Please upload PDF files in the HR_DOCUMENTS directory or root folder.")
        return
    
    # Initialize chatbot
    hr_bot = NPCILHRBot(vector_store)
    
    # Chat input
    user_query = st.text_input("Ask a question about HR policies:")
    
    if user_query:
        try:
            # Generate response
            response = hr_bot.generate_response(user_query)
            
            # Display response
            st.write("Bot's Response:")
            st.info(response)
        except Exception as e:
            st.error(f"Error generating response: {e}")
    
    # Sidebar information
    st.sidebar.title("About NPCIL HR Bot")
    st.sidebar.info(
        "An AI-powered assistant to help you navigate HR policies "
        "quickly and efficiently."
    )

if __name__ == "__main__":
    main()
