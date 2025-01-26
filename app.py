import streamlit as st
from rag_chatbot import NPCILHRBot

# Initialize HR Bot
@st.cache_resource
def initialize_bot():
    return NPCILHRBot()

def main():
    st.title("NPCIL HR Bot 🤖")
    st.subheader("Your Intelligent HR Policy Assistant")
    
    # Initialize bot
    hr_bot = initialize_bot()
    
    # Chat input
    user_query = st.text_input("Ask a question about HR policies:")
    
    if user_query:
        # Generate response
        response = hr_bot.generate_response(user_query)
        
        # Display response
        st.write("Bot's Response:")
        st.info(response)
    
    # Sidebar information
    st.sidebar.title("About NPCIL HR Bot")
    st.sidebar.info(
        "An AI-powered assistant to help you navigate HR policies "
        "quickly and efficiently."
    )

if __name__ == "__main__":
    main()
