import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Function to list PDF files in a directory
def list_pdf_files(directory):
    pdf_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            pdf_files.append(os.path.join(directory, filename))
    return pdf_files

# ...

def main():
    load_dotenv()

    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        st.write("OPENAI_API_KEY is not set. Please add your key in .env file.")
        exit(1)

    st.set_page_config(page_title="Knowledge Base Chatbot", page_icon="💬")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Ayurvedic-Doctor-Bot 💬")

    with st.chat_message("assistant"):
        st.write("Hello👋, How can I help you today?")

    user_input = st.chat_input("Ask your query")

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        handle_user_input(user_input)

    # Add a sidebar for specifying the directory containing PDF files
    with st.sidebar:
        st.subheader("Directory with PDF documents")
        pdf_directory = st.text_input("Enter the path to the directory:")
        if pdf_directory and st.button("Process"):
            pdf_files = list_pdf_files(pdf_directory)
            if not pdf_files:
                st.warning("No PDF files found in the specified directory.")
            else:
                with st.spinner("Processing..."):
                    # Get data from the documents in the directory
                    raw_data = get_data(pdf_files)

                    # Divide data into chunks
                    chunks = get_chunks(raw_data)

                    # Convert to embeddings and create a vector store
                    vectorstore = get_vectorstore(chunks)

                    if vectorstore:
                        st.caption("Processing Completed!")

                    # Create a conversation chain (user input)
                    st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()
