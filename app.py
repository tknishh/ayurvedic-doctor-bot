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

# Fixed directory containing PDF documents
pdf_directory = "data/"

# Function to list PDF files in a directory
def list_pdf_files(directory):
    pdf_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            pdf_files.append(os.path.join(directory, filename))
    return pdf_files

# Function to extract data from all the PDF documents
def get_data(docs):
    data = ""
    for doc in docs:
        pdf_reader = PdfReader(doc)
        for page in pdf_reader.pages:
            data += page.extract_text()
    return data

# Convert data to chunks
def get_chunks(raw_data):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(raw_data)
    return chunks

# Convert chunks to embeddings and create a vector store
def get_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

# ... (other functions remain the same)

def main():
    load_dotenv()

    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        st.write("OPENAI_API_KEY is not set. Please add your key in .env file.")
        exit(1)

    st.set_page_config(page_title="Ayurvedic Doctor Chatbot", page_icon="💬")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Ayurvedic Doctor Chatbot 💬")

    with st.chat_message("assistant"):
        st.write("Hello👋, How can I help you today?")

    user_input = st.chat_input("Ask your query")

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        handle_user_input(user_input)

    # Process PDF files from the fixed directory
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
