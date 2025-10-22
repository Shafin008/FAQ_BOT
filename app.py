"""
PDF Chatbot Application

This application provides a Streamlit-based interface for uploading PDF documents,
processing them into a vector database, and answering questions about their content
using a combination of retrieval-augmented generation with Groq LLM and OpenAI embeddings.

Dependencies:
    - streamlit: For the web interface
    - langchain: For document processing and retrieval chains
    - openai: For embedding generation
    - groq: For LLM access
    - faiss: For vector storage
"""

import os  # Import for operating system related functionalities
from dotenv import load_dotenv  # Import to load environment variables from .env file
import streamlit as st  # Import Streamlit for the web application interface
from langchain_groq import ChatGroq  # Import for using Groq's language models
from langchain_ollama import OllamaEmbeddings  # Import for local embeddings (not used but available)
# Import for splitting documents into chunks
# New import
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import for creating document chains
# New imports
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate  # Import for creating prompt templates

from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS  # Import for vector database
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader  # Import for loading PDFs
from langchain_openai import OpenAIEmbeddings  # Import for OpenAI embeddings
from langchain_core.documents import Document  # Import for document type
from typing import List, Optional  # Import for type hints
from langchain_ollama import OllamaEmbeddings
import ollama

# Load environment variables from .env file if present
load_dotenv()

# Constants for application configuration
MODEL_NAME: str = "Llama3-8b-8192"  # Name of the Groq LLM model to use
VECTOR_STORE_NAME: str = "FAQ-BOT"  # Name of the vector store (not used in current implementation)
PERSIST_DIRECTORY: str = "./faiss_db"  # Directory to save the FAISS vector database

# Load model once (Runs only the first time)
@st.cache_resource
def load_embedding_model():
    model_name = "mxbai-embed-large"
    ollama.pull(model_name)  # Ensure model is downloaded
    return model_name

# EMBEDDING_MODEL_NAME: str = "text-embedding-ada-002"  # Name of the OpenAI embedding model
EMBEDDING_MODEL_NAME: str = load_embedding_model()


# Streamlit UI Setup
st.title("PDFSage: Ask Your Documents Anything 🤖")  # Set the title of the Streamlit app
st.write("Upload any PDF document and ask questions about its content. The chatbot will provide answers based on the information in your document.")  # Description of the app

st.write("**How to use this app:**")  # Instructions for using the app
st.write("1. Enter your required API keys in the section below")  # Step 1: Enter API keys
st.write("2. Upload PDF file(s) using the file uploader")  # Step 2: Upload PDF
st.write("3. Click 'Start Engine' to process your document")  # Step 3: Process document
st.write("4. Ask questions about the content of your PDF(s)")  # Step 4: Ask questions

# API Key Input
with st.expander("Click here to enter your API KEYs"):  # Expandable section for API key input
    GROQ_API_KEY: str = st.text_input("Groq API Key", type="password")  # Input for Groq API key
    # OPENAI_API_KEY: str = st.text_input("OpenAI API Key", type="password")  # Input for OpenAI API key
    # if not GROQ_API_KEY or not OPENAI_API_KEY:  # Check if API keys are provided
    #     st.info("Please add your API keys to continue.", icon="🗝️")  # Display info message if keys are missing
    #     st.stop()  # Stop the app if keys are missing

    
    if not GROQ_API_KEY:  # Check if API keys are provided
        st.info("Please add your API keys to continue.", icon="🗝️")  # Display info message if keys are missing
        st.stop()  # Stop the app if keys are missing


# Function to process PDF
def process_documents(uploaded_files) -> List[Document]:
    """Loads and splits multiple uploaded PDF files into chunks.
    
    Args:
        uploaded_files: List of uploaded PDF files.
    
    Returns:
        List[Document]: List of document chunks.
    """
    all_documents = []  # Initialize an empty list to store all documents
    for i, file in enumerate(uploaded_files): # Iterate over each uploaded file
        # Create a unique filename for each uploaded file
        temp_filename = f"temp_{i}.pdf"  
        with open(temp_filename, "wb") as f:
            f.write(file.read())  # Write the file content to disk
        loader = PyPDFLoader(temp_filename)  # Load the PDF file using PyPDFLoader
        documents = loader.load()  # Load the documents from the PDF
        for doc in documents:  # Iterate over each document
            doc.metadata["source"] = file.name  # Add the source file name to the document metadata
        all_documents.extend(documents)  # Add the documents to the list
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Initialize the text splitter
    return splitter.split_documents(all_documents)  # Split the documents into chunks and return

# File Upload
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)  # File uploader for PDFs

# Vector Embedding
def vector_embedding() -> None:
    """Processes and embeds documents into a vector database."""
    if "vector_db" in st.session_state:  # Check if vector database exists in session state
        del st.session_state.vector_db  # Reset vector DB if it exists
    if uploaded_files and len(uploaded_files) > 0:  # Check if files are uploaded
        with st.spinner(f"Processing {len(uploaded_files)} document(s)..."):  # Show spinner while processing documents
            st.session_state.documents = process_documents(uploaded_files)  # Process and store documents in session state
        with st.spinner("Initializing Vector Database..."):  # Show spinner while initializing vector database
            # st.session_state.embeddings = OpenAIEmbeddings(  # Initialize OpenAI embeddings
            #     model=EMBEDDING_MODEL_NAME,
            # )
            st.session_state.embeddings = OllamaEmbeddings(  # Initialize OpenAI embeddings
                model=EMBEDDING_MODEL_NAME,
            )
            st.session_state.vector_db = FAISS.from_documents(  # Create FAISS vector database from documents
                st.session_state.documents,
                st.session_state.embeddings,
            )
            st.session_state.vector_db.save_local(PERSIST_DIRECTORY)  # Save the vector database locally
    else:
        st.error("Please upload at least one PDF file.")  # Show error if no files are uploaded

# Main Function
def main() -> None:
    """Runs the main chatbot application."""
    if "messages" not in st.session_state:  # Check if chat messages exist in session state
        st.session_state.messages = []  # Initialize an empty list for chat messages

    if st.button("Start Engine"):  # Button to start the engine
        if uploaded_files:  # Check if files are uploaded
            vector_embedding()  # Process and embed documents
            st.success("The Engine is ready.", icon="✅")  # Show success message
            st.session_state.messages = []  # Reset chat history
        else:
            st.write("Please Upload Your File.")  # Prompt to upload files
    
    for message in st.session_state.messages:  # Iterate over chat messages
        with st.chat_message(message["role"]):  # Display chat message based on role (user/assistant)
            st.markdown(message["content"])  # Display the message content

    prompt_template = ChatPromptTemplate.from_template(  # Create a prompt template for the chatbot
        """
        You are an AI language model assistant who is expert in answering questions about the uploaded document from the provided context. Please provide the most accurate response based on the question. Please, remember what answer you gave previously. Use the previous conversation to improve your responses.
        
        Also, at the end of your answer, ask the user if they are satisfied with your answer or need further assistance. If the user is satisfied and the {input} is positive such as 'yes', 'thanks', thank you', etc. then you don't need to answer based on the context,you just give a positive and satisfactory reply and stop asking anything. For further assistance, if the {input} is 'no' or 'no thanks', that means you don't need to answer based on the context, you just give a positive and satisfactory reply and stop asking anything. Don't write extra words in your answer if the user is satisfied or don't need assistance.

        <context>
        {context}
        <context>

        <Previous Conversation>
        {history}
        <Previous Conversation>

        Question: {input}
        
        """
    )

    query_prompt: Optional[str] = st.chat_input("Ask a question about your document...")  # Input for user query
    if uploaded_files and query_prompt:  # Check if files are uploaded and query is provided
        with st.chat_message("user"):  # Display user message
            st.markdown(query_prompt)  # Display the user query
        st.session_state.messages.append({"role": "user", "content": query_prompt})  # Add user query to chat history

        with st.spinner("Generating response..."):  # Show spinner while generating response
            history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])  # Create chat history string

            llm = ChatGroq(model=MODEL_NAME, api_key=GROQ_API_KEY)  # Initialize Groq LLM

            document_chain = create_stuff_documents_chain(llm, prompt_template)  # Create document chain for processing

            retriever = st.session_state.vector_db.as_retriever()  # Create retriever from vector database

            retrieval_chain = create_retrieval_chain(retriever, document_chain)  # Create retrieval chain

            response = retrieval_chain.invoke({"input": query_prompt, "history": history})  # Invoke the retrieval chain with user query and history
        
        with st.chat_message("assistant"):  # Display assistant message
            st.markdown(response['answer'])  # Display the assistant's response
        st.session_state.messages.append({"role": "assistant", "content": response['answer']})  # Add assistant response to chat history

if __name__ == "__main__":
    main()  # Run the main function when the script is executed