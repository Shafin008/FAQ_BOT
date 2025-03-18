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
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Import for splitting documents into chunks
from langchain.chains.combine_documents import create_stuff_documents_chain  # Import for creating document chains
from langchain_core.prompts import ChatPromptTemplate  # Import for creating prompt templates
from langchain.chains import create_retrieval_chain  # Import for creating retrieval chains
from langchain_community.vectorstores import FAISS  # Import for vector database
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader  # Import for loading PDFs
from langchain_openai import OpenAIEmbeddings  # Import for OpenAI embeddings
from langchain_core.documents import Document  # Import for document type
from typing import List, Optional  # Import for type hints

# Load environment variables from .env file if present
load_dotenv()

# Constants for application configuration
DATA_PATH: str = "./data/"  # Path to data directory (not used in current implementation)
MODEL_NAME: str = "Llama3-8b-8192"  # Name of the Groq LLM model to use
EMBEDDING_MODEL_NAME: str = "text-embedding-ada-002"  # Name of the OpenAI embedding model
VECTOR_STORE_NAME: str = "FAQ-BOT"  # Name of the vector store (not used in current implementation)
PERSIST_DIRECTORY: str = "./faiss_db"  # Directory to save the FAISS vector database

# Streamlit UI Setup
st.title("PDFSage: Ask Your Documents Anything 🤖")
st.write("Upload any PDF document and ask questions about its content. The chatbot will provide answers based on the information in your document.")

st.write("**How to use this app:**")
st.write("1. Enter your API keys in the section below")
st.write("2. Upload a PDF file using the file uploader")
st.write("3. Click 'Start Engine' to process your document")
st.write("4. Ask questions about the content of your PDF")

# API key instructions
st.write(
    "To use this app, you need to provide a Groq API Key which you can get [here](https://console.groq.com/keys) and an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys)")

# User API Key Input section in an expandable container
with st.expander("Click here to enter your API KEYs"):
    GROQ_API_KEY: str = st.text_input("Groq API Key", type="password")  # Text input for Groq API key with password masking
    if not GROQ_API_KEY:  # Check if Groq API key is provided
        st.info("Please add your Groq API key to continue.", icon="🗝️")  # Show info message if key is missing
        st.stop()  # Stop execution if key is missing
    
    OPENAI_API_KEY: str = st.text_input("OpenAI API Key", type="password")  # Text input for OpenAI API key with password masking
    if not OPENAI_API_KEY:  # Check if OpenAI API key is provided
        st.info("Please add your OpenAI API key to continue.", icon="🗝️")  # Show info message if key is missing
        st.stop()  # Stop execution if key is missing


def process_pdf(file) -> List[Document]:
    """
    Loads and splits an uploaded PDF file into chunks.
    
    Args:
        file: The uploaded PDF file object
        
    Returns:
        List[Document]: A list of document chunks after splitting
    """
    with open("temp.pdf", "wb") as f:  # Open a temporary file to save the uploaded PDF
        f.write(file.read())  # Write the uploaded file content to the temporary file
    
    loader = PyPDFLoader("temp.pdf")  # Create a PDF loader for the temporary file
    documents = loader.load()  # Load the PDF content into documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Create a text splitter with 1000 character chunks and 200 character overlap
    return splitter.split_documents(documents)  # Split the documents and return the chunks

# File upload section 
uploaded_file: Optional[bytes] = st.file_uploader("Upload a PDF file", type=["pdf"])  # Create a file uploader for PDF files


def vector_embedding() -> None:
    """
    Processes and embeds documents into a vector database.
    
    This function handles:
    1. Processing uploaded PDF files
    2. Initializing OpenAI embeddings
    3. Creating and saving a FAISS vector database
    """
    if "vector_db" in st.session_state:  # Check if vector_db already exists in session state
        del st.session_state.vector_db  # Delete existing vector database from session state
    
    if uploaded_file:  # Check if a file has been uploaded
        with st.spinner("Loading & Chunkifying Documents..."):  # Show a spinner while processing documents
            st.session_state.documents = process_pdf(uploaded_file)  # Process the uploaded PDF and store chunks in session state
    
        with st.spinner("Initializing Vector Database..."):  # Show a spinner while initializing the vector database
            st.session_state.embeddings = OpenAIEmbeddings(  # Initialize OpenAI embeddings
                model=EMBEDDING_MODEL_NAME,  # Use the specified embedding model
                api_key=OPENAI_API_KEY  # Use the provided OpenAI API key
            )
        
            st.session_state.vector_db = FAISS.from_documents(  # Create a FAISS vector database from documents
                st.session_state.documents,  # Use the documents stored in session state
                st.session_state.embeddings,  # Use the embeddings stored in session state
            )
            st.session_state.vector_db.save_local(PERSIST_DIRECTORY)  # Save the vector database locally


def main() -> None:
    """
    Runs the main chatbot application.
    
    This function handles:
    1. Managing chat history
    2. Processing user input
    3. Retrieving relevant document chunks
    4. Generating responses using the LLM
    5. Displaying the conversation
    """

    if "messages" not in st.session_state:  # Check if messages already exist in session state
        st.session_state.messages = []  # Initialize empty messages list in session state

    
    if st.button("Start Engine"):  # Create a button to start the engine
        # making sure user uploads file before the embedding starts
        if uploaded_file:
            vector_embedding()  # Call the vector_embedding function when button is clicked
            st.success("The Engine is ready.", icon="✅")  # Show success message when engine is ready
            st.session_state.messages = []  # Reset chat history
        else:
            # If the user don't supply any documents then it gives a reply
            st.write("Please Upload Your File.")
        
    
    for message in st.session_state.messages:  # Iterate through all messages in the chat history
        with st.chat_message(message["role"]):  # Create a chat message container with appropriate role
            st.markdown(message["content"])  # Display the message content using markdown

    # Define the prompt template for the chatbot
    prompt_template = ChatPromptTemplate.from_template(
        """
        You are an AI language model assistant who is expert in answering questions about the uploaded document from the provided context. Please provide the most accurate response based on the question. Also, at the end of your answer, ask the user if they are satisfied with your answer or need further assistance. If the user is satisfied and the {input} is positive such as 'yes', 'thanks', thank you', etc. then you don't need to answer based on the context, you just give a positive and satisfactory reply and stop asking anything. For further assistance, if the {input} is 'no' or 'no thanks', that means you don't need to answer based on the context, you just give a positive and satisfactory reply and stop asking anything. Don't write extra words in your answer if the user is satisfied or don't need assistance.
        <context>
        {context}
        <context>

        Question: {input}
        """
    )
    
    query_prompt: Optional[str] = st.chat_input("Enter Your Question Regarding the course....")  # Create a chat input for user questions
    
    if uploaded_file:
        if query_prompt:  # Check if user has entered a question
            with st.chat_message("user"):  # Create a chat message container for user
                st.markdown(query_prompt)  # Display the user's question using markdown
            st.session_state.messages.append({"role": "user", "content": query_prompt})  # Add user message to chat history

            with st.spinner("Generating response..."):  # Show a spinner while generating response
                llm = ChatGroq(  # Initialize ChatGroq LLM
                    model=MODEL_NAME,  # Use the specified model
                    api_key=GROQ_API_KEY  # Use the provided Groq API key
                )
                
                document_chain = create_stuff_documents_chain(  # Create a chain for combining documents with prompt
                    llm,  # Use the initialized LLM
                    prompt_template  # Use the defined prompt template
                )

                retriever = st.session_state.vector_db.as_retriever()  # Convert vector database to retriever
                
                retrieval_chain = create_retrieval_chain(  # Create a retrieval chain
                    retriever,  # Use the created retriever
                    document_chain  # Use the created document chain
                )

                response = retrieval_chain.invoke(  # Invoke the retrieval chain
                    {'input': query_prompt}  # Pass the user's question as input
                )
            
            with st.chat_message("assistant"):  # Create a chat message container for assistant
                st.markdown(response['answer'])  # Display the assistant's response using markdown
            st.session_state.messages.append({"role": "assistant", "content": response['answer']})  # Add assistant message to chat history
    else:
        st.write("Please Upload Your File.")


# Entry point for the application
if __name__ == "__main__":
    main()  # Call the main function when script is executed directly