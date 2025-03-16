import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings

# Loading Environment and constants
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Constants
DATA_PATH = './data/'
MODEL_NAME = 'Llama3-8b-8192'
EMBEDDING_MODEL_NAME = "text-embedding-3-large"
VECTOR_STORE_NAME = "FAQ-BOT"
PERSIST_DIRECTORY = "./faiss_db"


# Vector Embedding
def vector_embedding():
    if "vector_db" not in st.session_state:

        # initialize embedding model
        st.session_state.embeddings=OpenAIEmbeddings(
            model=EMBEDDING_MODEL_NAME
        )
        st.write("Initializing embedding model...")

        # Document loading
        # initializing loader
        st.session_state.loader = PyPDFDirectoryLoader(DATA_PATH)
        # loading documents
        st.session_state.docs = st.session_state.loader.load()

        # Chunkifying documents
        # initializing text splitter
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        # chunkify documents
        st.session_state.documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.write("Loading & Chunkifying Documents...")

        # creating our vector database
        st.session_state.vector_db = FAISS.from_documents(
            st.session_state.documents,
            st.session_state.embeddings,
        ) 
        # saving session
        st.session_state.vector_db.save_local(PERSIST_DIRECTORY)   
        st.write("Initiating Vector Database...")

def main():
    # Creating system prompt template
    prompt_template = ChatPromptTemplate.from_template(
    """
    You are an AI language model assistant who is expert in answering questions about the data engineering course from the provided context. Please provide the most accurate response based on the question. Also, at the end of your answer, ask the user if the user is satisfied with your answer or not or if the user need further assistance. 
    <context>
    {context}
    <context>

    Questions:{input}
    """
    )

    # Title and Description for the app
    st.title("FAQ Chatbot 🤖")
    st.write("Please click the button below to get the ENGINE ready 🔥🔥🔥. Once the ENGINE is ready, please ask your questions.")
    
    # Button for creating database
    if st.button("Start Engine"):
        vector_embedding()
        st.write("Vector Database is ready....")

    # User query
    query_prompt = st.chat_input("Enter Your Question Regarding the course....")
    if query_prompt:
        with st.spinner("Generating response..."):
            # llm  
            # Initialize the language model
            llm = ChatGroq(
                model=MODEL_NAME,
                api_key=GROQ_API_KEY)
            
            document_chain = create_stuff_documents_chain(
                llm,
                prompt_template
            )

            retriever = st.session_state.vector_db.as_retriever()

            retrieval_chain = create_retrieval_chain(
                 retriever,
                 document_chain
            )

            response = retrieval_chain.invoke(
            {'input': query_prompt}
            )
            
            st.write(response['answer'])

if __name__ == "__main__":
    main()
