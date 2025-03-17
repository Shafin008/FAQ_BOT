# 📚 FAQ Chatbot Application

![RAG BASED APP](https://machinelearningmastery.com/wp-content/uploads/2024/08/mlm-awan-rag-applications-llamaindex.png)

<p align="center">RAG based application cycle</p>

This is an **AI-powered web-based chatbot** application designed to answer frequently asked questions about the **Data Engineering Zoomcamp** [course](https://github.com/DataTalksClub/data-engineering-zoomcamp). It utilizes **LangChain, FAISS, and Llama3** to provide accurate and contextual responses from a **233-page PDF** of FAQs. It leverages advanced technologies for retrieval-augmented generation (RAG), making it a powerful tool for course-related queries. Key features include:

- **PDF Document Processing**: The app loads and chunks PDF files from a specified directory (`./data/`), enabling users to base queries on course materials.
- **Vector Database Management**: It uses FAISS (Facebook AI Similarity Search) to store document embeddings, ensuring fast and efficient retrieval of relevant information.
- **AI-Powered Q&A**: Combines OpenAI embeddings for document representation with Grok's Llama3-8b-8192 model for generating accurate, context-aware responses.
- **Interactive User Interface**: Built with Streamlit, offering a simple interface where users can input API keys, start the engine, and ask questions via a chat input field.
- **Deployment Readiness**: Successfully deployed on Streamlit Cloud, accessible at [course-faq-app](https://course-faq-bot.streamlit.app/), demonstrating scalability and accessibility.

These features make the chatbot an effective tool for learners, providing instant answers based on course documents and ensuring a user-friendly experience.

## 🚀 Features  
- **AI-powered Responses**: Uses `Llama3-8b-8192` for intelligent responses.  
- **PDF Knowledge Base**: Extracts and processes FAQs from the course document.  
- **Vector Search**: Utilizes `FAISS` for efficient retrieval.  
- **Embeddings**: Uses `OpenAIEmbeddings` for document vectorization.  
- **Streamlit UI**: Interactive chatbot with a user-friendly interface.  

## 🛠️ Tech Stack  
- **Python**  
- **Streamlit** (for UI)  
- **LangChain** (for LLM and retrieval chain)  
- **FAISS** (for vector storage)  
- **OpenAIEmbeddings** (for text embedding)  
 

## 📂 Project Structure  
```
📁 project-root  
 ┣ 📁 data/                  # PDF source files  
 ┣ 📁 faiss_db/              # FAISS vector database  
 ┣ 📜 app.py                 # Main chatbot application  
 ┣ 📜 .env                   # Environment variables (API keys)  
 ┗ 📜 README.md              # Project documentation
 ┗ 📜 requirements.txt       # Lists all Python dependencies for installation

```

## 🔧 Installation & Setup  

1. **Clone the repository**  
   ```bash
   git clone https://github.com/Shafin008/FAQ_BOT.git
   cd FAQ_BOT
   ```

2. **Create a virtual environment**  
   ```bash
   python -m venv venv  
   source venv/bin/activate  # On Windows: venv\Scripts\activate  
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt  
   ```

## 🔧 Usage Instructions
Once set up, users can run and interact with the chatbot as follows:

1. **Run the App Locally**: Execute the following command:
   ```bash
   streamlit run app.py
   ```
   This will launch the app in a web browser at `http://localhost:8501`.

2. **Interact with the Chatbot**:
   - First, enter your Groq and OpenAI API keys in the provided password fields on the Streamlit interface.
   - Click the "Start Engine" button to initialize the vector database, which processes the PDF documents and generates embeddings.
   - Use the chat input field to ask questions about the data engineering course. The bot will respond based on the retrieved document chunks, using the Grok LLM for natural language generation.
   - Each response includes a prompt asking if the user is satisfied or needs further assistance, enhancing user engagement.

The interface is designed for simplicity, making it accessible for users with minimal technical background.

## 🔧 Deployment Details
The app is already deployed on Streamlit Cloud, accessible at [https://course-faq-bot.streamlit.app/](https://course-faq-bot.streamlit.app/). For users wishing to deploy their own version:

1. Push the repository to GitHub.
2. Connect it to Streamlit Cloud via their platform.
3. Set the following secrets in Streamlit Cloud’s settings:
   ```
   OPENAI_API_KEY=your-openai-API-key
   GROQ_API_KEY=your-groq-API-key
   ```
4. Deploy the app, and it will be live for users to access.

This deployment method ensures scalability and accessibility, with the live URL providing a demonstration of the chatbot’s functionality.

## 🔧 Dependencies and Technical Stack
The project relies on several key libraries, as detailed in `requirements.txt`. A summary of the technical stack includes:

| Library         | Purpose                                      |
|-----------------|----------------------------------------------|
| `streamlit`     | Web app framework for the user interface     |
| `langchain`     | RAG pipeline, LLM integration, document processing |
| `faiss-cpu`     | Vector store for efficient document retrieval |
| `pypdf`         | PDF processing and loading                   |
| `python-dotenv` | Environment variable management              |

These libraries ensure robust functionality, with `langchain` being particularly crucial for the RAG pipeline, integrating OpenAI embeddings and Groq’s LLM.

## 📌 Future Enhancements  
- Multi-course support  
- Integration with other LLMs  
- Improved UI with advanced search filters

#### 🤝 Contributing and Licensing
The project welcomes contributions from the community. Users can submit issues or pull requests on GitHub, fostering collaboration and improvement. The project is licensed under the MIT License, which is permissive and allows for both commercial and non-commercial use. Users should create a `LICENSE` file if not already present, with details available at [MIT License](https://opensource.org/licenses/MIT).

#### Acknowledgements
This project builds on the capabilities of several open-source tools and platforms, including:
- [Streamlit](https://streamlit.io/) for the web interface.
- [LangChain](https://langchain.com/) for RAG and LLM integration.


