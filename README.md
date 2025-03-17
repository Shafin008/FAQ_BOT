# 📚 FAQ Chatbot Application

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
 ┗ 📜 requirements.txt       # Library Requirements

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

4. **Set up environment variables**  
   - Create a `.env` file and add:  
     ```
     GROQ_API_KEY=your_groq_api_key
     OPENAI_API_KEY=your_openai_api_key
     ```

5. **Run the chatbot**  
   ```bash
   streamlit run app.py  
   ```

## 🏃 Usage  
- Click the **"Start Engine"** button to initialize the vector database.  
- Enter your **course-related query** in the chat input.  
- Get accurate answers with follow-up suggestions.  

## 📌 Future Enhancements  
- Multi-course support  
- Integration with other LLMs  
- Improved UI with advanced search filters  

## 🤝 Contributing  
Contributions are welcome! Please fork the repo, create a branch, and submit a pull request.  

## 📜 License  
MIT License.  
