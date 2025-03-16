# 📚 FAQ Chatbot for Data Engineering Zoomcamp  

This is an **AI-powered chatbot** designed to answer frequently asked questions about the **Data Engineering Zoomcamp** course. It utilizes **LangChain, FAISS, and Llama3** to provide accurate and contextual responses from a **233-page PDF** of FAQs.  

## 🚀 Features  
- **AI-powered Responses**: Uses `Llama3-8b-8192` for intelligent responses.  
- **PDF Knowledge Base**: Extracts and processes FAQs from the course document.  
- **Vector Search**: Utilizes `FAISS` for efficient retrieval.  
- **Embeddings**: Uses `OllamaEmbeddings` for document vectorization.  
- **Streamlit UI**: Interactive chatbot with a user-friendly interface.  

## 🛠️ Tech Stack  
- **Python**  
- **Streamlit** (for UI)  
- **LangChain** (for LLM and retrieval chain)  
- **FAISS** (for vector storage)  
- **OllamaEmbeddings** (for text embedding)  
 

## 📂 Project Structure  
```
📁 project-root  
 ┣ 📁 data/                  # PDF source files  
 ┣ 📁 faiss_db/              # FAISS vector database  
 ┣ 📜 app.py                 # Main chatbot application  
 ┣ 📜 .env                   # Environment variables (API keys)  
 ┗ 📜 README.md              # Project documentation  
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
     GROQ_API_KEY=your_api_key
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
