# PDF Question-Answering Chatbot 🤖

## 📌 Introduction
The **PDF Question-Answering Chatbot** is an AI-powered assistant designed to help users extract information from any PDF document. This versatile tool allows you to upload your PDFs and ask questions about their content, receiving accurate and contextual responses. The system is built with **Streamlit**, **LangChain**, **FAISS**, and leverages state-of-the-art LLMs to generate dynamic responses.

## ✨ Features
✅ Process and analyze **any PDF document** you upload\
✅ Ask questions about document content and receive AI-generated answers\
✅ Uses **FAISS** for fast and efficient vector search capabilities\
✅ Supports **Groq** and **OpenAI** models for natural language processing\
✅ Clean, interactive UI built with **Streamlit**\
✅ Secure API key management via **Streamlit Expander**\
✅ Document chunking for optimal context retrieval

## 🛠️ Tech Stack
- **Python**  
- **Streamlit** (for user interface)  
- **LangChain** (for document processing and retrieval chains)  
- **FAISS** (for vector storage and similarity search)  
- **OpenAI Embeddings** (for text embedding)
- **Groq LLM** (for response generation)

## 🚀 How It Works
1. Upload any PDF document through the Streamlit interface
2. The application splits the document into manageable chunks
3. These chunks are embedded using **OpenAI's embedding model**
4. FAISS stores these embeddings in a vector database for efficient retrieval
5. When you ask a question, the system retrieves the most relevant document sections
6. **Groq's LLM** processes these sections along with your question to generate a contextual response
7. The chatbot presents answers in a conversational format, asking if you need additional information

## 🏗️ Project Structure
```
PDF_Chatbot/
│── faiss_db/              # Persisted FAISS database
│── pdf_app.py             # Main Streamlit application
│── requirements.txt       # Python dependencies
│── temp.pdf               # Temporary storage for uploaded PDF
│── .env                   # Environment variables (API keys)
│── README.md              # Documentation
```

## 🏗️ Installation & Setup
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Shafin008/FAQ_BOT.git
cd FAQ_BOT
```

### 2️⃣ Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate    # On Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Set Up API Keys
You'll need API keys for both services:

- **Groq API Key:** [Get it here](https://console.groq.com/keys)
- **OpenAI API Key:** [Get it here](https://platform.openai.com/account/api-keys)

You can either:
- Add them to a `.env` file in the project root
- Input them directly in the Streamlit interface when prompted

### 5️⃣ Run the App
```bash
streamlit run app.py
```

## 🤖 Using the Chatbot
1. **Enter Your API Keys**: Provide your Groq and OpenAI API keys in the password fields (click "Click here to enter your API KEYs")
2. **Upload a PDF**: Expand the file upload section and upload any PDF document you want to query
3. **Start Engine**: Click the "Start Engine" button to process your document into the vector database
4. **Ask Questions**: Use the chat input to ask specific questions about the document content
5. **Get Answers**: Receive contextual responses based on the document's content
6. **Follow-up**: The bot will ask if you're satisfied or need more information

## 💡 Example Use Cases
- **Academic Research**: Extract specific information from research papers
- **Legal Document Analysis**: Query complex legal documents for specific clauses or information
- **Technical Documentation**: Ask questions about technical manuals or guides
- **Course Materials**: Learn from educational PDFs with interactive Q&A
- **Financial Reports**: Extract insights from financial documents through natural language queries

## 🔮 Future Enhancements
✅ Add support for multiple document formats (DOCX, TXT, etc.)\
✅ Implement document summarization capabilities\
✅ Add option to compare information across multiple documents\
✅ Create visualization tools for document insights\
✅ Deploy as a standalone web service with user accounts

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an Issue if you have suggestions or encounter problems.

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 📬 Contact
If you have any questions or suggestions, feel free to reach out:

📧 Email: [shafinmahmud114@gmail.com](mailto:shafinmahmud114@gmail.com)  
🐦 Twitter: [@shafinmahmud114](https://x.com/shafinmahmud114)  
💼 LinkedIn: [Shafin Mahmud Jalal](https://www.linkedin.com/in/shafin-mahmud-jalal-8a76b3143/)

---

⭐ If you find this project helpful, please consider giving it a star!