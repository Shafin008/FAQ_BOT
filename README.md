# PDF Question-Answering Chatbot 🤖

## 📌 Introduction
The **PDF Question-Answering Chatbot** is an AI-powered assistant designed to help users extract information from **multiple PDF documents**. This versatile tool allows you to upload **one or more PDFs** and ask questions about their content, receiving accurate and contextual responses. The system is built with **Streamlit**, **LangChain**, **FAISS**, and leverages state-of-the-art LLMs to generate dynamic responses.

🔗 **Live App:** [course-faq-bot.streamlit.app](https://course-faq-bot.streamlit.app/)

## ✨ Features
✅ Process and analyze **multiple PDF documents** in a single session  
✅ Ask questions about the combined content of uploaded PDFs and receive AI-generated answers  
✅ Uses **FAISS** for fast and efficient vector search capabilities  
✅ Supports **Groq** and **OpenAI** models for natural language processing  
✅ Clean, interactive UI built with **Streamlit**  
✅ Secure API key management via **Streamlit Expander**  
✅ Document chunking for optimal context retrieval  

## 🛠️ Tech Stack
- **Python**  
- **Streamlit** (for user interface)  
- **LangChain** (for document processing and retrieval chains)  
- **FAISS** (for vector storage and similarity search)  
- **OpenAI Embeddings** (for text embedding)  
- **Groq LLM** (for response generation)  

## 🚀 How It Works
1. Upload **one or more PDF documents** through the Streamlit interface.  
2. The application splits the documents into manageable chunks.  
3. These chunks are embedded using **OpenAI's embedding model**.  
4. FAISS stores these embeddings in a vector database for efficient retrieval.  
5. When you ask a question, the system retrieves the most relevant document sections from **all uploaded PDFs**.  
6. **Groq's LLM** processes these sections along with your question to generate a contextual response.  
7. The chatbot presents answers in a conversational format, asking if you need additional information.  

## 🏗️ Project Structure
```
FAQ_BOT/
│── data/                  # Temporary storage for uploaded PDF files
│── faiss_db/              # Persisted FAISS database
│── app.py                 # Main Streamlit application
│── requirements.txt       # Python dependencies
│── README.md              # Documentation
│── .env                   # Environment variables (API keys)
```

## 🏗️ Installation & Setup

### 🌐 Using the Deployed App
The easiest way to try the app is to visit the live deployment:  
- **Live App:** [course-faq-bot.streamlit.app](https://course-faq-bot.streamlit.app/)

### 1️⃣ Clone the Repository (For Local Development)
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
1. **Enter Your API Keys**: Provide your Groq and OpenAI API keys in the password fields (click "Click here to enter your API KEYs").  
2. **Upload PDFs**: Use the file uploader to upload **one or more PDF documents** you want to query.  
3. **Start Engine**: Click the "Start Engine" button to process your documents into the vector database.  
4. **Ask Questions**: Use the chat input to ask specific questions about the content of the uploaded PDFs.  
5. **Get Answers**: Receive contextual responses based on the combined content of all uploaded documents.  
6. **Follow-up**: The bot will ask if you're satisfied or need more information.  

## 💡 Example Use Cases
- **Academic Research**: Extract specific information from **multiple research papers** simultaneously.  
- **Legal Document Analysis**: Query **multiple legal documents** for specific clauses or information.  
- **Technical Documentation**: Ask questions about **multiple technical manuals or guides** at once.  
- **Course Materials**: Learn from **multiple educational PDFs** with interactive Q&A.  
- **Financial Reports**: Extract insights from **multiple financial documents** through natural language queries.  

## 🔮 Future Enhancements
✅ Add support for multiple document formats (DOCX, TXT, etc.).  
✅ Implement document summarization capabilities.  
✅ Add option to compare information across multiple documents.  
✅ Create visualization tools for document insights.  
✅ Deploy as a standalone web service with user accounts.  

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