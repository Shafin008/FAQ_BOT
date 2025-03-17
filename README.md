# FAQ Chatbot for Data Engineering Zoomcamp 🚀

## 📌 Introduction
The **FAQ Chatbot** is an AI-powered assistant designed to help users with queries related to the **[Data Engineering Zoomcamp](https://github.com/DataTalksClub/data-engineering-zoomcamp)**. This chatbot processes a 233-page FAQ document and provides relevant answers in an interactive manner. The system is built with **Streamlit**, **LangChain**, and **FAISS**, using **LLMs** to generate responses dynamically.

🔗 **Live App:** [course-faq-bot.streamlit.app](https://course-faq-bot.streamlit.app/)

## ✨ Features
✅ Provides quick and accurate answers to FAQs from the **Data Engineering Zoomcamp** course.\
✅ Uses **FAISS** for fast vector search on a 233-page FAQ document.\
✅ Supports **OpenAI** & **Groq LLMs** for natural language understanding.\
✅ Simple and interactive UI with **Streamlit**.\
✅ Secure API key input via **Streamlit Expander**.

## 🛠️ Tech Stack
- **Python**  
- **Streamlit** (for UI)  
- **LangChain** (for LLM and retrieval chain)  
- **FAISS** (for vector storage)  
- **OpenAIEmbeddings** (for text embedding)  

## 🚀 How It Works
1. The chatbot loads the FAQ document (233-page PDF) and splits it into manageable chunks.
2. These chunks are embedded using **OpenAI's embedding model**.
3. FAISS is used to store and retrieve relevant embeddings.
4. A retrieval chain fetches the most relevant data, which is then processed by the language model.
5. The chatbot provides responses in a conversational format, asking if the user needs further assistance.

## 🏗️ Project Structure
```
FAQ_BOT/
│── data/                  # Directory containing FAQ PDF files
│── faiss_db/              # Persisted FAISS database
│── app.py                 # Main Streamlit application
│── requirements.txt       # Python dependencies
│── .env.example           # Example environment variables
│── README.md              # Documentation
```

## 🏗️ Installation & Setup
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Shafin008/FAQ_BOT.git
cd FAQ_BOT
```

### 2️⃣ Create a Virtual Environment (Optional but Recommended)
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
To use the chatbot, you need API keys for **Groq** and **OpenAI**.

- **Groq API Key:** [Get it here](https://console.groq.com/keys)
- **OpenAI API Key:** [Get it here](https://platform.openai.com/account/api-keys)

### 5️⃣ Run the App
```bash
streamlit run app.py
```

## 🎥 Demo

![Demo Video](https://youtu.be/k57B7RFoUMw)

## 🤖 Interact with the Chatbot
1. **Enter API Keys:** Provide your **Groq** and **OpenAI** API keys in the password fields on the Streamlit interface.
2. **Start Engine:** Click the **"Start Engine"** button to initialize the vector database, which processes the FAQ PDF document and generates embeddings.
3. **Ask a Question:** Use the chat input field to ask questions about the data engineering course.
4. **Get Responses:** The bot retrieves relevant document chunks and uses **Groq's LLM** for natural language generation.
5. **User Engagement:** Each response includes a follow-up prompt asking if you are satisfied or need further assistance.

## 🔮 Future Enhancements
✅ Add support for multiple courses.\
✅ Implement a feedback system to improve responses.\
✅ Deploy a **REST API** for broader accessibility.\
✅ Integrate voice-based interaction.

## 🤝 Contributing
We welcome contributions! Feel free to submit a PR or open an issue.

## 📜 License
This project is licensed under the **MIT License**.

## 📬 Contact
📧 Email: [shafinmahmud114@gmail.com](mailto:shafinmahmud114@gmail.com)  
🐦 Twitter: [@shafinmahmud114](https://x.com/shafinmahmud114)  
💼 LinkedIn: [Shafin Mahmud Jalal](https://www.linkedin.com/in/shafin-mahmud-jalal-8a76b3143/)

---

⭐ If you find this project helpful, give it a **star**!

