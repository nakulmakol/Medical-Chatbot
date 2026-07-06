# MediBot-an AI medical chatbot
An AI-powered medical chatbot that leverages **Retrieval-Augmented Generation (RAG)**, **LangChain**, **FAISS**, and **Groq Llama 3.3 70B** to deliver context-aware responses to medical queries using a curated medical knowledge base.

---

## 🌐 Live Demo

🚀 **Try it here:**

**https://medical-chatbot-production-ed14.up.railway.app/**

---

## 📖 Overview

MediBot AI is an intelligent healthcare assistant designed to answer medical questions using a combination of semantic search and Large Language Models.

Instead of relying solely on an LLM, the chatbot first retrieves the most relevant information from a medical knowledge base using **FAISS vector search** and **Sentence Transformers**. This retrieved context is then provided to **Groq's Llama 3.3 70B** model to generate accurate and context-aware responses.

The project demonstrates the practical implementation of Retrieval-Augmented Generation (RAG) in healthcare applications.

---
<img width="2872" height="1538" alt="image" src="https://github.com/user-attachments/assets/825ee2f7-fd9f-42ea-b40e-4cce68e8f76f" />

## ✨ Features

- 🤖 AI-powered medical chatbot
- 🧠 Retrieval-Augmented Generation (RAG)
- 📚 Semantic search using FAISS Vector Database
- 🔍 Context-aware responses
- ⚡ Powered by Groq Llama 3.3 70B
- 🏥 Medical knowledge retrieval
- 📱 Fully responsive user interface
- 🌐 Railway deployment
- 🔒 Secure API key management using environment variables

---

## 🛠️ Tech Stack

### Backend

- Python
- Flask
- Flask-CORS

### AI & LLM

- LangChain
- LangChain Community
- LangChain Groq
- LangChain HuggingFace
- Groq API
- Llama 3.3 70B

### Embeddings

- Sentence Transformers
- all-MiniLM-L6-v2

### Vector Database

- FAISS

### Frontend

- HTML5
- CSS3
- JavaScript

### Deployment

- Railway

---

## 🏗️ Architecture

```
                 User
                   │
                   ▼
          Responsive Web Interface
                   │
                   ▼
             Flask Backend
                   │
      ┌────────────┴────────────┐
      │                         │
      ▼                         ▼
 FAISS Vector Store        Groq LLM
      │
      ▼
Medical Knowledge Base
```

---

## 📂 Project Structure

```
Medical-Chatbot/
│
├── app.py
├── requirements.txt
├── Procfile
├── railway.json
├── .env.example
│
├── templates/
│   └── index.html
│
├── static/
│
├── vectorstore/
│   ├── index.faiss
│   └── index.pkl
│
├── research/
│
└── README.md
```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/nakulmakol/Medical-Chatbot.git
```

```bash
cd Medical-Chatbot
```

### Create a virtual environment

```bash
python -m venv venv
```

### Activate the environment

#### Windows

```bash
venv\Scripts\activate
```

#### macOS/Linux

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Create a `.env` file

```env
GROQ_API_KEY=YOUR_GROQ_API_KEY
```

### Run the application

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

## 💬 Example Questions

- What are the symptoms of diabetes?
- Explain hypertension in simple words.
- What are the side effects of paracetamol?
- What causes asthma?
- What is insulin resistance?
- How is dengue diagnosed?
- What is high blood pressure?

---

## 🚀 Deployment

The application is deployed on **Railway**.

**Live URL**

https://medical-chatbot-production-ed14.up.railway.app/

---

## 📈 Future Improvements

- Conversation history
- User authentication
- Voice-based interaction
- Multi-language support
- Medical report summarization
- Appointment assistance
- Drug interaction checking
- PDF medical report upload

---

## ⚠️ Disclaimer

This chatbot is intended **only for educational and informational purposes**.

It does **not** replace professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for medical concerns.

---

## 👨‍💻 Author

**Nakul Makol**

- GitHub: **https://github.com/nakulmakol**
- LinkedIn: **https://www.linkedin.com/in/nakul-makol-b3abb6310/**

---

## ⭐ If you found this project interesting

If you like this project, consider giving it a **⭐ Star** on GitHub.

It helps others discover the project and supports future development.
