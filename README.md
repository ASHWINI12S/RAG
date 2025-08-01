﻿Here's a well-formatted and comprehensive `README.md` for your RAG (Retrieval-Augmented Generation) PDF Question-Answering app using Streamlit and LangChain:

---

````markdown
# 📄 RAG-based PDF Question Answering App

This is a Streamlit web application that uses **Retrieval-Augmented Generation (RAG)** with LangChain to answer user queries based on the contents of a PDF document. The application uses HuggingFace sentence embeddings, FAISS vector store, and Groq’s `gemma2-9b-it` model for generating responses.

---

## 🚀 Features

- Upload and read content from a PDF file.
- Split the PDF content into manageable text chunks.
- Convert text chunks into vector embeddings using HuggingFace Transformers.
- Store and search vector embeddings with FAISS.
- Use Groq's `gemma2-9b-it` model for answering questions.
- Ask natural language questions based on PDF content.
- Simple web interface built with Streamlit.

---

## 🛠️ Tech Stack

| Tool            | Purpose                                  |
|-----------------|------------------------------------------|
| Streamlit       | Web app UI                               |
| LangChain       | RAG pipeline                             |
| HuggingFace     | Sentence embedding model                 |
| FAISS           | Vector store for semantic search         |
| PyPDF2          | PDF text extraction                      |
| Groq API        | LLM backend for answer generation        |

---

## 📂 Project Structure

```bash
rag-pdf-app/
│
├── app.py               # Main Streamlit application
├── Cheenai_LTT.pdf      # Sample PDF (optional)
└── README.md            # Project documentation
````

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/rag-pdf-app.git
cd rag-pdf-app
```

### 2. Create & Activate Virtual Environment

```bash
python -m venv venv
.\venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Sample `requirements.txt`:**

```txt
streamlit
langchain
langchain-community
PyPDF2
faiss-cpu
sentence-transformers
```

### 4. Add Your Groq API Key

Replace this line in `app.py` with your own key:

```python
groqapi = 'your_groq_api_key'
```

---

## 🧠 How It Works

1. **PDF Upload**: Reads content from a local PDF using `PyPDF2`.
2. **Text Splitting**: Splits content into chunks using LangChain’s `RecursiveCharacterTextSplitter`.
3. **Embedding Generation**: Uses HuggingFace's `all-MiniLM-L6-v2` model to embed chunks.
4. **Vector Store**: Chunks are stored in a FAISS index.
5. **Retriever**: Fetches the most relevant chunks based on user queries.
6. **RAG Prompting**: Combines retrieved context with the user question and prompts Groq’s LLM.
7. **Answer Display**: Outputs the generated response in Streamlit.

---

## 🖼️ App Preview

![App Screenshot](https://via.placeholder.com/700x400?text=Streamlit+RAG+App)

---

## 📌 Example Usage

1. Launch the app:

```bash
streamlit run app.py
```

2. The app will:

   * Automatically load the PDF.
   * Display success messages when processing is complete.
   * Prompt you to ask a question.
   * Return a helpful answer based only on the content of the PDF.

---

## ❓ FAQ

**Q: Can I use another PDF?**
Yes! Modify the `uploaded_file` path in the code to use any local PDF.

**Q: Do I need GPU or heavy compute?**
No, the heavy lifting is done by Groq’s cloud-hosted model.

**Q: Is it secure?**
Keep your `groqapi` private. Never share your key publicly.

---

