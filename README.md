# 📄 RAG Document Q&A using OpenAI

A Retrieval-Augmented Generation (RAG) application built with **Streamlit**, **LangChain**, **OpenAI**, and **FAISS** that allows users to ask questions from PDF documents and receive context-aware answers.

---

## 🚀 Features

* Upload and process PDF documents
* Generate vector embeddings using OpenAI Embeddings
* Store document embeddings in FAISS Vector Database
* Retrieve relevant document chunks using semantic search
* Generate accurate answers using OpenAI GPT models
* View retrieved document chunks used to generate responses
* Simple and interactive Streamlit interface

---

## 🛠️ Tech Stack

* Python
* Streamlit
* LangChain
* OpenAI GPT-4o Mini
* OpenAI Embeddings
* FAISS Vector Store
* PyPDF Loader
* Dotenv

---

## 📂 Project Structure

```bash
RAG_Document_QA/
│
├── Documents/
│   ├── sample1.pdf
│   ├── sample2.pdf
│
├── app.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/RAG_Document_QA.git
cd RAG_Document_QA
```

### 2. Create Virtual Environment

#### Conda

```bash
conda create -n rag_openai python=3.10 -y
conda activate rag_openai
```

#### OR venv

```bash
python -m venv venv
```

Activate:

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root.

```env
OPENAI_API_KEY=your_openai_api_key
```

---

## 📚 Add Documents

Create a folder named:

```bash
Documents
```

Place all PDF files inside it.

Example:

```bash
Documents/
├── AI.pdf
├── ML.pdf
├── DataScience.pdf
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 🔄 Workflow

1. Application loads PDFs from the `Documents` folder.
2. PDFs are split into smaller chunks.
3. OpenAI Embeddings are generated.
4. Embeddings are stored in FAISS.
5. User asks a question.
6. Relevant document chunks are retrieved.
7. GPT-4o Mini generates an answer using retrieved context.
8. Source chunks are displayed for transparency.

---

## 📸 Application Screens

### Home Screen

* Create document embeddings
* Ask questions about your PDFs

### Results Screen

* AI-generated answer
* Response time
* Retrieved document chunks

---

## Example Questions

```text
What is machine learning?

Summarize chapter 3.

What are the key findings of the report?

Explain the conclusion section.
```

---

## 📦 Required Libraries

```text
streamlit
langchain
langchain-openai
langchain-community
faiss-cpu
pypdf
python-dotenv
openai
```

---

## 🔒 Important Notes

* Do not commit your `.env` file.
* Do not commit your virtual environment folder (`venv/`).
* Add both to `.gitignore`.

Example:

```gitignore
# Virtual Environment
venv/
.venv/

# Environment Variables
.env

# Python
__pycache__/
*.pyc

# Streamlit
.streamlit/
```

---

## Future Improvements

* PDF Upload Feature
* Chat History
* Multiple Vector Databases
* Hybrid Search
* Source Page References
* Conversation Memory
* Support for DOCX and TXT files

---

## 👨‍💻 Author

**Tarun Gangadhar**

If you found this project useful, consider giving it a ⭐ on GitHub.
