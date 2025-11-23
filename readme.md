# PDF Question-Answering System (RAG)

A Retrieval-Augmented Generation (RAG) application that allows a user to upload a **ZIP file containing multiple PDFs**, processes all documents, and answers questions strictly based on their content.

The system is built using:

- **Groq Llama 3** models  
- **Local SentenceTransformer embeddings**  
- **AstraDB (Cassandra)** vector storage  
- **LangChain Classic (v1+)**  
- **Gradio Web UI**

---

## Features

- Upload a ZIP file containing one or more PDFs  
- Extracts and combines text from all documents  
- Recursive text chunking for better retrieval  
- Embeddings generated using:  
  `sentence-transformers/all-MiniLM-L6-v2`  
- Vector storage with AstraDB  
- Retrieval-based question answering using LangChain  
- Clean and simple Gradio interface for interactive Q&A

---

## Project Structure
```
Text Summarizer/
│── app.py
│── README.md
│── .env
│── requirements.txt
```

---

## How It Works

1. User uploads a ZIP file containing PDF documents  
2. The system extracts text from all PDFs  
3. Text is chunked and converted into embeddings  
4. Embeddings are stored inside AstraDB  
5. A retriever fetches the most relevant chunks  
6. Groq Llama 3 generates grounded answers using the retrieved context  

---

## Requirements

- Python 3.10+  
- AstraDB account and application token  
- Groq API key  
- All dependencies listed in `requirements.txt`

---

## Limitations

- Only ZIP uploads are supported (folder upload is not available yet)  
- PDF text extraction quality depends on PDF formatting  
- No support for DOCX, TXT, or image-based PDFs  
- No conversation memory or chat history  

---

## Planned Improvements

- Direct folder upload support  
- Better PDF preprocessing for cleaner extraction  
- Multi-model support  
- Source highlighting (show the PDF + chunk origin)  
- Chat history and follow-up question awareness  

---

## Acknowledgements

This project was inspired and supported by learning resources from:

- freeCodeCamp  
- Krish Naik (YouTube Machine Learning and NLP tutorials)

Their tutorials and open educational content greatly helped in understanding
RAG pipelines, vector databases, and modern LLM application development.

