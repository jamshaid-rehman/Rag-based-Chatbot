---
title: RAG Chatbot
emoji: 📚
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.2.0
app_file: app.py
pinned: false
---

# 📚 RAG Chatbot - Document Question Answering System

A Retrieval-Augmented Generation (RAG) chatbot that answers questions based on uploaded PDF documents using Groq LLM and semantic search.

## 🚀 Features

- ✅ Multiple PDF upload support
- ✅ Text extraction from all pages
- ✅ Semantic text chunking
- ✅ Vector similarity search for relevant content
- ✅ Groq LLM integration (llama3-8b-8192)
- ✅ Source references with page numbers
- ✅ Document preview feature
- ✅ Chat history export
- ✅ Clean Gradio interface

## 🎯 How to Use

1. **Upload Documents**: Click "Upload Files" and select PDF files
2. **Process**: Click "Process Documents" to prepare files for Q&A
3. **Preview**: (Optional) Expand "Document Preview" to see summaries
4. **Ask Questions**: Type questions in the text box and click Submit
5. **View Sources**: Answers include page numbers and source documents
6. **Export**: Download your chat history anytime

## 🛠️ Technology Stack

- **Gradio**: User interface
- **Groq API**: LLM inference (llama3-8b-8192)
- **Sentence Transformers**: Semantic embeddings
- **PyPDF2**: PDF text extraction
- **scikit-learn**: Cosine similarity calculations

## 📝 Project Info

This chatbot demonstrates Retrieval-Augmented Generation (RAG) architecture for document-based question answering.

### Enhancements Implemented:
1. Sentence transformers for semantic search
2. Source references with page numbers
3. Multi-format support (PDF + DOCX)
4. Document preview feature
5. Chat history export

## 🔒 Privacy

- Documents are processed in memory only
- No permanent storage of uploaded files
- Chat history is local to user session