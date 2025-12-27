import gradio as gr
import PyPDF2
from groq import Groq
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
from datetime import datetime
import re
from docx import Document

# Initialize Groq client and embedding model
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Global variables
document_chunks = []
chunk_embeddings = None
chat_history = []
document_metadata = []

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF with page tracking"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        pages_text = []
        
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text.strip():
                pages_text.append({
                    'text': page_text,
                    'page': page_num + 1,
                    'source': os.path.basename(pdf_file)
                })
        
        return pages_text
    except Exception as e:
        return []

def extract_text_from_docx(docx_file):
    """Extract text from DOCX file"""
    try:
        doc = Document(docx_file)
        text = []
        
        for para_num, paragraph in enumerate(doc.paragraphs):
            if paragraph.text.strip():
                text.append({
                    'text': paragraph.text,
                    'page': para_num // 10 + 1,  # Approximate page
                    'source': os.path.basename(docx_file)
                })
        
        return text
    except Exception as e:
        return []

def smart_chunk_text(pages_data, chunk_size=400, overlap=50):
    """Create chunks while preserving metadata"""
    chunks_with_metadata = []
    
    for page_data in pages_data:
        text = page_data['text']
        words = text.split()
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_text = ' '.join(words[i:i + chunk_size])
            if chunk_text.strip():
                chunks_with_metadata.append({
                    'text': chunk_text,
                    'page': page_data['page'],
                    'source': page_data['source']
                })
    
    return chunks_with_metadata

def generate_document_summary(text, max_length=300):
    """Generate a quick summary of the document"""
    words = text.split()
    if len(words) <= max_length:
        return text
    
    summary = ' '.join(words[:max_length]) + "..."
    return summary

def process_documents(files):
    """Process multiple PDF/DOCX files with embeddings"""
    global document_chunks, chunk_embeddings, document_metadata
    
    if not files:
        return "⚠️ Please upload at least one file.", ""
    
    document_chunks = []
    document_metadata = []
    all_pages_data = []
    summaries = []
    
    # Extract text from all files
    for file in files:
        file_path = file.name
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            pages_data = extract_text_from_pdf(file_path)
        elif file_ext == '.docx':
            pages_data = extract_text_from_docx(file_path)
        else:
            continue
        
        if pages_data:
            all_pages_data.extend(pages_data)
            
            # Generate summary
            full_text = ' '.join([p['text'] for p in pages_data])
            summary = generate_document_summary(full_text)
            summaries.append(f"**{os.path.basename(file_path)}**\n{summary}\n")
    
    if not all_pages_data:
        return "⚠️ No text could be extracted from the files.", ""
    
    # Create chunks with metadata
    document_chunks = smart_chunk_text(all_pages_data)
    
    # Generate embeddings using sentence-transformers
    chunk_texts = [chunk['text'] for chunk in document_chunks]
    chunk_embeddings = embedding_model.encode(chunk_texts, show_progress_bar=False)
    
    summary_text = "\n---\n".join(summaries)
    status = f"✅ Processed {len(files)} file(s) into {len(document_chunks)} chunks!"
    
    return status, summary_text

def retrieve_relevant_chunks(question, top_k=3):
    """Retrieve relevant chunks using semantic similarity"""
    global document_chunks, chunk_embeddings
    
    if not document_chunks or chunk_embeddings is None:
        return []
    
    # Encode question
    question_embedding = embedding_model.encode([question])[0]
    
    # Calculate cosine similarity
    similarities = cosine_similarity([question_embedding], chunk_embeddings)[0]
    
    # Get top k indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Return relevant chunks with metadata
    relevant_chunks = [document_chunks[i] for i in top_indices]
    return relevant_chunks

def format_sources(chunks):
    """Format source references"""
    sources = []
    seen = set()
    
    for chunk in chunks:
        source_key = f"{chunk['source']} (Page {chunk['page']})"
        if source_key not in seen:
            sources.append(source_key)
            seen.add(source_key)
    
    return sources

def answer_question(question, history):
    """Generate answer with source references"""
    global chat_history
    
    if not document_chunks:
        bot_message = "⚠️ Please upload and process documents first!"
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": bot_message})
        return history
    
    if not question.strip():
        bot_message = "⚠️ Please enter a question."
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": bot_message})
        return history
    
    # Retrieve relevant context
    relevant_chunks = retrieve_relevant_chunks(question, top_k=3)
    context = "\n\n".join([chunk['text'] for chunk in relevant_chunks])
    
    # Create prompt
    prompt = f"""Based on the following context from the documents, answer the question accurately.
If the answer cannot be found in the context, say so clearly.

Context:
{context}

Question: {question}

Provide a clear and concise answer:"""
    
    try:
        # Call Groq API
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on document context. Be accurate and cite information carefully."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=1024,
        )
        
        answer = chat_completion.choices[0].message.content
        
        # Add source references
        sources = format_sources(relevant_chunks)
        sources_text = "\n\n**Sources:**\n" + "\n".join([f"- {s}" for s in sources])
        
        full_answer = answer + sources_text
        
        # Store in history
        chat_history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": question,
            "answer": full_answer
        })
        
        # Gradio 6.x format: use dictionaries with 'role' and 'content'
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": full_answer})
        return history
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": error_msg})
        return history

def export_chat_history():
    """Export chat history"""
    if not chat_history:
        return None
    
    history_text = "RAG Chatbot - Chat History\n"
    history_text += "=" * 60 + "\n\n"
    
    for entry in chat_history:
        history_text += f"[{entry['timestamp']}]\n"
        history_text += f"Q: {entry['question']}\n"
        history_text += f"A: {entry['answer']}\n"
        history_text += "-" * 60 + "\n\n"
    
    filename = f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(history_text)
    
    return filename

# Create Gradio Interface
with gr.Blocks(title="Enhanced RAG Chatbot") as demo:
    gr.Markdown(
        """
        # 📚 Enhanced RAG Chatbot
        ### Features: Semantic Search • Source References • Multi-format Support
        Upload PDF or DOCX files and ask questions!
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            file_upload = gr.File(
                label="📁 Upload Files (PDF/DOCX)",
                file_types=[".pdf", ".docx"],
                file_count="multiple"
            )
            process_btn = gr.Button("🔄 Process Documents", variant="primary", size="lg")
            status_text = gr.Textbox(
                label="Status",
                interactive=False,
                lines=2
            )
            
            with gr.Accordion("📄 Document Preview", open=False):
                preview_text = gr.Markdown()
            
            gr.Markdown("---")
            gr.Markdown("### 💾 Export Options")
            export_btn = gr.Button("Download Chat History")
            download_file = gr.File(label="Download", visible=True)
        
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=500
            )
            question_input = gr.Textbox(
                label="💬 Ask a Question",
                placeholder="What would you like to know about the documents?",
                lines=2
            )
            with gr.Row():
                submit_btn = gr.Button("🚀 Submit", variant="primary")
                clear_btn = gr.Button("🗑️ Clear Chat")
    
    gr.Markdown(
        """
        ### 🎯 Enhancements Included:
        - ✅ **Sentence Transformers** for semantic search (better than TF-IDF)
        - ✅ **Source References** with page numbers in answers
        - ✅ **DOCX Support** in addition to PDF
        - ✅ **Document Preview** before asking questions
        - ✅ **Chat History Export** feature
        
        ### 📖 How to Use:
        1. Upload PDF or DOCX files
        2. Click "Process Documents"
        3. View document previews (optional)
        4. Ask questions about the content
        5. Export conversation history anytime
        """
    )
    
    # Event handlers
    process_btn.click(
        fn=process_documents,
        inputs=[file_upload],
        outputs=[status_text, preview_text]
    )
    
    submit_btn.click(
        fn=answer_question,
        inputs=[question_input, chatbot],
        outputs=[chatbot]
    ).then(
        lambda: "",
        outputs=[question_input]
    )
    
    question_input.submit(
        fn=answer_question,
        inputs=[question_input, chatbot],
        outputs=[chatbot]
    ).then(
        lambda: "",
        outputs=[question_input]
    )
    
    clear_btn.click(
        lambda: [],
        outputs=[chatbot]
    )
    
    export_btn.click(
        fn=export_chat_history,
        outputs=[download_file]
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())