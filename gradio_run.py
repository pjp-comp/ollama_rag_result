import gradio as gr
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document
import ollama
import re
import os
import json
import pdfplumber
import camelot
import asyncio
import hashlib
import pickle
import faiss
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi

# Directory to store processed PDFs
VECTOR_STORAGE_DIR = "./vector_storage"
DATA_FILE = "./trainable_data/req_res.json"

# Ensure storage directory exists
if not os.path.exists(VECTOR_STORAGE_DIR):
    os.makedirs(VECTOR_STORAGE_DIR)

# Generate a unique hash for each PDF (to check if already processed)
def generate_pdf_hash(pdf_path):
    hasher = hashlib.md5()
    with open(pdf_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def extract_text_and_tables(pdf_path):
    """Extracts text, structured tables, JSON objects, and code snippets from a PDF."""
    full_text = ""
    tables = []
    json_objects = []
    code_blocks = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            full_text += text + "\n"

    # Extract tables using Camelot
    try:
        extracted_tables = camelot.read_pdf(
            pdf_path, pages="all", flavor="stream", strip_text="\n", edge_tol=500
        )
        for table in extracted_tables:
            tables.append(table.df.to_string(index=False, header=False))
    except Exception as e:
        print(f"⚠️ Error extracting tables: {e}")

    return (
        full_text or "",  # Text content
        "\n\n".join(tables) or "",  # Tables
        "\n\n".join(json_objects) or "",  # JSON data
        "\n\n".join(code_blocks) or "",  # Code blocks
    )


def process_pdf(pdf_path):
    """Process PDF and store vectors in FAISS."""
    pdf_hash = generate_pdf_hash(pdf_path)
    index_path = os.path.join(VECTOR_STORAGE_DIR, f"{pdf_hash}.faiss")

    # Extract text and tables
    text, table_data, json_data, code_data = extract_text_and_tables(pdf_path)
    full_content = f"{text}\n\nTables:\n{table_data}\n\nJSON Data:\n{json_data}\n\nCode:\n{code_data}"

    # Create document chunks
    document = [Document(page_content=full_content, metadata={"source": pdf_path})]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_documents(document)

    # Generate embeddings and store in FAISS
    embeddings = OllamaEmbeddings(model="deepseek-r1")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 🔥 Save FAISS index
    vectorstore.save_local(index_path)
    print(f"✅ FAISS index saved at {index_path}")

    return None, vectorstore, vectorstore.as_retriever()


def load_existing_vectorstore(pdf_path):
    """Load FAISS vector store from disk and return retriever."""
    pdf_hash = generate_pdf_hash(pdf_path)
    index_path = os.path.join(VECTOR_STORAGE_DIR, f"{pdf_hash}.faiss")

    if not os.path.exists(index_path):  # 🔥 Check if the index file exists
        print(f"⚠️ FAISS index not found at {index_path}. Reprocessing PDF...")
        return process_pdf(pdf_path)  # ✅ Reprocess if missing

    embeddings = OllamaEmbeddings(model="deepseek-r1")
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    
    return None, vectorstore, vectorstore.as_retriever()  # ✅ Ensure correct tuple format


def save_to_json(question, context, response):
    """Save query-response pairs to a JSON file for fine-tuning."""
    data_entry = {
        "question": question,
        "context": context,
        "response": response
    }

    # Load existing data if available
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # Append new entry
    data.append(data_entry)

    # Save updated data back to file
    with open(DATA_FILE, "w") as file:
        json.dump(data, file, indent=4)

    print(f"✅ Saved interaction to {DATA_FILE}")
    
def clean_context(docs):
    """Clean retrieved documents to remove duplicates and unnecessary whitespace."""
    seen_chunks = set()
    cleaned_docs = []
    
    for doc in docs:
        chunk = doc.page_content.strip()
        if chunk not in seen_chunks:  # Avoid duplicates
            seen_chunks.add(chunk)
            cleaned_docs.append(chunk)
    
    return "\n\n".join(cleaned_docs)


def combine_docs(docs):
    """Combine document chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)

def query_deepseek_r1(question, context):
    """Pass user query and context to DeepSeek-R1"""
    formatted_prompt = f"Answer the following question based on the provided context:\n\nQuestion: {question}\n\nContext: {context}\n\nPlease provide a detailed answer."

    response = ollama.chat(
        model="deepseek-r1",
        messages=[{"role": "user", "content": formatted_prompt}],
    )

    response_content = response["message"]["content"]
    final_answer = re.sub(r"<think>.*?</think>", "", response_content, flags=re.DOTALL).strip()

    # ✅ Store interaction in JSON for fine-tuning
    save_to_json(question, context, final_answer)
    
    return final_answer

def rag_pipeline(user_query, pdf_path, max_chunks=5):
    """Optimized RAG pipeline: Retrieve, clean, and generate a response."""
    _, _, retriever = load_existing_vectorstore(pdf_path)
    relevant_docs = retriever.invoke(user_query)  
    
    print(f"✅ relavent docs : {relevant_docs}")
    
    if not relevant_docs:
        return "No relevant information found in the database."

    # BM25 Reranking for better retrieval
    tokenized_docs = [doc.page_content.split() for doc in relevant_docs]
    bm25 = BM25Okapi(tokenized_docs)
    ranked_docs = sorted(relevant_docs, key=lambda x: bm25.get_scores([user_query])[0], reverse=True)

    # Limit top N chunks
    ranked_docs = ranked_docs[:max_chunks]
    
    

    # ✅ Clean and combine context before querying the model
    context = clean_context(ranked_docs)

    return query_deepseek_r1(user_query, context)

def chat_with_rag(user_input):
    """Chatbot UI interaction"""
    return rag_pipeline(user_input, "./pdfs/test.pdf")


# 🌟 **ASYNC Gradio UI for Faster Responses**
async def chat_with_rag_async(user_input):
    return await asyncio.to_thread(chat_with_rag, user_input)


# 🌟 **Gradio UI**
with gr.Blocks() as demo:
    gr.Markdown("## 🚀 RAG-Powered Chatbot with DeepSeek-R1 & FAISS")

    with gr.Row():
        input_box = gr.Textbox(label="Ask a Question", placeholder="Enter your query...")

    output_box = gr.Textbox(label="AI Response")

    submit_btn = gr.Button("Submit")
    submit_btn.click(chat_with_rag_async, inputs=[input_box], outputs=[output_box])

# **Launch Gradio**
if __name__ == "__main__":
    demo.launch()
