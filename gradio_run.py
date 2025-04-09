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
import numpy as np  # ✅ Add this import at the top


# Directory to store processed PDFs
VECTOR_STORAGE_DIR = "./vector_storage"
DATA_FILE = "./trainable_data/req_res.json"
os.makedirs(VECTOR_STORAGE_DIR, exist_ok=True)


# Ensure storage directory exists
# if not os.path.exists(VECTOR_STORAGE_DIR):
#     os.makedirs(VECTOR_STORAGE_DIR)

JSON_FOLDER_PATH = "./pdfs/json_folder/"


# ✅ Load JSON files once
def load_json_files(json_dir="./pdf/json_folder/"):
    """Load all JSON files from the given folder into a dictionary."""
    json_data = {}
    if not os.path.exists(json_dir):
        print(f"⚠️ JSON folder not found: {json_dir}")
        return json_data

    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(json_dir, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    json_data[filename] = json.load(f)
            except json.JSONDecodeError as e:
                print(f"⚠️ Skipping invalid JSON file: {filename}, Error: {e}")
    
    return json_data



# Generate a unique hash for each PDF (to check if already processed)
def generate_pdf_hash(pdf_path):
    hasher = hashlib.md5()
    with open(pdf_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


import json
import re
def extract_text_and_tables(pdf_path):
    """Extract text, tables, and JSON placeholders from a PDF."""
    full_text = ""
    tables = []
    json_objects = []
    code_blocks = []

    # ✅ Load all available JSON files into memory
    json_lookup = load_json_files()

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""

            # ✅ Replace {{filename.json}} placeholders with actual JSON content
            matches = re.findall(r"\{\{(.*?)\}\}", text)  # Capture placeholders
            for match in matches:
                json_filename = f"{match.strip()}.json"  # Ensure proper filename format
                if json_filename in json_lookup:
                    json_content = json.dumps(json_lookup[json_filename], indent=4)
                    text = text.replace(f"{{{{{match}}}}}", json_content)  # Correct replacement
                else:
                    print(f"⚠️ JSON file not found for placeholder: {json_filename}")

            full_text += text + "\n"

            # ✅ Extract JSON blocks directly from PDF
            json_matches = re.findall(r'\{[\s\S]*?\}', text)
            for match in json_matches:
                try:
                    cleaned_json = re.sub(r'[\x00-\x1F\x7F]', '', match)  # Remove control characters
                    json_obj = json.loads(cleaned_json)
                    json_objects.append(json.dumps(json_obj, indent=4))
                except json.JSONDecodeError as e:
                    print(f"⚠️ Skipping invalid JSON block on page {page_num}: {e}")
                    print(f"🔎 JSON Content: {match}")  # Debugging output to inspect problematic JSON

            # ✅ Extract Code Blocks
            code_matches = re.findall(r'```[\s\S]+?```|(?:\n\s{4,}.*)+', text)
            code_blocks.extend(code_matches)

    return full_text, "\n\n".join(tables), "\n\n".join(json_objects), "\n\n".join(code_blocks)


def extract_text_and_tables_bk(pdf_path):
    """Extracts text, structured tables, JSON objects, and code snippets from a PDF."""
    full_text = ""
    tables = []
    json_objects = []
    code_blocks = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            full_text += text + "\n"

            # ✅ Extract JSON objects
            json_matches = re.findall(r'\{[\s\S]*?\}', text)  # Detect JSON blocks
            for match in json_matches:
                try:
                    json_obj = json.loads(match)  # Validate JSON
                    json_objects.append(json.dumps(json_obj, indent=4))  # Store formatted JSON
                except json.JSONDecodeError:
                    pass  # Ignore invalid JSON

            # ✅ Extract Code Blocks (Markdown-style ` ``` ` or indented code)
            code_matches = re.findall(r'```[\s\S]+?```|(?:\n\s{4,}.*)+', text)
            code_blocks.extend(code_matches)

    # ✅ Extract tables using Camelot
    try:
        extracted_tables = camelot.read_pdf(
            pdf_path, pages="all", flavor="stream", strip_text="\n", edge_tol=500
        )
        for table in extracted_tables:
            tables.append(table.df.to_string(index=False, header=False))
    except Exception as e:
        print(f"⚠️ Error extracting tables: {e}")

    return (
        full_text or "",  # Extracted text content
        "\n\n".join(tables) or "",  # Extracted tables
        "\n\n".join(json_objects) or "",  # Extracted JSON data
        "\n\n".join(code_blocks) or "",  # Extracted code blocks
    )


def get_pdf_hash(pdf_path):
    """Generate a hash for the PDF file to track changes."""
    hasher = hashlib.md5()
    with open(pdf_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

import numpy as np
import faiss
import os

import numpy as np
import faiss

def process_pdf(pdf_path):
    """Process the PDF only if the vector store does not exist."""
    
    pdf_hash = get_pdf_hash(pdf_path)
    index_path = os.path.join(VECTOR_STORAGE_DIR, f"{pdf_hash}.faiss")

    # ✅ Avoid redundant processing if FAISS index exists
    if os.path.exists(index_path):
        print(f"✅ Loading vector store from {index_path}")
        return load_existing_vectorstore(pdf_path)

    print(f"⚠️ No vector store found. Processing {pdf_path}...")

    text, table_data, json_data, code_data = extract_text_and_tables(pdf_path)
    full_content = f"{text}\n\nTables:\n{table_data}\n\nJSON Data:\n{json_data}\n\nCode:\n{code_data}"

    document = [Document(page_content=full_content, metadata={"source": pdf_path})]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_documents(document)

    embeddings = OllamaEmbeddings(model="deepseek-r1")

    # ✅ Dynamically fetch the embedding dimension
    sample_vector = embeddings.embed_query("test sentence")
    dimension = len(sample_vector)  

    # ✅ Ensure FAISS is created with the correct dimensions
    # index = faiss.IndexFlatL2(dimension)

    index = faiss.IndexFlatL2(dimension)
    vectors = [embeddings.embed_query(chunk.page_content) for chunk in chunks]
    index.add(np.array(vectors, dtype=np.float32))

    faiss.write_index(index, index_path)
    print(f"✅ Vector store saved at {index_path}")
    return text_splitter, index, None  # Returning None for retriever since FAISS works differently


def process_pdf_bk(pdf_path):
    """Process PDF and store vectors in FAISS."""
    pdf_hash = generate_pdf_hash(pdf_path)
    index_path = os.path.join(VECTOR_STORAGE_DIR, f"{pdf_hash}.faiss")

    # Extract text and tables
    text, table_data, json_data, code_data = extract_text_and_tables(pdf_path)

    print("📄 Extracted Text:\n", text)
    print("\n📊 Extracted Tables:\n", table_data)
    print("\n📜 Extracted JSON Data:\n", json_data)
    print("\n💻 Extracted Code Blocks:\n", code_data)
    
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
    """Load FAISS vector store if it exists."""
    index_path = os.path.join(VECTOR_STORAGE_DIR, f"{get_pdf_hash(pdf_path)}.faiss")
    
    if not os.path.exists(index_path):
        print(f"⚠️ FAISS index not found at {index_path}. Reprocessing PDF...")
        return process_pdf(pdf_path)  # Regenerate if missing

    print(f"✅ Loading FAISS index from {index_path}")
    index = faiss.read_index(index_path)
    
    # Ensure the FAISS object has a 'search' method
    if not hasattr(index, "search"):
        print("⚠️ FAISS index is not correctly loaded. Missing search method.")
        return None  # Ensure we do not proceed with an invalid index
    
    return index


def load_existing_vectorstore_bk(pdf_path):
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
        if not hasattr(doc, "page_content"):
            print(f"⚠️ Skipping invalid document: {doc}")  # Debugging: Print invalid docs
            continue

        chunk = doc.page_content.strip()
        if chunk not in seen_chunks:
            seen_chunks.add(chunk)
            cleaned_docs.append(chunk)

    return "\n\n".join(cleaned_docs)



def combine_docs(docs):
    """Combine document chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)

def query_deepseek_r1(question, context):
    """Pass user query and context to DeepSeek-R1 and enforce JSON response."""
    
    formatted_prompt = f"""
    Answer the following question based on the provided context.
    Ensure the response is a valid JSON object:
    
    Question: {question}
    Context: {context}
    """
    
    response = ollama.chat(
        model="deepseek-r1",
        messages=[{"role": "user", "content": formatted_prompt}],
    )

    response_content = response["message"]["content"].strip()
    
    # ✅ Ensure the response is valid JSON before saving
    try:
        json_response = json.loads(response_content)  # Validate JSON
        save_to_json(question, context, json.dumps(json_response, indent=4))
        return json.dumps(json_response, indent=4)  # Return formatted JSON
    except json.JSONDecodeError:
        print("⚠️ Warning: Model did not return valid JSON!")
        return '{"error": "Invalid JSON response from model"}'




def query_deepseek_r1_working(question, context):
    """Pass user query and context to DeepSeek-R1"""


    
    
    prompt_style = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

        ### Instruction:
        You are a highly skilled, creative, and experienced presentation designer with many years of expertise, specializing in creating visually captivating slides. You have helped startups secure investments and have been part of the journey toward achieving 'Unicorn' status.
        Your Task which is to generate a JSON code. The final JSON object (or an array containing a single slide in the slides array) must be valid, error-free, and formatted exactly as required for direct conversion to HTML. It must include updated styling, positions. If you need any clarification on specific code snippets or instructions, please ask.
        Re-validate the layout of the slide, if the design of a slide could be better which can lead to more user satisfaction then go for it. When you have the best possible JSON code, only then return it to the user.

        Please generate a json object as per given instruction in context. 

        ### Question:
        {question}

        \n\n
        Context: {context}\n\n please generate json and give me output.
        \n\n

        ### Response:
        # """.strip()

    question = f"{prompt_style}"
    
    # formatted_prompt = f"Answer the following question based on the provided context:\n\nQuestion: {question}\n\nContext: {context}\n\nPlease provide a detailed answer."
    formatted_prompt = f"Generate a like a sample json object only for given topic and based on the provided context:\n\Topic: {question}\n\nContext: {context}."
    
    response = ollama.chat(
        model="deepseek-r1",
        messages=[{"role": "user", "content": formatted_prompt}],
    )

    response_content = response["message"]["content"]
    final_answer = re.sub(r"<think>.*?</think>", "", response_content, flags=re.DOTALL).strip()

    # ✅ Store interaction in JSON for fine-tuning
    save_to_json(question, context, final_answer)
    
    return final_answer

import numpy as np

import numpy as np

def retrieve_faiss_documents(index, query, top_k=5):
    """Retrieve top-k similar documents from FAISS index."""
    try:
        if index is None or not hasattr(index, "search"):
            raise ValueError("FAISS index is not correctly loaded. Missing search method.")

        embeddings = OllamaEmbeddings(model="deepseek-r1")
        query_vector = np.array([embeddings.embed_query(query)], dtype=np.float32)

        distances, indices = index.search(query_vector, top_k)
        # return indices.tolist()  # Convert to a list for processing
        retrieved_docs = []
        for idx in indices[0]:  # Convert FAISS indices to actual document text
            if 0 <= idx < len(chunks):
                retrieved_docs.append(chunks[idx])
            else:
                print(f"⚠️ Skipping invalid document index: {idx}")

        return retrieved_docs  # Return actual document texts

    except Exception as e:
        print(f"⚠️ Error retrieving FAISS documents: {e}")
        return []


def rag_pipeline(user_query, pdf_path):
    """Optimized RAG pipeline: Retrieve, clean, and generate a response."""
    
    index = load_existing_vectorstore(pdf_path)  # ✅ Ensure we get only the FAISS index

    # Retrieve top-k relevant documents
    relevant_indices = retrieve_faiss_documents(index, user_query)

    if not relevant_indices:
        print("⚠️ No valid relevant documents found.")
        return "No relevant information found in the database."

    # Clean context before passing to LLM
    cleaned_context = clean_context(relevant_indices)

    # Generate response
    return query_deepseek_r1(user_query, cleaned_context)


 
def rag_pipeline_bk(user_query, pdf_path, max_chunks=5):
    """Optimized RAG pipeline: Retrieve, clean, and generate a response."""
    _, _, retriever = load_existing_vectorstore(pdf_path)
    
    user_query = f"""check how web based application works and check all formats and rules and regulation for set of fields for slides. {user_query}"""
    relevant_docs = retriever.invoke(user_query)  
    
    
    # if not relevant_docs:
    #     return "No relevant information found in the database."
    if not relevant_docs:
        return json.dumps({"answer": "No relevant information found.", "metadata": {"source": "None", "confidence": "Low"}}, indent=4)


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
    return rag_pipeline(user_input, "./pdfs/sample_ppt_instructions.pdf")


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
