import gradio as gr
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import ollama
import re
import os
import json
import pdfplumber
import camelot


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
            
            # Extract JSON objects
            json_matches = re.findall(r'\{[\s\S]*?\}', text)
            for match in json_matches:
                try:
                    json_obj = json.loads(match)
                    json_objects.append(json.dumps(json_obj, indent=4))
                except json.JSONDecodeError:
                    pass

            # Extract Code Blocks
            code_matches = re.findall(r'```[\s\S]+?```|(?:\n\s{4,}.*)+', text)
            code_blocks.extend(code_matches)
    
    # Extract tables using Camelot
    try:
        extracted_tables = camelot.read_pdf(
            pdf_path, pages="all", flavor="stream", strip_text="\n", edge_tol=500
        )
        for table in extracted_tables:
            tables.append(table.df.to_string(index=False, header=False))
    except Exception as e:
        print(f"⚠️ Error extracting tables: {e}")

    # Ensure all four return values exist
    return full_text, "\n\n".join(tables), "\n\n".join(json_objects) or "", "\n\n".join(code_blocks) or ""
from langchain.schema import Document

def process_pdf(pdf_path):
    if pdf_path is None or not pdf_path.endswith(".pdf"):
        print("Invalid PDF path")
        return None, None, None

    # Extract text, tables, JSON, and code separately
    text, table_data, json_data, code_data = extract_text_and_tables(pdf_path)

    # Combine extracted content
    full_content = f"{text}\n\nTables:\n{table_data}\n\nJSON Data:\n{json_data}\n\nCode:\n{code_data}"

    # Create a document object for chunking
    document = [Document(page_content=full_content, metadata={"source": pdf_path})]

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)
    chunks = text_splitter.split_documents(document)

    # Generate embeddings using DeepSeek-R1
    embeddings = OllamaEmbeddings(model="deepseek-r1")

    # Store vectors in ChromaDB
    """ vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory="./ice_make"
    ) """
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./ice_make",
        metadatas=[{"score": score} for score in similarity_scores]  # Store similarity scores
    )


    retriever = vectorstore.as_retriever()

    return text_splitter, vectorstore, retriever



def process_pdf_bk(pdf_path):
    if pdf_path is None or not pdf_path.endswith(".pdf"):
        print("Invalid PDF path")
        return None, None, None

    # Load the PDF file
    loader = PyMuPDFLoader(pdf_path)
    data = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=1000)
    chunks = text_splitter.split_documents(data)

    # Generate embeddings using DeepSeek-R1
    embeddings = OllamaEmbeddings(model="deepseek-r1")

    # Store vectors in ChromaDB
    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory="./ice_make"
    )
    retriever = vectorstore.as_retriever()

    return text_splitter, vectorstore, retriever



from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

def load_existing_vectorstore():
    """Load stored vector database from disk."""
    embeddings = OllamaEmbeddings(model="deepseek-r1")
    vectorstore = Chroma(persist_directory="./ice_make", embedding_function=embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

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

DATA_FILE = "./trainable_data/req_res.json"

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
    
def query_deepseek_r1(question, context):
    """Pass user query and context to DeepSeek-R1 and store response."""
    # formatted_prompt = f"Question: {question}\n\nContext: {context}"
    formatted_prompt = f"Answer the following question based on the provided context:\n\nQuestion: {question}\n\nContext: {context}\n\nPlease provide a detailed answer."

    response = ollama.chat(
        model="deepseek-r1",
        messages=[{"role": "user", "content": formatted_prompt}],
    )

    response_content = response["message"]["content"]
    final_answer = re.sub(r"<think>.*?</think>", "", response_content, flags=re.DOTALL).strip()

    # Store interaction for fine-tuning
    save_to_json(question, context, final_answer)

    return final_answer

def rag_pipeline_working(user_query):
    """RAG pipeline: Retrieve, Augment, and Generate an Answer."""
    retriever = load_existing_vectorstore()

    # Retrieve relevant document chunks
    # relevant_docs = retriever.get_relevant_documents(user_query, top_k=10) 
    relevant_docs = retriever.invoke(user_query)  # Updated method


    if not relevant_docs:
        return "No relevant information found in the database."

    # Print retrieved context for debugging
    print("\n🔍 Retrieved Context:\n" + "-"*50)
    for i, doc in enumerate(relevant_docs):
        print(f"Chunk {i+1}: {doc.page_content}\n")

    context = combine_docs(relevant_docs)

    # Query DeepSeek-R1 with the context
    answer = query_deepseek_r1(user_query, context)
    
    # print("\n🤖 Answer WITHOUT Context:")
    # answer = query_deepseek_r1(user_query, "")
    
    return answer

def rag_pipeline(user_query, max_chunks=5, min_similarity=0.75):
    """RAG pipeline: Retrieve, filter, and generate an answer."""
    retriever = load_existing_vectorstore()

    # Retrieve relevant document chunks
    relevant_docs = retriever.invoke(user_query)

    if not relevant_docs:
        return "No relevant information found in the database."

    # Rerank and filter by similarity
    relevant_docs = sorted(relevant_docs, key=lambda x: x.metadata.get("score", 1.0), reverse=True)
    relevant_docs = [doc for doc in relevant_docs if doc.metadata.get("score", 1.0) >= min_similarity]

    # Limit to top 'max_chunks'
    relevant_docs = relevant_docs[:max_chunks]

    # Clean and combine context
    context = clean_context(relevant_docs)

    # Query DeepSeek-R1 with the refined context
    answer = query_deepseek_r1(user_query, context)

    return answer

# 🎨 Gradio Interface
def chat_with_rag(user_input):
    response = rag_pipeline(user_input)
    return response
    
def get_model_info():
    model_info = ollama.show("deepseek-r1")

    # Extracting the model name safely
    modelfile_lines = model_info['modelfile'].split("\n")
    model_name = modelfile_lines[1].replace("# FROM ", "").strip() if len(modelfile_lines) > 1 else "Unknown"

    formatted_info = """
    📌 **Model Name:** {}
    
    🗂️ **Format:** {}
    📦 **Parameter Size:** {}
    🧠 **Architecture:** {}
    🖥️ **Quantization Level:** {}
    
    ⚙️ **Model Specifications:**
    - Context Length: {}
    - Attention Heads: {}
    - Feed Forward Length: {}
    - Rope Frequency Base: {}
    
    📜 **License:** {}
    """.format(
        model_name,
        model_info['details'].format,
        model_info['details'].parameter_size,
        model_info['modelinfo']['general.architecture'],
        model_info['details'].quantization_level,
        model_info['modelinfo']['qwen2.context_length'],
        model_info['modelinfo']['qwen2.attention.head_count'],
        model_info['modelinfo']['qwen2.feed_forward_length'],
        model_info['modelinfo']['qwen2.rope.freq_base'],
        model_info['license']
    ).strip()

    return formatted_info


# Create Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## RAG-Powered Chatbot with DeepSeek-R1 & ChromaDB")
    
    with gr.Row():
        input_box = gr.Textbox(label="Ask a Question", placeholder="Enter your query...")
    
    output_box = gr.Textbox(label="AI Response")
    
    submit_btn = gr.Button("Submit")
    
    submit_btn.click(chat_with_rag, inputs=[input_box], outputs=[output_box])
    
    model_info_box = gr.Textbox(label="Model Information", interactive=False, lines=15, show_label=True)
    model_btn = gr.Button("Show Model Info")

    model_btn.click(get_model_info, outputs=[model_info_box])




# Main function
def main():
    # pdf_path = "./pdfs/AR_25833_ICEMAKE_2023_2024.pdf"  # Update this with the correct file path
    # text_splitter, vectorstore, retriever = process_pdf(pdf_path)

    # if vectorstore:
    #     print("PDF processed successfully and stored in ChromaDB.")
    # else:
    #     print("Failed to process PDF.")
    
    # retriever = load_existing_vectorstore()

    # # Define a user query
    # query = "what is page number for 'Tools: Our keys to the outside world' available?"

    # # Retrieve relevant document chunks
    # relevant_docs = retriever.get_relevant_documents(query)

    # # Combine retrieved chunks into context
    # context = combine_docs(relevant_docs)

    # # Query DeepSeek-R1 model
    # answer = query_deepseek_r1(query, context)

    # print("Answer:", answer)
    
    # query = "What is DeepSeek-R1?"
    # print("\n🧠 Answer WITH Context:")
    # print(rag_pipeline(query))

    # print("\n🤖 Answer WITHOUT Context:")
    # print(query_deepseek_r1(query, ""))

    demo.launch()

if __name__ == "__main__":
    main()
