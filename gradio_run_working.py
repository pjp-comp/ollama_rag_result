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

def process_pdf(pdf_path):
    if pdf_path is None or not pdf_path.endswith(".pdf"):
        print("Invalid PDF path")
        return None, None, None

    # Load the PDF file
    loader = PyMuPDFLoader(pdf_path)
    data = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)
    chunks = text_splitter.split_documents(data)

    # Generate embeddings using DeepSeek-R1
    embeddings = OllamaEmbeddings(model="deepseek-r1")

    # Store vectors in ChromaDB
    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory="./ice_make"
    )
    retriever = vectorstore.as_retriever()

    return text_splitter, vectorstore, retriever



def load_existing_vectorstore():
    """Load stored vector database from disk."""
    embeddings = OllamaEmbeddings(model="deepseek-r1")
    vectorstore = Chroma(persist_directory="./ice_make", embedding_function=embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

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

def rag_pipeline(user_query):
    """RAG pipeline: Retrieve, Augment, and Generate an Answer."""
    retriever = load_existing_vectorstore()

    # Retrieve relevant document chunks
    relevant_docs = retriever.get_relevant_documents(user_query, top_k=10) 

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
    pdf_path = "./pdfs/AR_25833_ICEMAKE_2023_2024_050920241786.pdf"  # Update this with the correct file path
    text_splitter, vectorstore, retriever = process_pdf(pdf_path)

    if vectorstore:
        print("PDF processed successfully and stored in ChromaDB.")
    else:
        print("Failed to process PDF.")
    
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
