
from langchain_community.document_loaders import JSONLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import gradio as gr
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

#sentence transformer model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

#loading chroma db
chroma_client = chromadb.PersistentClient(path="./chroma_legal_db")
collection = chroma_client.get_collection("legal_documents")

#defining function to get results from chroma db
def get_RagOp(query):
    query_emb = embed_model.encode([query])
    retrieved_data = ""
    results = collection.query(
        query_embeddings = query_emb,
        n_results = 5,
        include = ["documents", "metadatas"]
    )
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        retrieved_data += (("Article: "+str(meta['Article'])) if meta['Article'] != "none" else "") + " "
        retrieved_data += (("Title: " + meta['Title']) if meta['Title'] != "none" else "") + " "
        retrieved_data += (("Section Title: " + meta['Section_Title']) if meta['Section_Title'] != "none" else "") + " "
        retrieved_data += (("Chapter Title: " + meta['Chapter_Title']) if meta['Chapter_Title'] != "none" else "") + " "
        retrieved_data += (("Chapter: "+str(meta['Chapter'])) if meta['Chapter'] != "none" else "") + " "
        retrieved_data += (("Section: "+str(meta['Section'])) if meta['Section'] != "none" else "") + " "
        retrieved_data += (("Source: " + meta['source']) if meta['source'] != "none" else "") + " "

        retrieved_data += doc + " "
    return retrieved_data

#function to get message input for model
def get_messages(message,history):
    messages = [
        {
            "role": "system",
            "content": 
                "You are a legal advisor designed to assist users with questions related to Indian law. "
                +"You are supported by a Retrieval-Augmented Generation (RAG) system that provides relevant context from official legal documents, including: "
                +"The Constitution of India, Indian Penal Code (IPC), Code of Criminal Procedure (CrPC), Code of Civil Procedure (CPC), Indian Evidence Act (IEA), and Motor Vehicles Act (MVA). "
                +"When answering legal questions: Carefully read and rely on the retrieved context from the RAG system. Clearly cite sources (e.g., \"Section 326A of the IPC\") only if they appear in the context. "
                +"If context is insufficient or missing, say: “The provided legal context does not contain sufficient information to answer this accurately.” "
                +"Do not fabricate or guess legal provisions or insert invalid, outdated, or repealed sections. "
                +"Do not hallucinate malformed citations like `[1][2A]`, `Schedule 3[1][1]`, or unrelated acts like the 'Dangerous Drugs Act' unless they are explicitly present in the retrieved documents. "
                +"Keep your responses focused, factually grounded, and free of speculative reasoning. Avoid mixing real and fake sections. "
                +"If you receive greetings or non-legal questions (e.g., “Hi,” “How are you?”), respond naturally without referring to legal documents or the RAG system. "
                +"You are not allowed to provide legal advice on matters that fall outside Indian statutory or constitutional law unless explicitly present in the context. "
                +"Your goal is to ensure legally correct, helpful, and concise responses that build trust and clarity for the user."
            
        }
    ]
    
    messages.extend(history)
    rag_context = get_RagOp(message)
    messages.append({"role":"assistant","content":rag_context})
    messages.append({"role":"user","content":message})
    return messages

#function that takes messages and history as input and returns models response
def get_output(message,history):
    response = client.chat.completions.create(
        extra_body={},
        model=model_qwendeeps,
        messages=get_messages(message,history),
        stream=True
    )
    chunks = []
    for chunk in response:
        chunks.append(chunk.choices[0].delta.content or "")
        yield "".join(chunks)

load_dotenv()  # This loads variables from .env into os.environ

#creating client using open router api
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPEN_ROUTER_API"),
)

model_qwendeeps = "deepseek/deepseek-r1-0528-qwen3-8b:free"

#creating gradio chat interface
demo = gr.ChatInterface(
    get_output,
    type="messages",
    chatbot=gr.Chatbot(height=450),
    textbox=gr.Textbox(placeholder="Ask me a yes or no question", container=False, scale=7),
    title="LegalMind",
    description="AI that understands Indian law — powered by Retrieval-Augmented Generation (RAG). It references official legal   documents including the Constitution of India, Indian Penal Code (IPC), Code of Criminal Procedure (CrPC), Code of Civil Procedure (CPC), Indian Evidence Act (IEA), and Motor Vehicles Act (MVA).",
    theme="ocean",
)
if __name__ == "__main__":
    demo.launch()