#!/usr/bin/env python3

from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PythonLoader, JSONLoader
from langchain.schema import Document
import os
import glob

# Import necessary libraries for custom embeddings
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline
from langchain.embeddings.base import Embeddings
from langchain_huggingface import HuggingFacePipeline
import warnings

# Suppress specific warning
warnings.filterwarnings("ignore", message="To copy construct from a tensor")

class CustomHuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"):
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device
        self.model.to(device)
        # Set to evaluation mode
        self.model.eval()
    
    def _get_embedding(self, text):
        # Tokenize input text
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean pooling - take average of all token embeddings
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            
        # Convert to numpy array and return
        return embeddings[0].cpu().numpy()
    
    def embed_documents(self, texts):
        """Generate embeddings for a list of documents."""
        return [self._get_embedding(text) for text in texts]
    
    def embed_query(self, text):
        """Generate embedding for a query."""
        return self._get_embedding(text)

def clean_qa_output(qa_response):
    # If the response is a dictionary with a 'result' key
    if isinstance(qa_response, dict) and 'result' in qa_response:
        return qa_response['result'].strip()
    # If the response is a string
    elif isinstance(qa_response, str):
        return qa_response.strip()
    # If it's some other type of object with content
    elif hasattr(qa_response, 'content'):
        return qa_response.content.strip()
    else:
        return str(qa_response)

def load_python_files(directory):
    """Load all Python files from the specified directory"""
    all_docs = []
    python_files = glob.glob(os.path.join(directory, "*.py"))
    
    for file_path in python_files:
        try:
            print(f"Loading Python file: {file_path}")
            docs = PythonLoader(file_path).load()
            
            # Enhance metadata to include full file path
            for doc in docs:
                doc.metadata["source"] = file_path
                doc.metadata["file_type"] = "python"
            
            all_docs.extend(docs)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return all_docs

def load_json_files(directory):
    """Load all JSON files from the specified directory"""
    all_docs = []
    json_files = glob.glob(os.path.join(directory, "*.json"))
    
    for file_path in json_files:
        try:
            print(f"Loading JSON file: {file_path}")
            # Extract data using JSONLoader with a simple jq-like extraction
            # The '.' extracts the whole JSON document
            loader = JSONLoader(
                file_path=file_path,
                jq_schema='.',
                text_content=False
            )
            
            try:
                docs = loader.load()
                
                # If JSONLoader didn't preserve the source, add it manually
                for doc in docs:
                    doc.metadata["source"] = file_path
                    doc.metadata["file_type"] = "json"
                
                all_docs.extend(docs)
            except Exception as json_error:
                print(f"Error parsing JSON, falling back to raw content: {json_error}")
                # Fallback: If JSONLoader fails, just load the raw content
                with open(file_path, 'r') as f:
                    content = f.read()
                    doc = Document(
                        page_content=content,
                        metadata={"source": file_path, "file_type": "json"}
                    )
                    all_docs.append(doc)
                    
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return all_docs

def main():
    print("Loading documents...")
    # Load Python files from repo-exp/scripts directory
    scripts_dir = os.path.join("repo-exp", "scripts")
    python_docs = load_python_files(scripts_dir)
    print(f"Loaded {len(python_docs)} Python documents")
    
    # Load JSON files from repo-exp/credentials directory
    credentials_dir = os.path.join("repo-exp", "credentials")
    json_docs = load_json_files(credentials_dir)
    print(f"Loaded {len(json_docs)} JSON documents")
    
    # Combine all documents
    all_docs = python_docs + json_docs
    print(f"Total documents loaded: {len(all_docs)}")
    
    if not all_docs:
        print("No documents were loaded. Exiting.")
        return
    
    print("Splitting documents...")
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(all_docs)
    print(f"Created {len(texts)} text chunks")
    
    # Print a sample of the chunks with their metadata to verify source tracking
    if texts:
        sample = texts[0]
        print("\nSample chunk metadata:")
        print(f"Source: {sample.metadata.get('source', 'Unknown')}")
        print(f"File type: {sample.metadata.get('file_type', 'Unknown')}")
        print(f"Content preview: {sample.page_content[:100]}...\n")
    
    print("Initializing embedding model...")
    # Create the embeddings object
    # Detect if MPS (Apple Silicon) is available, otherwise use CPU
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device set to use {device}")
    embeddings = CustomHuggingFaceEmbeddings(device=device)
    
    print("Creating vector database...")
    # Create Chroma vector database
    persist_directory = "chroma_db"
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist()
    print(f"Vector database created and persisted to {persist_directory}")
    
    print("Loading language model...")
    # Initialize the language model
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.25,
        repetition_penalty=1.2
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    
    print("Creating QA chain...")
    # Create the QA chain with source document retrieval
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True  # Return source documents with answers
    )
    
    # Interactive question answering loop
    print("\n=== Ask questions about the code (type 'exit' to quit) ===")
    while True:
        query = input("\nEnter your question: ")
        if query.lower() == 'exit':
            break
            
        print("Processing query...")
        result = qa.invoke(query)
        
        # Extract and format the answer
        if isinstance(result, dict) and "result" in result:
            answer = result["result"]
            print("\nAnswer:")
            print(answer)
            
            # Display source documents if available
            if "source_documents" in result:
                print("\nSources:")
                sources_seen = set()
                for i, doc in enumerate(result["source_documents"][:3]):  # Show top 3 sources
                    source = doc.metadata.get("source", "Unknown source")
                    if source not in sources_seen:
                        sources_seen.add(source)
                        # Extract and display just the filename, not the full path
                        filename = os.path.basename(source)
                        print(f"  - {filename}")
        else:
            formatted_ans = clean_qa_output(result)
            print("\nAnswer:")
            print(formatted_ans)

if __name__ == "__main__":
    main() 