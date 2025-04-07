#!/usr/bin/env python3
"""
RAG system using Chroma DB and Llama-3.1-8B-Instruct model with MLX
Optimized for Apple Silicon (M-series) processors
"""

from langchain.vectorstores import Chroma
from langchain.text_splitter import PythonCodeTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PythonLoader
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

import os
import glob
import torch
import numpy as np
import subprocess
from typing import Any, List, Mapping, Optional, Dict, Union
import warnings
import time

from transformers import AutoTokenizer, AutoModel

# Suppress specific warnings
warnings.filterwarnings("ignore", message="To copy construct from a tensor")

class CustomHuggingFaceEmbeddings(Embeddings):
    """Custom embedding class using Hugging Face models"""
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device="mps"):
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

class MLXLLM(LLM):
    """LLM wrapper for MLX-powered Llama model."""
    
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    max_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Check if MLX is installed
        try:
            import mlx
            import mlx_lm
        except ImportError:
            print("Installing MLX and MLX-LM...")
            subprocess.run(["pip", "install", "mlx", "mlx-lm"], check=True)
            
        # Print status message
        print(f"Using MLX with model: {self.model_name}")
        print("Note: The first run may take longer as MLX downloads and processes the model")
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run MLX inference via command line."""
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        # Format the prompt for Llama-3.1-Instruct
        formatted_prompt = f"<|system|>\nYou are a helpful, respectful and honest assistant.\n<|user|>\n{prompt}\n<|assistant|>"
        
        # Prepare the MLX command
        cmd = [
            "python", "-m", "mlx_lm.generate",
            "--model", self.model_name,
            "--max-tokens", str(max_tokens),
            "--temp", str(self.temperature),
            "--top-p", str(self.top_p),
            "--prompt", formatted_prompt
        ]
        
        # Run the mlx_lm command
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        end_time = time.time()
        output = result.stdout
        
        # Extract the generated response
        try:
            # For Llama 3.1, extract content after the assistant tag
            response_parts = output.split("<|assistant|>")
            if len(response_parts) > 1:
                answer = response_parts[1].strip()
            else:
                # Alternative extraction method
                answer = output.split(formatted_prompt, 1)[1].strip() if formatted_prompt in output else output
        except (IndexError, AttributeError):
            # Fallback to returning full output if parsing fails
            answer = output
        
        print(f"Generation completed in {end_time - start_time:.2f} seconds")
        return answer
    
    @property
    def _llm_type(self) -> str:
        return "mlx-llama-3.1"

def load_python_files(directory_path):
    """Load all Python files from a directory into documents."""
    all_docs = []
    python_files = glob.glob(os.path.join(directory_path, "**", "*.py"), recursive=True)
    
    for file in python_files: 
        try:
            docs = PythonLoader(file).load()
            for doc in docs: 
                doc.metadata["source"] = file
            all_docs.extend(docs)
            print(f"Loaded {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return all_docs

def create_vector_database(documents, embeddings, persist_dir="chroma_db"):
    """Create and persist a vector database from documents."""
    # Check if the database already exists
    if os.path.exists(persist_dir) and os.path.isdir(persist_dir):
        print(f"Found existing vector database at {persist_dir}")
        # Load the existing database
        vectordb = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
        return vectordb
    
    # If not, create a new one
    # Split documents into chunks using a Python-aware splitter
    text_splitter = PythonCodeTextSplitter(chunk_size=1500, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    
    print(f"Split documents into {len(chunks)} chunks")
    
    # Create and persist the Chroma vector database
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    
    return vectordb

def setup_qa_chain(vectordb, llm):
    """Set up a retrieval QA chain."""
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True
    )
    return qa

def clean_qa_output(qa_response):
    """Clean and format QA output for display."""
    if isinstance(qa_response, dict) and 'result' in qa_response:
        return qa_response['result'].strip()
    elif isinstance(qa_response, str):
        return qa_response.strip()
    elif hasattr(qa_response, 'content'):
        return qa_response.content.strip()
    else:
        return str(qa_response)

def main():
    # Set up device for embeddings
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} for embeddings")
    
    # Initialize embedding model
    embeddings = CustomHuggingFaceEmbeddings(device=device)
    
    # Define persistent directory for vector database
    persist_directory = "chroma_llama_db"
    
    # Check if the vector database already exists
    vector_db_exists = os.path.exists(persist_directory) and os.path.isdir(persist_directory)
    
    if vector_db_exists:
        print(f"Found existing vector database at {persist_directory}")
        # Ask if user wants to rebuild the database
        rebuild = input("Do you want to rebuild the vector database? (y/n, default: n): ").lower()
        if rebuild != 'y':
            # Load existing database
            vectordb = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )
            print("Using existing vector database")
        else:
            # User wants to rebuild, get directory and create new database
            scripts_dir = input("Enter the directory path containing Python files: ")
            if not os.path.exists(scripts_dir):
                print(f"Directory {scripts_dir} does not exist.")
                return
            
            # Load documents
            all_docs = load_python_files(scripts_dir)
            print(f"Loaded {len(all_docs)} Python files")
            
            # Create vector database
            vectordb = create_vector_database(all_docs, embeddings, persist_directory)
            print(f"Vector database rebuilt and persisted to {persist_directory}")
    else:
        # No existing database, create new one
        scripts_dir = input("Enter the directory path containing Python files: ")
        if not os.path.exists(scripts_dir):
            print(f"Directory {scripts_dir} does not exist.")
            return
        
        # Load documents
        all_docs = load_python_files(scripts_dir)
        print(f"Loaded {len(all_docs)} Python files")
        
        # Create vector database
        vectordb = create_vector_database(all_docs, embeddings, persist_directory)
        print(f"Vector database created and persisted to {persist_directory}")
    
    # Initialize MLX-powered LLM
    llm = MLXLLM(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        max_tokens=512,
        temperature=0.2,
        top_p=0.9
    )
    
    # Set up QA chain
    qa = setup_qa_chain(vectordb, llm)
    
    # Interactive query loop
    while True:
        query = input("\nEnter your question (or 'exit' to quit): ")
        if query.lower() in ["exit", "quit", "q"]:
            break
            
        print("Generating answer...")
        start_time = time.time()
        result = qa.invoke(query)
        end_time = time.time()
        
        formatted_answer = clean_qa_output(result)
        print(f"\nAnswer: {formatted_answer}")
        print(f"Query completed in {end_time - start_time:.2f} seconds")
        
        # Print sources
        if isinstance(result, dict) and 'source_documents' in result:
            print("\nSources:")
            for i, doc in enumerate(result['source_documents'][:3]):  # Show top 3 sources
                print(f"Source {i+1}: {doc.metadata['source']}")

if __name__ == "__main__":
    main() 