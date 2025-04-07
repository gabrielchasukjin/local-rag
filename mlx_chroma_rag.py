#!/usr/bin/env python3
"""
MLX-accelerated RAG system using Chroma DB and Phi-3 model
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
from typing import Any, List, Mapping, Optional
import warnings

from transformers import AutoTokenizer, AutoModel

# Suppress specific warning
warnings.filterwarnings("ignore", message="To copy construct from a tensor")

class CustomHuggingFaceEmbeddings(Embeddings):
    """Custom embedding class using Hugging Face models"""
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

class MLXLLM(LLM):
    """LLM wrapper for MLX-powered Phi-3 model."""
    
    model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    max_tokens: int = 256
    temperature: float = 0.25
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run MLX inference via command line."""
        cmd = [
            "python", "-m", "mlx_lm.generate",
            "--model", self.model_name,
            "--max-tokens", str(self.max_tokens),
            "--temp", str(self.temperature),
            "--prompt", prompt
        ]
        
        # Run the mlx_lm command
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout
        
        # Extract the generated response (adjust based on actual mlx_lm output format)
        try:
            # The output format might depend on mlx_lm, this is a basic approach
            # that assumes the response follows the prompt
            answer = output.split(prompt, 1)[1].strip()
            return answer
        except (IndexError, AttributeError):
            # Fallback to returning full output if parsing fails
            return output
    
    @property
    def _llm_type(self) -> str:
        return "mlx-phi-3"

def load_python_files(directory_path):
    """Load all Python files from a directory into documents."""
    all_docs = []
    python_files = glob.glob(os.path.join(directory_path, "*.py"))
    
    for file in python_files: 
        docs = PythonLoader(file).load()
        for doc in docs: 
            doc.metadata["source"] = file
        all_docs.extend(docs)
    
    return all_docs

def create_vector_database(documents, embeddings, persist_dir="chroma_db"):
    """Create and persist a vector database from documents."""
    # Split documents into chunks using a Python-aware splitter
    text_splitter = PythonCodeTextSplitter(chunk_size=1750, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    
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
        retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
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
    # Install MLX if not already installed
    try:
        import mlx
        import mlx_lm
    except ImportError:
        print("Installing MLX and MLX-LM...")
        subprocess.run(["pip", "install", "mlx", "mlx-lm"], check=True)
    
    # Set up device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device set to use {device}")
    
    # Initialize embedding model
    embeddings = CustomHuggingFaceEmbeddings(device=device)
    
    # Specify the directory containing Python files
    scripts_dir = os.path.join("repo-exp", "scripts")
    
    # Load documents
    all_docs = load_python_files(scripts_dir)
    print(f"Loaded {len(all_docs)} Python files")
    
    # Create vector database
    persist_directory = "chroma_db"
    vectordb = create_vector_database(all_docs, embeddings, persist_directory)
    print(f"Vector database created and persisted to {persist_directory}")
    
    # Initialize MLX-powered LLM
    llm = MLXLLM(
        model_name="microsoft/Phi-3-mini-4k-instruct",
        max_tokens=256,
        temperature=0.25
    )
    
    # Set up QA chain
    qa = setup_qa_chain(vectordb, llm)
    
    # Interactive query loop
    while True:
        query = input("\nEnter your question (or 'exit' to quit): ")
        if query.lower() in ["exit", "quit", "q"]:
            break
            
        print("Generating answer...")
        result = qa.invoke(query)
        formatted_answer = clean_qa_output(result)
        print(f"\nAnswer: {formatted_answer}")
        
        # Optionally print sources
        if isinstance(result, dict) and 'source_documents' in result:
            print("\nSources:")
            for i, doc in enumerate(result['source_documents'][:2]):  # Show top 2 sources
                print(f"Source {i+1}: {doc.metadata['source']}")

if __name__ == "__main__":
    main() 