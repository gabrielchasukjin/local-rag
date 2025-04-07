#!/usr/bin/env python3

import os
import torch
import warnings
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from langchain.embeddings.base import Embeddings

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

def main():
    # Path to the persisted Chroma database
    persist_directory = "chroma_db"
    
    if not os.path.exists(persist_directory):
        print(f"Error: Database directory '{persist_directory}' not found!")
        print("You need to run chroma_script.py first to create the database.")
        return
    
    print(f"Loading existing Chroma database from {persist_directory}...")
    
    # Create the embeddings object (must match the one used to create the database)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    embeddings = CustomHuggingFaceEmbeddings(device=device)
    
    # Load the existing vector database
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    # Print database stats
    collection = vectordb._collection
    count = collection.count()
    print(f"Database contains {count} document chunks")
    
    # Initialize the language model for answering questions
    print("Loading language model...")
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
    
    # Create the QA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True
    )
    
    # Interactive question answering loop
    print("\n=== Ask questions about the code (type 'exit' to quit) ===")
    while True:
        query = input("\nEnter your question: ")
        if query.lower() == 'exit':
            break
        
        if query.lower() == 'explore':
            explore_database(vectordb)
            continue
            
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
                for i, doc in enumerate(result["source_documents"][:3]):
                    source = doc.metadata.get("source", "Unknown source")
                    if source not in sources_seen:
                        sources_seen.add(source)
                        filename = os.path.basename(source)
                        print(f"  - {filename}")
                        print(f"    Content snippet: {doc.page_content[:100]}...")
        else:
            print("\nAnswer:")
            print(str(result))

def explore_database(vectordb):
    """Explore the contents of the database"""
    print("\n=== Database Explorer ===")
    print("What would you like to do?")
    print("1. List all files in the database")
    print("2. Show sample chunks")
    print("3. Back to QA")
    
    choice = input("Enter your choice (1-3): ")
    
    if choice == "1":
        # List all unique files in the database
        results = vectordb._collection.get(include=["metadatas"])
        if results and "metadatas" in results:
            all_sources = set()
            for metadata in results["metadatas"]:
                if "source" in metadata:
                    all_sources.add(metadata["source"])
            
            print(f"\nFound {len(all_sources)} unique files in the database:")
            for i, source in enumerate(sorted(all_sources), 1):
                print(f"{i}. {os.path.basename(source)} ({source})")
    
    elif choice == "2":
        # Show sample chunks
        results = vectordb._collection.get(include=["documents", "metadatas"], limit=5)
        if results and "documents" in results and "metadatas" in results:
            print("\nSample chunks from the database:")
            for i, (doc, metadata) in enumerate(zip(results["documents"], results["metadatas"]), 1):
                source = metadata.get("source", "Unknown source")
                print(f"\n--- Chunk {i} from {os.path.basename(source)} ---")
                print(f"Content preview: {doc[:200]}...")
    
    print("\nReturning to QA mode...")

if __name__ == "__main__":
    main() 