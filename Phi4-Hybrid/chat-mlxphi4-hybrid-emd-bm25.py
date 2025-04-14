import os
import glob 
import torch
import warnings
from typing import Any, List, Optional
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import PythonCodeTextSplitter, RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import (
    PythonLoader, 
    TextLoader,
    JSONLoader,
    UnstructuredMarkdownLoader,
    UnstructuredFileLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    UnstructuredHTMLLoader,
    UnstructuredXMLLoader,
    UnstructuredRTFLoader,
    UnstructuredEPubLoader,
    UnstructuredODTLoader,
    UnstructuredEmailLoader,
    UnstructuredURLLoader,
    GitLoader
)
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import subprocess
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from collections import defaultdict
from langchain_core.retrievers import BaseRetriever

# Suppress all deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="To copy construct from a tensor")

# Set environment variable to handle tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define file type handlers
FILE_HANDLERS = {
    # Code files
    '.py': PythonLoader,
    '.js': UnstructuredFileLoader,
    '.jsx': UnstructuredFileLoader,
    '.ts': UnstructuredFileLoader,
    '.tsx': UnstructuredFileLoader,
    '.java': UnstructuredFileLoader,
    '.cpp': UnstructuredFileLoader,
    '.c': UnstructuredFileLoader,
    '.h': UnstructuredFileLoader,
    '.hpp': UnstructuredFileLoader,
    '.cs': UnstructuredFileLoader,
    '.go': UnstructuredFileLoader,
    '.rb': UnstructuredFileLoader,
    '.php': UnstructuredFileLoader,
    '.swift': UnstructuredFileLoader,
    '.kt': UnstructuredFileLoader,
    '.scala': UnstructuredFileLoader,
    '.rs': UnstructuredFileLoader,
    
    # Data files
    '.json': JSONLoader,
    '.csv': CSVLoader,
    '.xml': UnstructuredXMLLoader,
    '.yaml': UnstructuredFileLoader,
    '.yml': UnstructuredFileLoader,
    
    # Documentation
    '.md': UnstructuredMarkdownLoader,
    '.txt': TextLoader,
    '.rtf': UnstructuredRTFLoader,
    '.epub': UnstructuredEPubLoader,
    
    # Office documents
    '.xls': UnstructuredExcelLoader,
    '.xlsx': UnstructuredExcelLoader,
    '.ppt': UnstructuredPowerPointLoader,
    '.pptx': UnstructuredPowerPointLoader,
    '.odt': UnstructuredODTLoader,
    
    # Web content
    '.html': UnstructuredHTMLLoader,
    '.htm': UnstructuredHTMLLoader,
    '.url': UnstructuredURLLoader,
    
    # Email
    '.eml': UnstructuredEmailLoader,
    '.msg': UnstructuredEmailLoader,
    
    # Git
    '.git': GitLoader,
}

def get_file_loader(file_path: str):
    """Get the appropriate loader for a file based on its extension."""
    ext = os.path.splitext(file_path)[1].lower()
    return FILE_HANDLERS.get(ext, UnstructuredFileLoader)

def load_documents_from_directory(directory: str) -> List[Document]:
    """Recursively load documents from a directory and its subdirectories."""
    all_docs = []
    
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(directory):
        # Skip chroma_db directory
        if 'chroma_db' in dirs:
            dirs.remove('chroma_db')
        
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # Get the appropriate loader for the file
                loader_class = get_file_loader(file_path)
                loader = loader_class(file_path)
                
                # Load the document
                docs = loader.load()
                
                # Add source information to metadata
                for doc in docs:
                    doc.metadata["source"] = file_path
                    doc.metadata["file_type"] = os.path.splitext(file)[1][1:].lower()
                
                all_docs.extend(docs)
                print(f"Loaded: {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
    
    return all_docs

class CustomHuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def _get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            
        return embeddings[0].cpu().numpy()
    
    def embed_documents(self, texts):
        return [self._get_embedding(text) for text in texts]
    
    def embed_query(self, text):
        return self._get_embedding(text)

class MLXLLM(LLM):
    """LLM wrapper for MLX-powered Phi-4 model."""
    
    model_name: str = "microsoft/Phi-4-mini-instruct"
    max_tokens: int = 256
    temperature: float = 0.25
    tokenizer: Any = None
    model: Any = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Since MLX conversion is complex, let's use subprocess approach
        # which is more reliable for basic use cases
        self.tokenizer = None  # Not needed for subprocess approach
        self.model = None      # Not needed for subprocess approach
    
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
        
        # Extract the generated response
        try:
            # Assuming the response follows the prompt in output
            answer = output.split(prompt, 1)[1].strip()
            return answer
        except (IndexError, AttributeError):
            # Fallback to returning full output if parsing fails
            return output
    
    @property
    def _llm_type(self) -> str:
        return "mlx-phi-4"

class HybridRetriever:
    """Custom hybrid retriever that combines multiple retrieval methods."""
    
    def __init__(self, documents: List[Document], embedding_model: Embeddings, k: int = 4):
        self.documents = documents
        self.embedding_model = embedding_model
        self.k = k
        
        # Prepare text for BM25
        self.texts = [doc.page_content for doc in documents]
        self.tokenized_texts = [text.split() for text in self.texts]
        self.bm25 = BM25Okapi(self.tokenized_texts)
        
        # Prepare TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.texts)
        
        # Prepare embeddings
        self.embeddings = self.embedding_model.embed_documents(self.texts)
        self.embeddings = np.array(self.embeddings)
    
    def _get_bm25_scores(self, query: str) -> List[float]:
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        return scores
    
    def _get_tfidf_scores(self, query: str) -> List[float]:
        query_vec = self.tfidf_vectorizer.transform([query])
        scores = np.dot(self.tfidf_matrix, query_vec.T).toarray().flatten()
        return scores
    
    def _get_embedding_scores(self, query: str) -> List[float]:
        query_embedding = self.embedding_model.embed_query(query)
        scores = np.dot(self.embeddings, query_embedding)
        return scores
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        if not scores:
            return scores
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [0.5] * len(scores)
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        # Get scores from different methods
        bm25_scores = self._get_bm25_scores(query)
        tfidf_scores = self._get_tfidf_scores(query)
        embedding_scores = self._get_embedding_scores(query)
        
        # Ensure scores are lists before normalization
        bm25_scores_list = bm25_scores.tolist() if isinstance(bm25_scores, np.ndarray) else list(bm25_scores)
        tfidf_scores_list = tfidf_scores.tolist() # Known to be np.ndarray
        embedding_scores_list = embedding_scores.tolist() # Known to be np.ndarray
        
        # Normalize scores using the lists
        normalized_bm25 = self._normalize_scores(bm25_scores_list)
        normalized_tfidf = self._normalize_scores(tfidf_scores_list)
        normalized_embedding = self._normalize_scores(embedding_scores_list)
        
        # Combine scores using rank fusion (reciprocal rank fusion)
        combined_scores = []
        # Convert normalized lists to numpy arrays for efficient argsort
        normalized_bm25_arr = np.array(normalized_bm25)
        normalized_tfidf_arr = np.array(normalized_tfidf)
        normalized_embedding_arr = np.array(normalized_embedding)
        
        for i in range(len(self.documents)):
            # Get ranks for each method using NumPy arrays
            bm25_rank = len(normalized_bm25_arr) - np.argsort(normalized_bm25_arr).argsort()[i]
            tfidf_rank = len(normalized_tfidf_arr) - np.argsort(normalized_tfidf_arr).argsort()[i]
            embedding_rank = len(normalized_embedding_arr) - np.argsort(normalized_embedding_arr).argsort()[i]
            
            # Calculate reciprocal rank fusion score
            rrf_score = (1 / (60 + bm25_rank)) + (1 / (60 + tfidf_rank)) + (1 / (60 + embedding_rank))
            combined_scores.append(rrf_score)
        
        # Get top k documents using the list of combined scores
        # Convert combined_scores list to numpy array for argsort
        top_indices = np.argsort(np.array(combined_scores))[-self.k:][::-1]
        
        # Deduplicate results
        seen = set()
        unique_docs = []
        for idx in top_indices:
            doc = self.documents[idx]
            content = doc.page_content
            if content not in seen:
                seen.add(content)
                unique_docs.append(doc)
        
        return unique_docs
        
    # Add LangChain retriever interface method aliases
    def invoke(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)
        
    async def ainvoke(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)

def setup_qa_chain(vectordb, llm, documents: List[Document], embedding_model: Embeddings):
    # Create hybrid retriever
    hybrid_retriever = HybridRetriever(documents, embedding_model)
    
    # Wrap the retriever in a LangChain BaseRetriever class that just delegates to our custom retriever
    class RetrieverWrapper(BaseRetriever):
        retriever: Any = None # Declare the retriever field
        
        def __init__(self, retriever, **kwargs):
            super().__init__(**kwargs)
            self.retriever = retriever
            
        def _get_relevant_documents(self, query: str) -> List[Document]:
            return self.retriever.get_relevant_documents(query)
    
    # Create a RetrievalQA chain with the wrapped hybrid retriever
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=RetrieverWrapper(hybrid_retriever),
        return_source_documents=True
    )
    return qa

def clean_qa_output(qa_response):
    if not isinstance(qa_response, dict):
        return str(qa_response)
    
    # Extract the main result
    result = qa_response.get('result', '').strip()
    
    # Remove the performance metrics if present
    if "Prompt:" in result:
        result = result.split("Prompt:")[0].strip()
    
    # Extract source documents
    source_docs = qa_response.get('source_documents', [])
    
    # Format the output
    output = []

    if source_docs:
        output.append("\nSOURCE CODE REFERENCES:")
        output.append("-"*25)
        
        # Group documents by source file
        docs_by_source = {}
        for doc in source_docs:
            source = doc.metadata.get('source', 'Unknown')
            if source not in docs_by_source:
                docs_by_source[source] = []
            docs_by_source[source].append(doc.page_content)
        
        # Print each source file's content
        for source, contents in docs_by_source.items():
            output.append(f"\nFile: {source}")
            output.append("-"*25)
            # Remove duplicates while preserving order
            unique_contents = []
            seen = set()
            for content in contents:
                if content not in seen:
                    seen.add(content)
                    unique_contents.append(content)
            
            for content in unique_contents:
                output.append(content.strip())
                output.append("-"*25)

    output.append("RESPONSE:")
    output.append(result)
    
    return "\n".join(output)

def load_or_create_database():
    persist_directory = "chroma_db"
    
    # Check if database exists
    if os.path.exists(persist_directory):
        print("Loading existing database...")
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        embeddings = CustomHuggingFaceEmbeddings(device=device)
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        return vectordb
    
    print("Creating new database...")
    
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load all documents from the script directory and its subdirectories
    all_docs = load_documents_from_directory(script_dir)
    
    # Use RecursiveCharacterTextSplitter for better handling of different file types
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1750,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    texts = text_splitter.split_documents(all_docs)
    
    # Create embeddings and database
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    embeddings = CustomHuggingFaceEmbeddings(device=device)
    
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vectordb

def main():
    print("Initializing code chat system...")
    
    # Load or create database
    vectordb = load_or_create_database()
    
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load all documents
    all_docs = load_documents_from_directory(script_dir)
    
    # Initialize embeddings
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    embeddings = CustomHuggingFaceEmbeddings(device=device)
    
    # Initialize language model
    print("Initializing language model with MLX...")
    
    llm = MLXLLM()
    
    # Setup QA chain with hybrid retrieval
    qa = setup_qa_chain(vectordb, llm, all_docs, embeddings)
    
    print("\nWelcome to Code Chat! You can ask questions about the codebase.")
    print("Type 'exit' to quit.\n")
    
    while True:
        query = input("\nYour question: ").strip()
        if query.lower() == 'exit':
            break
            
        try:
            ans = qa.invoke(query)
            formatted_ans = clean_qa_output(ans)
            print(formatted_ans)
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again with a different question.")

if __name__ == "__main__":
    main() 