import os
import glob
import torch
import warnings
from typing import Any, List, Optional
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import PythonCodeTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PythonLoader
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModel
import subprocess

# Suppress all deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="To copy construct from a tensor")

# Set environment variable to handle tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
        cmd = [
            "python", "-m", "mlx_lm.generate",
            "--model", self.model_name,
            "--max-tokens", str(self.max_tokens),
            "--temp", str(self.temperature),
            "--prompt", prompt
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout
        
        try:
            answer = output.split(prompt, 1)[1].strip()
            return answer
        except (IndexError, AttributeError):
            return output
    
    @property
    def _llm_type(self) -> str:
        return "mlx-phi-3"

def setup_qa_chain(vectordb, llm):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
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

    # if source_docs:
    #     output.append("\nSOURCE CODE REFERENCES:")
    #     output.append("-"*25)
        
    #     # Group documents by source file
    #     docs_by_source = {}
    #     for doc in source_docs:
    #         source = doc.metadata.get('source', 'Unknown')
    #         if source not in docs_by_source:
    #             docs_by_source[source] = []
    #         docs_by_source[source].append(doc.page_content)
        
    #     # Print each source file's content
    #     for source, contents in docs_by_source.items():
    #         output.append(f"\nFile: {source}")
    #         output.append("-"*25)
    #         # Remove duplicates while preserving order
    #         unique_contents = []
    #         seen = set()
    #         for content in contents:
    #             if content not in seen:
    #                 seen.add(content)
    #                 unique_contents.append(content)
            
    #         for content in unique_contents:
    #             output.append(content.strip())
    #             output.append("-"*25)

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
    # Load documents
    all_docs = []
    scripts_dir = os.path.join("repo-exp", "scripts")
    python_files = glob.glob(os.path.join(scripts_dir, "*.py"))
    
    for file in python_files:
        docs = PythonLoader(file).load()
        for doc in docs:
            doc.metadata["source"] = file
        all_docs.extend(docs)
    
    # Chunk documents
    text_splitter = PythonCodeTextSplitter(chunk_size=1750, chunk_overlap=100)
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
    
    # Initialize LLM
    print("Initializing language model...")
    llm = MLXLLM(
        model_name="microsoft/Phi-3-mini-4k-instruct",
        max_tokens=256,
        temperature=0.25
    )
    
    # Setup QA chain
    qa = setup_qa_chain(vectordb, llm)
    
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