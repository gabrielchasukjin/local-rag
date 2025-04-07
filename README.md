# MLX-Powered RAG System with Llama-3.1-8B

A Retrieval Augmented Generation (RAG) system using Apple's MLX framework with Llama-3.1-8B-Instruct model. This system is specifically optimized for Apple Silicon (M-series) processors.

## Features

- Uses Llama-3.1-8B-Instruct model with Apple's MLX framework for maximum performance on Mac
- Implements RAG pattern using Chroma as the vector database
- Optimized for Apple Silicon with Metal Performance Shaders (MPS)
- Intelligent vector database caching for faster repeat runs

## Requirements

- macOS with Apple Silicon (M1/M2/M3 series)
- Python 3.9+
- MLX and MLX-LM
- PyTorch with MPS support
- LangChain

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install langchain langchain_community torch transformers chromadb mlx mlx-lm
```

Note: MLX will be automatically installed if not present when you run the script.

## How It Works

The system uses MLX, Apple's machine learning framework, to run the Llama model natively on Apple Silicon:

1. Document indexing using the Hugging Face embedding model with MPS acceleration
2. Vector database creation with Chroma
3. Query processing with MLX-powered Llama-3.1-8B-Instruct

MLX provides significant performance benefits over standard PyTorch on Apple Silicon by leveraging Apple's Neural Engine and optimized Metal compute shaders.

## Usage

Run the main script:

```bash
python coreml_llama_rag.py
```

The script will:
1. Install MLX if needed
2. Check if a vector database already exists:
   - If it exists, ask if you want to use it or rebuild it
   - If you choose to use it, the script will skip the document loading and indexing steps
   - If you choose to rebuild it or it doesn't exist, it will prompt for a directory containing Python files
3. Create embeddings and store them in a Chroma vector database (if needed)
4. Load the Llama model with MLX
5. Start an interactive query loop where you can ask questions about the code

### Reusing the Vector Database

The system intelligently checks for an existing vector database and offers to reuse it, which:
- Saves significant time when running the script multiple times
- Allows you to quickly start querying without reprocessing documents
- Provides an option to rebuild the database when needed (if you've changed the source files)

## Performance

The system is optimized for Apple Silicon:
- MLX provides near-native performance for Llama inference
- MPS acceleration for embedding generation
- Efficient chunking and retrieval strategies

On Apple Silicon Macs, this implementation significantly outperforms standard PyTorch implementations.

## Troubleshooting

- If you encounter issues with MLX, try updating to the latest version: `pip install -U mlx mlx-lm`
- Make sure your Mac has enough memory to load the model (8GB+ recommended)
- For slower performance, try adjusting the chunk size or using a smaller model
- If you get authentication errors with Hugging Face, ensure you're logged in: `huggingface-cli login`

## License

This project is provided under the MIT License. 