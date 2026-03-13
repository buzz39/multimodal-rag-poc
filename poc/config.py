"""Configuration for the Multimodal RAG POC."""

import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Models
VISION_MODEL = "gpt-4o"
LLM_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"

# Paths
SAMPLE_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "sample-data")
VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "vectorstore")

# Chunking
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 400

# Retrieval
TOP_K = 8
