# Multimodal RAG POC — Tower Inspection Reports

A proof-of-concept multimodal RAG system that ingests tower inspection reports (text + images) and lets you ask questions via a chat interface. Answers reference both textual data and visual content from inspection photos.

## Stack

- **Framework:** LangChain (multi-vector retrieval)
- **Document Parsing:** PyMuPDF + markdown parsing
- **Vision Model:** GPT-4o (describes inspection images)
- **Embeddings:** OpenAI text-embedding-3-small
- **Vector Store:** ChromaDB (local, zero-config)
- **LLM:** GPT-4o
- **UI:** Streamlit

See [`poc/ARCHITECTURE.md`](poc/ARCHITECTURE.md) for detailed justifications.

## Quick Start

### 1. Install dependencies

```bash
cd poc
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
export OPENAI_API_KEY=sk-your-key-here
# Or create poc/.env with: OPENAI_API_KEY=sk-your-key-here
```

### 3. Ingest documents

```bash
cd poc
python ingest.py
```

This parses all reports in `sample-data/`, sends images to GPT-4o for description, embeds everything, and stores in a local ChromaDB.

#### Using your own confidential documents

```bash
# Place your PDFs in sample-data/confidential/
cp /path/to/your/*.pdf ../sample-data/confidential/

# Ingest from the confidential directory
python ingest.py --data-dir ../sample-data/confidential/
```

The system handles engineering drawings, A&E plans, structural analyses, and other technical PDFs. Pages with many small image fragments (common in CAD exports) are automatically rendered as full-page images for accurate vision model analysis.

### 4. Run the demo

```bash
cd poc
streamlit run app.py
```

### 5. Or query from CLI

```bash
cd poc
python query.py "Which towers need corrective action?"
```

## Sample Questions

- "Which towers have issues that need corrective action?"
- "What's the grounding resistance for each tower?"
- "Which tower is in Gaffney and what's its condition?"
- "What's the FCC ASR number for the tower in Kimberly, NV?"
- "Describe the foundation condition of tower A0028075"

## Project Structure

```
sample-data/          # 5 tower inspection reports (PDF + markdown) with photos
sample-data/confidential/  # Your own documents (gitignored, never committed)
poc/
  ARCHITECTURE.md     # Component choices + justifications
  requirements.txt    # Python dependencies
  config.py           # Configuration (models, paths, env vars)
  ingest.py           # Document ingestion pipeline
  query.py            # Query engine
  app.py              # Streamlit chat UI
  vectorstore/        # ChromaDB data (gitignored)
```

## Data Privacy & Confidentiality

This system is designed to work with confidential engineering documents. Here's how data is handled:

- **Local processing:** Documents are parsed locally using PyMuPDF. No documents are uploaded to any cloud service in their original form.
- **OpenAI API:** Text chunks and images are sent to the OpenAI API for embedding and vision analysis. Per [OpenAI's API data usage policy](https://openai.com/policies/api-data-usage-policies), **API data is NOT used to train models**. Your data is not retained beyond the API call.
- **Embeddings only:** The vector store contains only embeddings (numerical vectors) and text chunks — not the original documents.
- **Gitignored:** The `sample-data/confidential/` directory is gitignored to prevent accidental commits of sensitive documents.
- **Your responsibility:** Ensure your organization's data policies allow sending document content to the OpenAI API. For air-gapped environments, swap GPT-4o for a local model (e.g., LLaVA for vision, local embeddings).
