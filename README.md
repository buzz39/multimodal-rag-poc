<div align="center">

# 🔍 Multimodal RAG — Tower Inspection Reports

**AI-powered search across text, images, and diagrams in engineering documents**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?logo=openai&logoColor=white)](https://openai.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-green)](https://www.trychroma.com/)

A proof-of-concept multimodal **Retrieval-Augmented Generation** system that ingests tower inspection reports — including text, tables, photos, and technical diagrams — and enables natural-language queries through a conversational interface. Every answer is grounded in source documents with full attribution.

</div>

---

## 📑 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Configuration](#%EF%B8%8F-configuration)
- [Project Structure](#-project-structure)
- [Sample Questions](#-sample-questions)
- [Data Privacy & Security](#-data-privacy--security)
- [Scaling Roadmap](#-scaling-roadmap)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Multimodal Understanding** | Searches across text *and* images — inspection photos, radiation patterns, elevation drawings, and technical diagrams are all queryable |
| **Source Attribution** | Every answer cites the exact document, page, and image it was derived from — full auditability |
| **Vision-Powered Ingestion** | GPT-4o analyzes each image and generates rich, searchable descriptions automatically |
| **Engineering Document Support** | Handles CAD exports, A&E drawings, and complex PDFs with fragmented image layouts |
| **Conversational Chat UI** | Streamlit-based chat interface with message history and example queries |
| **CLI Access** | Query directly from the command line for scripting and automation |
| **Zero-Config Vector Store** | ChromaDB runs locally with no database server — just `pip install` and go |
| **Confidential Document Handling** | Dedicated gitignored directory for sensitive documents that never get committed |

---

## 🏗 Architecture

```
                        ┌─────────────────────────────────────────┐
                        │            INGESTION PIPELINE           │
                        └─────────────────────────────────────────┘

  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │  PDF / MD    │────▶│   PyMuPDF    │────▶│   GPT-4o     │────▶│   OpenAI     │
  │  Documents   │     │   Parser     │     │   Vision     │     │  Embeddings  │
  └──────────────┘     └──────┬───────┘     │  (images)    │     └──────┬───────┘
                              │             └──────────────┘            │
                              │  text chunks                           │  vectors
                              │                                        │
                              ▼                                        ▼
                        ┌─────────────────────────────────────────────────┐
                        │              ChromaDB Vector Store              │
                        │         (local, persistent, zero-config)        │
                        └──────────────────────┬──────────────────────────┘
                                               │
                        ┌──────────────────────────────────────────┐
                        │             QUERY PIPELINE               │
                        └──────────────────────────────────────────┘

                        ┌──────────────┐            ┌──────────────┐
                        │  User Query  │───────────▶│   Retrieve   │
                        │  (chat/CLI)  │            │  Top-K Chunks │
                        └──────────────┘            └──────┬───────┘
                                                           │
                                                           ▼
                                                   ┌──────────────┐
                                                   │   GPT-4o     │
                                                   │  (generate   │
                                                   │   answer +   │
                                                   │   sources)   │
                                                   └──────────────┘
```

Each stored chunk includes metadata — source file, page number, content type (text/image), and image path — enabling precise source attribution in every response.

> 📖 See [`poc/ARCHITECTURE.md`](poc/ARCHITECTURE.md) for detailed component decisions, trade-off analysis, and scaling strategy.

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **OpenAI API key** — [Get one here](https://platform.openai.com/api-keys)

### 1. Clone & install

```bash
git clone https://github.com/buzz39/multimodal-rag-poc.git
cd multimodal-rag-poc/poc
pip install -r requirements.txt
```

### 2. Configure your API key

```bash
# Option A: Environment variable
export OPENAI_API_KEY=sk-your-key-here

# Option B: Create a .env file
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

### 3. Ingest the sample data

```bash
python ingest.py
```

This parses all reports in `sample-data/`, sends images to GPT-4o for description, embeds everything, and stores the results in a local ChromaDB instance.

### 4. Start the chat UI

```bash
streamlit run app.py
```

Open your browser to **http://localhost:8501** and start asking questions.

---

## 💡 Usage

### Web Chat Interface

```bash
cd poc
streamlit run app.py
```

The Streamlit UI provides a conversational chat experience with:
- Message history across the session
- Example questions in the sidebar
- Expandable source sections showing which documents and images informed each answer

### Command-Line Queries

```bash
cd poc
python query.py "Which towers need corrective action?"
```

Returns the answer and a list of cited sources — useful for scripting and automation.

### Ingesting Your Own Documents

```bash
# Place your PDFs in the confidential directory (gitignored)
cp /path/to/your/*.pdf ../sample-data/confidential/

# Ingest from that directory
python ingest.py --data-dir ../sample-data/confidential/
```

The system handles engineering drawings, A&E plans, structural analyses, and other technical PDFs. Pages with many small image fragments (common in CAD exports) are automatically rendered as full-page images for accurate vision model analysis.

---

## ⚙️ Configuration

All settings are managed in [`poc/config.py`](poc/config.py) and can be overridden with environment variables.

| Setting | Default | Description |
|---------|---------|-------------|
| `OPENAI_API_KEY` | — | **Required.** Your OpenAI API key |
| `VISION_MODEL` | `gpt-4o` | Model used to describe images |
| `LLM_MODEL` | `gpt-4o` | Model used for response generation |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Model used for text embeddings |
| `CHUNK_SIZE` | `2000` | Token count per text chunk |
| `CHUNK_OVERLAP` | `400` | Token overlap between adjacent chunks |
| `TOP_K` | `8` | Number of chunks retrieved per query |

---

## 📁 Project Structure

```
multimodal-rag-poc/
├── docs/
│   └── one-pager.md              # Business overview & value proposition
├── poc/
│   ├── ARCHITECTURE.md           # Component decisions & scaling strategy
│   ├── app.py                    # Streamlit chat interface
│   ├── config.py                 # Configuration & environment variables
│   ├── ingest.py                 # Document ingestion pipeline
│   ├── query.py                  # Query engine & RAG logic
│   ├── requirements.txt          # Python dependencies
│   ├── .env.example              # Environment variable template
│   └── vectorstore/              # ChromaDB data (gitignored)
├── sample-data/
│   ├── tower-report-*.md         # 5 inspection reports (markdown)
│   ├── tower-report-*.pdf        # 5 inspection reports (PDF)
│   ├── images/                   # Inspection photos (12 JPGs)
│   │   └── diagrams/             # Coverage maps, elevation drawings,
│   │                             # radiation patterns, RF compliance charts
│   └── confidential/             # Your own documents (gitignored)
└── README.md
```

---

## 🧪 Sample Questions

Try these queries to explore the system's capabilities:

```
Which towers have issues that need corrective action?
What's the grounding resistance for each tower?
Which tower is in Gaffney and what's its condition?
What's the FCC ASR number for the tower in Kimberly, NV?
Describe the foundation condition of tower A0028075
List all towers and their locations
What equipment is installed on each tower?
Summarize the structural analysis results
```

---

## 🔒 Data Privacy & Security

This system is designed to work with confidential engineering documents. Here's how data is handled:

| Concern | How It's Addressed |
|---------|-------------------|
| **Local parsing** | Documents are parsed locally with PyMuPDF — no files are uploaded in their original form |
| **OpenAI API usage** | Text chunks and images are sent to the OpenAI API. Per [OpenAI's API data usage policy](https://openai.com/policies/api-data-usage-policies), **API data is NOT used to train models** |
| **Vector store contents** | Contains only embeddings (numerical vectors) and text chunks — not the original documents |
| **Confidential directory** | `sample-data/confidential/` is gitignored to prevent accidental commits |
| **Air-gapped alternative** | Swap GPT-4o for local models (e.g., LLaVA for vision, Ollama for embeddings) for offline environments |

> ⚠️ **Your responsibility:** Ensure your organization's data policies allow sending document content to the OpenAI API before using this system with sensitive data.

---

## 📈 Scaling Roadmap

This POC is intentionally simple. The architecture is designed to scale without re-architecture:

```
Phase 1 (Current)                          Phase 2 (Production)
─────────────────                          ────────────────────
Simple RAG + Metadata Filtering     →      Hybrid RAG + Knowledge Graph

• Vector similarity search                 • Add Neo4j graph layer
• Metadata WHERE filters                   • Entity extraction at ingest time
• Handles ~5–100 documents                 • Multi-hop reasoning
• Covers ~80% of real-world queries        • Handles 5,000+ documents
```

> 📖 See [`poc/ARCHITECTURE.md`](poc/ARCHITECTURE.md) for the full scaling strategy, including why GraphRAG isn't used from day one.

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [`poc/ARCHITECTURE.md`](poc/ARCHITECTURE.md) | Component decisions, trade-off analysis, and scaling strategy |
| [`docs/one-pager.md`](docs/one-pager.md) | Business overview, target industries, and expected outcomes |
| [`poc/.env.example`](poc/.env.example) | Environment variable template |

---

## 🤝 Contributing

Contributions are welcome! To get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Test with the sample data (`python ingest.py && streamlit run app.py`)
5. Commit your changes (`git commit -m "Add your feature"`)
6. Push to the branch (`git push origin feature/your-feature`)
7. Open a Pull Request

---

## 📄 License

This project is provided as-is for demonstration and evaluation purposes. See the repository for any license terms.
