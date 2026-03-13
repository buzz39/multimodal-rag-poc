# Architecture Decisions — Multimodal RAG POC

## Research Summary

We evaluated major open-source frameworks with 500+ GitHub forks:

| Framework | GitHub Forks | Multimodal RAG Support | Verdict |
|-----------|-------------|----------------------|---------|
| **LangChain** | 17k+ | Multi-vector retriever cookbook for text+tables+images | ✅ Selected |
| LlamaIndex | 7k+ | Good multimodal support via CLIP embeddings | Good but heavier |
| Unstructured.io | 2k+ | Document parsing (not full RAG pipeline) | Used as complement |
| RAGFlow | 3k+ | Full platform — too heavyweight for POC | Overkill |

## Component Decisions

### 1. Framework: LangChain

**Why:** Most mature ecosystem (17k+ forks), excellent multimodal RAG cookbook with Multi-Vector Retriever pattern. The pattern of "use vision model to describe images → embed descriptions → retrieve → generate" is well-documented in LangChain. Lighter integration surface than LlamaIndex for a POC — we don't need LlamaIndex's query engine abstractions.

### 2. Document Parser: PyMuPDF (fitz)

**Why:** Our documents are PDFs with embedded images. PyMuPDF extracts both text and images from PDFs efficiently without requiring external services (unlike Unstructured.io's hosted API). It's fast, well-maintained (8k+ stars), and has zero external dependencies. For a POC, we don't need Unstructured's partitioning complexity — we just need text + images extracted.

We also support the markdown source files directly since our sample data includes `.md` files with image references.

### 3. Vision Model: GPT-4o (via OpenAI API)

**Why:** Best-in-class vision understanding for technical diagrams. We need to extract text from tower inspection photos, radiation patterns, and technical diagrams. GPT-4o handles this well and the team already has OpenAI API access. Open-source alternatives (LLaVA, etc.) require GPU infrastructure — unnecessary complexity for a POC.

### 4. Embeddings: OpenAI text-embedding-3-small

**Why:** Cost-effective (62% cheaper than ada-002), 1536 dimensions, strong performance. Since we're already using OpenAI for the vision model and LLM, keeping embeddings in the same ecosystem simplifies API key management. For a POC, the marginal quality difference vs. larger models doesn't matter.

### 5. Vector Store: ChromaDB (local)

**Why:** Zero-config local vector store, perfect for POC. No database server to run. Persists to disk. The production system uses Supabase+pgvector — ChromaDB is the right choice for a demo that "just works" with `pip install`. Migration to pgvector later is straightforward since we're using LangChain's vector store abstraction.

### 6. LLM for Responses: GPT-4o

**Why:** Same model used for vision — consistent quality. Handles complex reasoning about tower inspection data well. Temperature set to 0 for factual consistency.

## Handling Confidential & Engineering Documents

Real-world tower documents (A&E drawings, structural analyses, compound plans) differ significantly from clean sample data:

- **Engineering drawings as image fragments:** CAD-exported PDFs often store drawings as hundreds of tiny image fragments rather than one coherent image. The ingestion pipeline detects this (>10 images per page) and renders the full page as a single image at 2x resolution, giving the vision model a complete view of the drawing.
- **Minimal extractable text:** Many engineering PDFs have most content in drawings, not in text layers. The vision model descriptions become the primary searchable content for these pages.
- **Mixed content:** Some documents (e.g., structural analysis reports) have rich text on some pages and pure drawings on others. The pipeline handles both seamlessly.
- **Data privacy:** Confidential documents live in `sample-data/confidential/` (gitignored). Only embeddings are persisted. See README for full privacy details.

## Pipeline Architecture

```
PDF/Markdown Documents
        ↓
   Parse & Extract
   (text chunks + images)
        ↓
   Vision Model (GPT-4o)
   describes each image
        ↓
   Embed everything
   (text chunks + image descriptions)
        ↓
   ChromaDB (local)
        ↓
   User Query → Retrieve top-k chunks → GPT-4o generates answer with references
```

Each stored document includes metadata: source file, page number, chunk type (text/image), and image path (if applicable). This enables source attribution in answers.

## Scaling Strategy: Simple RAG → Hybrid RAG

This POC is intentionally simple. That's a feature, not a limitation. Here's the roadmap from "works for 5 documents" to "works for 5,000."

### Phase 1: Simple RAG with Metadata Filtering (Current POC)

The current architecture is pure vector search with rich metadata. Every chunk is tagged with:
- **Tower ID** (e.g., `TWR-Z3-0042`)
- **Zone** (e.g., `Zone 3`)
- **Carrier** (e.g., `Verizon`, `T-Mobile`)
- **Compliance status** (`compliant`, `non-compliant`, `conditional`)
- **Document type** (`inspection`, `maintenance`, `compliance-audit`)

ChromaDB handles both semantic similarity search and metadata filtering in a single query. When a user asks "show me non-compliant towers in Zone 3," we combine embedding similarity with `where={"zone": "Zone 3", "compliance_status": "non-compliant"}`.

This handles ~80% of real-world queries: factual lookups, document retrieval, image search, and filtered queries across a manageable corpus.

### Phase 2: Hybrid RAG with Knowledge Graph (Production)

When the corpus grows to hundreds of documents and users start asking relationship queries, we add a knowledge graph layer (Neo4j or similar) alongside the vector store:

- **Entity extraction pipeline** runs at ingest time — identifies towers, zones, inspectors, violations, regulations, carriers, and equipment as graph nodes.
- **Relationships** are extracted and stored as edges: `INSPECTED_BY`, `LOCATED_IN`, `VIOLATES`, `MAINTAINED_BY`, `CONNECTED_TO`.
- **Multi-hop reasoning** becomes possible: *"Find all towers inspected by the same engineer that had violations in the last year"* — this requires traversing `Tower → INSPECTED_BY → Inspector → INSPECTED_BY → Tower → HAS_VIOLATION → Violation` which is trivial for a graph but impossible for pure vector search.
- **Vector store still handles semantic similarity** — "find reports similar to this one" or "show images that look like corrosion damage" remain vector queries.
- **Hybrid retriever** merges results from both sources, re-ranks, and feeds the combined context to the LLM.

```
Phase 1 (Current — Simple RAG):

  ┌──────────┐     ┌──────────┐     ┌──────────────┐     ┌──────────────┐     ┌─────────┐
  │   Docs   │────▶│  Parser  │────▶│  Embeddings  │────▶│ Vector Store │────▶│   LLM   │
  │ (PDF/MD) │     │ + Vision │     │  (OpenAI)    │     │  (ChromaDB)  │     │ (GPT-4o)│
  └──────────┘     └──────────┘     └──────────────┘     └──────────────┘     └─────────┘
                                                                ▲
                                                                │
                                                          User Query +
                                                        Metadata Filters


Phase 2 (Production — Hybrid RAG):

  ┌──────────┐     ┌──────────┐     ┌──────────────┐     ┌──────────────┐
  │   Docs   │────▶│  Parser  │────▶│  Embeddings  │────▶│ Vector Store │──┐
  │ (PDF/MD) │     │ + Vision │     │              │     │              │  │
  └──────────┘     └────┬─────┘     └──────────────┘     └──────────────┘  │
                        │                                                   │  ┌───────────────┐     ┌─────────┐
                        │           ┌──────────────┐     ┌──────────────┐  ├─▶│    Hybrid      │────▶│   LLM   │
                        └──────────▶│   Entity     │────▶│  Knowledge   │──┘  │   Retriever    │     │         │
                                    │  Extraction  │     │    Graph     │     │  (re-ranking)  │     └─────────┘
                                    └──────────────┘     └──────────────┘     └───────────────┘
                                                                                     ▲
                                                                                     │
                                                                               User Query
```

### Why Not GraphRAG from Day One?

Because it's over-engineering. A knowledge graph adds **3–4× complexity** to the system:

1. **Ontology design** — You need to define entity types, relationship types, and constraints before you ingest a single document. Get this wrong and you rebuild.
2. **Entity extraction pipeline** — NER models, coreference resolution, relationship extraction. Each adds latency, cost, and failure modes.
3. **Graph maintenance** — Schema evolution, deduplication, conflict resolution when two documents disagree about the same entity.
4. **Query routing** — Now you need a classifier to decide: does this query go to the vector store, the graph, or both? That's another model to tune.

The right approach: prove value with simple RAG first. When users start asking questions that simple RAG can't answer — relationship queries, aggregations across documents, temporal reasoning — that's when you layer on the graph. Not before.

### Ingestion at Scale

For hundreds of documents, the RAG model itself isn't the bottleneck. **The ingestion pipeline is.** Real challenges:

- **Messy PDFs** — Scanned documents, rotated pages, mixed layouts, embedded forms. No single parser handles everything; you need fallback chains (PyMuPDF → Tesseract OCR → manual review queue).
- **Varied formats** — PDFs, Word docs, scanned images, handwritten notes. Each needs a different extraction path.
- **Image extraction quality** — Diagrams lose context when extracted without surrounding text. Vision model descriptions need to capture spatial relationships, not just objects.
- **Chunking strategy** — Too small and you lose context; too large and retrieval precision drops. Domain-specific chunking (by report section, not arbitrary token count) matters at scale.

**Recommended orchestration:** Apache Airflow or Prefect for pipeline management. Each document flows through: `validate → parse → extract images → vision describe → chunk → embed → store`, with retry logic, dead-letter queues for failed documents, and quality checks (embedding coverage, chunk size distribution, metadata completeness) at each stage.

The unglamorous truth: getting RAG to work well at scale is 20% model architecture and 80% data engineering.
