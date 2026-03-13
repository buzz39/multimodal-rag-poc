"""
Ingestion pipeline for multimodal tower inspection reports.

Parses PDFs and markdown files, extracts text + images,
uses GPT-4o to describe images, embeds everything into ChromaDB.
"""

import os
import sys
import base64
import glob
import re
import fitz  # PyMuPDF
from PIL import Image
from io import BytesIO

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

import config


def extract_site_metadata(text: str, filename: str) -> dict:
    """Extract key site/tower identifiers from document text using regex patterns."""
    metadata = {"filename": filename}

    patterns = {
        "site_number": [
            r"Site\s*(?:Number|No\.?|ID|#)\s*[:\-]?\s*([A-Z0-9\-]+)",
            r"Site\s*[:\-]\s*([A-Z0-9\-]+)",
            r"FA(?:\s*Location)?\s*(?:Number|No\.?|#|ID)\s*[:\-]?\s*([A-Z0-9\-]+)",
        ],
        "site_name": [
            r"Site\s*Name\s*[:\-]?\s*(.+?)(?:\n|$)",
            r"Project\s*Name\s*[:\-]?\s*(.+?)(?:\n|$)",
        ],
        "tower_height": [
            r"(?:Tower|Structure|Overall)\s*Height\s*[:\-]?\s*([\d.]+\s*(?:ft|feet|m|meters))",
            r"Height\s*[:\-]?\s*([\d.]+\s*(?:ft|feet|m|meters))",
        ],
        "tower_type": [
            r"(?:Tower|Structure)\s*Type\s*[:\-]?\s*(.+?)(?:\n|$)",
            r"Type\s*of\s*(?:Tower|Structure)\s*[:\-]?\s*(.+?)(?:\n|$)",
        ],
        "location": [
            r"(?:Site\s*)?Address\s*[:\-]?\s*(.+?)(?:\n|$)",
            r"Location\s*[:\-]?\s*(.+?)(?:\n|$)",
            r"City\s*[,/]\s*State\s*[:\-]?\s*(.+?)(?:\n|$)",
        ],
        "carrier": [
            r"Carrier\s*[:\-]?\s*(.+?)(?:\n|$)",
            r"(?:Owner|Operator|Client)\s*[:\-]?\s*(.+?)(?:\n|$)",
        ],
        "latitude": [
            r"Lat(?:itude)?\s*[:\-]?\s*([\d.\-]+)",
        ],
        "longitude": [
            r"Lon(?:gitude)?\s*[:\-]?\s*([\d.\-]+)",
        ],
    }

    for field, field_patterns in patterns.items():
        for pattern in field_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if value and len(value) < 200:
                    metadata[field] = value
                    break

    return metadata


def build_context_header(metadata: dict) -> str:
    """Build a context header string from extracted metadata."""
    parts = []
    if metadata.get("site_number"):
        parts.append(f"Site: {metadata['site_number']}")
    if metadata.get("site_name"):
        parts.append(f"Name: {metadata['site_name']}")
    if metadata.get("location"):
        parts.append(f"Location: {metadata['location']}")
    if metadata.get("tower_height"):
        parts.append(f"Tower: {metadata['tower_height']}")
    if metadata.get("tower_type"):
        parts.append(f"Type: {metadata['tower_type']}")
    if metadata.get("carrier"):
        parts.append(f"Carrier: {metadata['carrier']}")
    if not parts:
        parts.append(f"File: {metadata.get('filename', 'unknown')}")
    return "[" + " | ".join(parts) + "]"


def extract_from_pdf(pdf_path: str) -> tuple[list[str], list[dict]]:
    """Extract text chunks and images from a PDF file."""
    doc = fitz.open(pdf_path)
    full_text = ""
    images = []

    MIN_IMAGE_SIZE = 100

    for page_num, page in enumerate(doc):
        full_text += page.get_text() + "\n"

        page_images = page.get_images(full=True)

        if len(page_images) > 5:
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            images.append({
                "data": img_data,
                "ext": "png",
                "page": page_num + 1,
                "index": 0,
                "source": os.path.basename(pdf_path),
                "render_type": "full_page",
            })
        else:
            for img_index, img in enumerate(page_images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                if base_image:
                    w = base_image.get("width", 0)
                    h = base_image.get("height", 0)
                    if w < MIN_IMAGE_SIZE and h < MIN_IMAGE_SIZE:
                        continue
                    images.append({
                        "data": base_image["image"],
                        "ext": base_image["ext"],
                        "page": page_num + 1,
                        "index": img_index,
                        "source": os.path.basename(pdf_path),
                    })

    doc.close()
    return full_text, images


def extract_from_markdown(md_path: str) -> tuple[str, list[dict]]:
    """Extract text and referenced images from a markdown file."""
    with open(md_path, "r") as f:
        content = f.read()

    image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    images = []
    base_dir = os.path.dirname(md_path)

    for match in re.finditer(image_pattern, content):
        alt_text, img_path = match.group(1), match.group(2)
        full_img_path = os.path.join(base_dir, img_path)
        if os.path.exists(full_img_path):
            with open(full_img_path, "rb") as f:
                img_data = f.read()
            ext = os.path.splitext(img_path)[1].lstrip(".")
            images.append({
                "data": img_data,
                "ext": ext,
                "alt_text": alt_text,
                "source": os.path.basename(md_path),
                "image_path": img_path,
            })

    clean_text = re.sub(image_pattern, '[Image: \\1]', content)
    return clean_text, images


def describe_image(vision_llm: ChatOpenAI, image_data: bytes, ext: str, context: str = "") -> str:
    """Use GPT-4o to describe an image from a tower inspection report."""
    b64 = base64.b64encode(image_data).decode("utf-8")
    mime = f"image/{ext}" if ext != "jpg" else "image/jpeg"

    prompt = (
        "You are analyzing images from a cell tower inspection report. "
        "Describe this image in detail, including:\n"
        "- What equipment/structure is shown\n"
        "- Any visible conditions (damage, corrosion, wear)\n"
        "- Any text, numbers, labels, or measurements visible\n"
        "- Any compliance or safety observations\n"
    )
    if context:
        prompt += f"\nContext from the report: {context}\n"

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]
    )
    response = vision_llm.invoke([message])
    return response.content


def ingest(data_dir: str = None, vectorstore_dir: str = None):
    """Run the full ingestion pipeline."""
    data_dir = data_dir or config.SAMPLE_DATA_DIR
    vectorstore_dir = vectorstore_dir or config.VECTORSTORE_DIR

    if not config.OPENAI_API_KEY:
        print("ERROR: Set OPENAI_API_KEY environment variable or create a .env file")
        sys.exit(1)

    print(f"📂 Data directory: {data_dir}")
    print(f"💾 Vector store: {vectorstore_dir}")

    # Initialize models
    vision_llm = ChatOpenAI(
        model=config.VISION_MODEL,
        api_key=config.OPENAI_API_KEY,
        max_tokens=500,
    )
    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        api_key=config.OPENAI_API_KEY,
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    all_documents = []
    all_site_metadata = []  # Collect metadata from all documents for summary

    # Process all supported files
    md_files = sorted(glob.glob(os.path.join(data_dir, "*.md")))
    pdf_files = sorted(glob.glob(os.path.join(data_dir, "*.pdf")))
    files_to_process = md_files + pdf_files

    for filepath in files_to_process:
        filename = os.path.basename(filepath)
        print(f"\n📄 Processing: {filename}")

        if filepath.endswith(".md"):
            text, images = extract_from_markdown(filepath)
        else:
            text, images = extract_from_pdf(filepath)

        # Extract site metadata from document text
        site_metadata = extract_site_metadata(text, filename)
        all_site_metadata.append(site_metadata)
        context_header = build_context_header(site_metadata)
        print(f"  🏷️  Metadata: {context_header}")

        # Split text into chunks, prepending context header to each
        chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            enriched_content = f"{context_header}\n{chunk}"
            all_documents.append(Document(
                page_content=enriched_content,
                metadata={
                    "source": filename,
                    "chunk_type": "text",
                    "chunk_index": i,
                    **{k: v for k, v in site_metadata.items() if k != "filename"},
                },
            ))
        print(f"  📝 {len(chunks)} text chunks")

        # Process images with vision model
        for img in images:
            print(f"  🖼️  Describing image from {filename}...")
            alt = img.get("alt_text", "")
            description = describe_image(vision_llm, img["data"], img["ext"], context=alt)

            content = f"{context_header}\n[Image from {filename}]"
            if alt:
                content += f"\nCaption: {alt}"
            content += f"\nDetailed description: {description}"

            all_documents.append(Document(
                page_content=content,
                metadata={
                    "source": filename,
                    "chunk_type": "image",
                    "image_path": img.get("image_path", ""),
                    "alt_text": alt,
                    **{k: v for k, v in site_metadata.items() if k != "filename"},
                },
            ))
        print(f"  🖼️  {len(images)} images described")

    # Create a summary document listing all sites/towers found
    if all_site_metadata:
        summary_lines = ["DOCUMENT SUMMARY — All towers/sites ingested:\n"]
        for i, meta in enumerate(all_site_metadata, 1):
            header = build_context_header(meta)
            summary_lines.append(f"{i}. {header} (from {meta.get('filename', 'unknown')})")
        summary_text = "\n".join(summary_lines)
        all_documents.append(Document(
            page_content=summary_text,
            metadata={
                "source": "system_summary",
                "chunk_type": "summary",
            },
        ))
        print(f"\n📋 Created summary document with {len(all_site_metadata)} sites")

    print(f"\n📊 Total documents to embed: {len(all_documents)}")

    # Create/overwrite vector store
    if os.path.exists(vectorstore_dir):
        import shutil
        shutil.rmtree(vectorstore_dir)

    vectorstore = Chroma.from_documents(
        documents=all_documents,
        embedding=embeddings,
        persist_directory=vectorstore_dir,
    )

    print(f"✅ Vector store created at {vectorstore_dir}")
    print(f"   {vectorstore._collection.count()} documents indexed")
    return vectorstore


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to data directory (default: sample-data/)")
    parser.add_argument("--vectorstore-dir", type=str, default=None,
                        help="Path to vector store directory")
    args = parser.parse_args()
    ingest(data_dir=args.data_dir, vectorstore_dir=args.vectorstore_dir)
