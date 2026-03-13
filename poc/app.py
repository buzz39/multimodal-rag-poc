"""
Streamlit chat UI for the Multimodal RAG POC.

Simple chat interface for querying tower inspection reports.
"""

import os
import streamlit as st
from query import get_query_engine

st.set_page_config(page_title="Tower Inspection RAG", page_icon="🗼", layout="wide")
st.title("🗼 Tower Inspection Report Assistant")
st.caption("Ask questions about tower inspection reports — answers reference both text and image content.")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    try:
        chain, retriever = get_query_engine()
        st.session_state.chain = chain
        st.session_state.retriever = retriever
    except Exception as e:
        st.error(f"Failed to load vector store. Run `python ingest.py` first.\n\nError: {e}")
        st.stop()

def generate_response(question):
    """Run the RAG chain and append the assistant message."""
    docs = st.session_state.retriever.invoke(question)
    answer = st.session_state.chain.invoke(question)

    sources = []
    for doc in docs:
        sources.append({
            "source": doc.metadata.get("source", "unknown"),
            "type": doc.metadata.get("chunk_type", "text"),
            "image_path": doc.metadata.get("image_path", ""),
        })

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer.content,
        "sources": sources,
    })


def render_sources(sources):
    """Render source expander for a message."""
    with st.expander("📎 Sources"):
        for s in sources:
            icon = "🖼️" if s["type"] == "image" else "📝"
            st.markdown(f"{icon} **{s['source']}** ({s['type']})")
            if s.get("image_path"):
                img_path = os.path.join(
                    os.path.dirname(__file__), "..", "sample-data", s["image_path"]
                )
                if os.path.exists(img_path):
                    st.image(img_path, width=300)


# Check if last message is a user question without a response
needs_response = (
    st.session_state.messages
    and st.session_state.messages[-1]["role"] == "user"
)

# Chat input
if prompt := st.chat_input("Ask about tower inspections..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    needs_response = True

# Generate response for any pending user question
if needs_response:
    with st.spinner("Searching reports..."):
        generate_response(st.session_state.messages[-1]["content"])

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            render_sources(msg["sources"])

# Sidebar with example queries
with st.sidebar:
    st.header("Example Questions")
    examples = [
        "List all towers and their locations",
        "Which towers need corrective action or repairs?",
        "What equipment is installed on each tower?",
        "Summarize the structural analysis results",
        "What are the wind speed ratings for the towers?",
        "Describe any compliance issues found",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": ex})
            st.rerun()
