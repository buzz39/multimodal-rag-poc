"""
Query engine for multimodal tower inspection RAG.

Retrieves relevant text chunks and image descriptions from ChromaDB,
then uses GPT-4o to generate answers with source references.
"""

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

import config

SYSTEM_PROMPT = """\
You are a tower inspection report analyst. You answer questions about cell tower \
inspection reports using the retrieved context below.

Rules:
- Answer based on the provided context
- Always cite which report(s) your answer comes from (e.g., "tower-report-001.md")
- If the context includes image descriptions, reference them naturally
- Be specific with numbers, measurements, and technical details
- If the context contains partial information, share what you found and clearly note what's missing or uncertain
- When multiple towers or sites are referenced, organize your answer clearly by site/tower
- Do not refuse to answer if there is any relevant information in the context — provide what's available

Context:
{context}
"""


def get_query_engine():
    """Initialize and return the query chain."""
    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        api_key=config.OPENAI_API_KEY,
    )
    vectorstore = Chroma(
        persist_directory=config.VECTORSTORE_DIR,
        embedding_function=embeddings,
    )
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": config.TOP_K, "fetch_k": 20},
    )

    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        api_key=config.OPENAI_API_KEY,
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "unknown")
            chunk_type = doc.metadata.get("chunk_type", "text")
            header = f"[Source: {source} | Type: {chunk_type}]"
            formatted.append(f"{header}\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    return chain, retriever


def query(question: str) -> dict:
    """Query the RAG system and return answer + sources."""
    chain, retriever = get_query_engine()

    # Get sources for display
    docs = retriever.invoke(question)
    answer = chain.invoke(question)

    sources = []
    for doc in docs:
        sources.append({
            "source": doc.metadata.get("source", "unknown"),
            "type": doc.metadata.get("chunk_type", "text"),
            "image_path": doc.metadata.get("image_path", ""),
            "preview": doc.page_content[:200],
        })

    return {
        "answer": answer.content,
        "sources": sources,
    }


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Which towers have issues that need attention?"
    print(f"Question: {q}\n")
    result = query(q)
    print(f"Answer:\n{result['answer']}\n")
    print("Sources:")
    for s in result["sources"]:
        print(f"  - {s['source']} ({s['type']})")
