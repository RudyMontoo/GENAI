import asyncio
from uuid import uuid4
from pathlib import Path

from dotenv import load_dotenv
from playwright.async_api import async_playwright
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

load_dotenv()

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

EMBEDDING_MODEL = "Alibaba-NLP/gte-multilingual-base"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "real_estate"

llm = None
vector_store = None


def initialize_components():
    global llm, vector_store

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    if llm is None:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.9,
            max_tokens=500,
        )

    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True},
        )

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=VECTORSTORE_DIR,
        )

    return llm, vector_store


# -----------------------------
# Playwright URL Loader (Async)
# -----------------------------
async def load_urls_with_playwright(urls: list[str]) -> list[Document]:
    documents = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        for url in urls:
            await page.goto(url, timeout=60_000)
            await page.wait_for_load_state("networkidle")

            text = await page.inner_text("body")

            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": "news", "url": url},
                )
            )

        await browser.close()

    return documents


def process_urls(urls):
    """
    Scrape JS-heavy URLs and store them in Chroma
    """
    yield("Initializing components...")
    initialize_components()

    yield("Loading data with Playwright...")
    data = asyncio.run(load_urls_with_playwright(urls))

    yield("Splitting data...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    docs = splitter.split_documents(data)

    yield("Storing data...")
    ids = [str(uuid4()) for _ in docs]
    vector_store.add_documents(docs, ids=ids)

    return docs


# -----------------------------
# Entry Point
# -----------------------------



def generate_answer(query: str):
    initialize_components()

    if not vector_store:
        raise RuntimeError("Vector database is not initialized")
    # 1. Retrieve documents
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(query)

    # 2. Build context + sources
    context = "\n\n".join(doc.page_content for doc in docs)
    sources = list({doc.metadata.get("url", "unknown") for doc in docs})

    # 3. Prompt
    prompt = ChatPromptTemplate.from_template(
        """
You are a real estate and finance expert.

Use ONLY the context below to answer the question.
If the answer is not present, say you don't know.

Context:
{context}

Question:
{question}

Answer clearly and concisely.
"""
    )

    # 4. LLM call
    messages = prompt.format_messages(
        context=context,
        question=query
    )
    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "sources": sources
    }

if __name__ == "__main__":
    urls = [
        "https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html",
        "https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html",
    ]

    process_urls(urls)

    # results = vector_store.similarity_search(
    #     "30 year mortgage rate",
    #     k=2,
    # )
    #
    # for r in results:
    #     print("-" * 80)
    #     print(r.page_content[:500])

    result = generate_answer(
        "tell me what was the 30 year fixed mortgage rate along with the data"
    )

    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")
