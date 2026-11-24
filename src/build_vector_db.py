import os
from langchain_community.document_loaders import TextLoader
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from src.embeddings import LocalBGEEmbeddings

FAQ_PATH = os.getenv("FAQ_PATH", "data/company_faq.txt")
QDRANT_PATH = os.getenv("QDRANT_PATH", "qdrant_local_db")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "resolve_knowledge_base")


def build_index() -> None:
    loader = TextLoader(FAQ_PATH, encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=40)
    chunks = splitter.split_documents(documents)

    embeddings = LocalBGEEmbeddings()
    client = QdrantClient(path=QDRANT_PATH)

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    vector_store.add_documents(chunks)
    client.close()


if __name__ == "__main__":
    build_index()