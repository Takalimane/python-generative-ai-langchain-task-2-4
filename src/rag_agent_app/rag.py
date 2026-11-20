from __future__ import annotations

from pathlib import Path
from typing import Iterable

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentIndex:
    def __init__(self, embeddings: Embeddings) -> None:
        self.embeddings = embeddings
        self.vector_store: FAISS | None = None
        self.documents: list[Document] = []

    def load_files(self, paths: Iterable[str | Path]) -> int:
        documents: list[Document] = []
        for raw_path in paths:
            path = Path(raw_path)
            if not path.exists():
                raise FileNotFoundError(f"Document does not exist: {path}")
            if path.suffix.lower() == ".pdf":
                documents.extend(PyPDFLoader(str(path)).load())
            elif path.suffix.lower() in {".txt", ".md", ".csv"}:
                documents.extend(TextLoader(str(path), encoding="utf-8").load())
            else:
                raise ValueError(f"Unsupported file type: {path.suffix}")

        if not documents:
            raise ValueError("No documents were loaded")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        for chunk in chunks:
            source = Path(chunk.metadata.get("source", "unknown")).name
            chunk.metadata["source"] = source

        self.documents = chunks
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        return len(chunks)

    def retrieve(self, query: str, k: int = 4) -> list[Document]:
        if self.vector_store is None:
            return []
        return self.vector_store.similarity_search(query, k=k)
