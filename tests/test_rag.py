from pathlib import Path
from langchain_core.embeddings import Embeddings

from rag_agent_app.rag import DocumentIndex


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0]


def test_index_loads_text_and_retrieves(tmp_path):
    document = tmp_path / "notes.txt"
    document.write_text("Datheon teaches secure software engineering.", encoding="utf-8")

    index = DocumentIndex(FakeEmbeddings())
    count = index.load_files([document])

    assert count >= 1
    results = index.retrieve("secure software")
    assert results
