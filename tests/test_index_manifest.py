from pathlib import Path

from langchain_core.documents import Document

from rag_modules.index_construction import IndexConstructionModule


def test_manifest_detects_corpus_change(tmp_path):
    module = IndexConstructionModule.__new__(IndexConstructionModule)
    module.model_name = "test-model"
    module.index_save_path = str(tmp_path)
    module.manifest_path = Path(tmp_path) / "manifest.json"
    chunks = [Document(page_content="content", metadata={"chunk_id": "one"})]
    module._write_manifest(chunks)
    assert module._manifest_matches(chunks)
    assert not module._manifest_matches([Document(page_content="changed", metadata={"chunk_id": "one"})])


def test_in_memory_index_build_does_not_write_manifest(monkeypatch, tmp_path):
    module = IndexConstructionModule.__new__(IndexConstructionModule)
    module.embeddings = object()
    module.vectorstore = None
    module.index_save_path = str(tmp_path)
    module.manifest_path = Path(tmp_path) / "manifest.json"
    vectorstore = object()
    monkeypatch.setattr(
        "rag_modules.index_construction.FAISS.from_documents",
        lambda documents, embedding: vectorstore,
    )

    result = module.build_vector_index(
        [Document(page_content="content", metadata={"chunk_id": "one"})],
        write_manifest=False,
    )

    assert result is vectorstore
    assert not module.manifest_path.exists()
