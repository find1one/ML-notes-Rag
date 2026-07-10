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
