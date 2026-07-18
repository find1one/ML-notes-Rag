import pytest
from langchain_core.documents import Document

from rag_modules.data_preparation import DataPreparationModule


def test_chunk_size_override_changes_recursive_split_length():
    module = DataPreparationModule("unused", chunk_size=1000, chunk_overlap=150)
    module.documents = [
        Document(
            page_content="# Heading\n\n" + ("word " * 400),
            metadata={
                "source": "/notes/example.md",
                "relative_path": "example.md",
                "parent_id": "parent",
                "title": "Heading",
                "topic": "Root",
                "chapter": "Root",
                "section_path": "Heading",
            },
        )
    ]

    chunks = module.chunk_documents()

    assert len(chunks) > 1
    assert max(len(chunk.page_content) for chunk in chunks) <= 1000


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (1000, -1), (1000, 1000)],
)
def test_invalid_chunk_settings_are_rejected(chunk_size, chunk_overlap):
    with pytest.raises(ValueError):
        DataPreparationModule("unused", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
