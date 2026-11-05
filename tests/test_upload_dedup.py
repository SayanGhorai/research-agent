from researchlens.ingestion import compute_file_hash, is_duplicate_pdf
from researchlens.schemas import PaperRecord, SourceProvider


def uploaded(title, content):
    return PaperRecord(
        title=title,
        source_provider=SourceProvider.UPLOAD,
        file_hash=compute_file_hash(content),
    )


def test_identical_pdf_is_detected():
    content = b"%PDF-1.4 original bytes"
    paper = uploaded("sanitized_paper.pdf", content)
    assert is_duplicate_pdf(content, [paper])


def test_different_pdf_with_same_name_is_allowed():
    paper = uploaded("paper.pdf", b"%PDF-1.4 first")
    assert not is_duplicate_pdf(b"%PDF-1.4 second", [paper])


def test_online_paper_is_not_treated_as_uploaded_pdf():
    content = b"%PDF-1.4 content"
    online = PaperRecord(
        title="paper",
        source_provider=SourceProvider.ARXIV,
        file_hash=compute_file_hash(content),
    )
    assert not is_duplicate_pdf(content, [online])