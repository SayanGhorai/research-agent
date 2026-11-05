"""Offline regression tests for uploaded-document scope and evidence grounding.

No IBM calls or academic-network calls. The PDFs from private user tests are not
redistributed or needed by this suite.
"""
from unittest.mock import MagicMock, patch

from researchlens.agents import ReadAgent, VerifyAgent
from researchlens.config import Settings
from researchlens.retrieval import CorpusIndex, select_evidence
from researchlens.schemas import (
    ClaimRecord, EvidenceChunk, PaperRecord, RunResult,
    SourceProvider, TaskType, TextCoverage, ValidationState,
)
from researchlens.workflow import ResearchWorkflow, index_uploaded_paper
from researchlens.watsonx_client import UsageRecord


def paper(pid, provider=SourceProvider.UPLOAD, title=None):
    return PaperRecord(
        paper_id=pid, title=title or pid, source_provider=provider,
        text_coverage=(TextCoverage.FULL_TEXT if provider == SourceProvider.UPLOAD
                       else TextCoverage.ABSTRACT_ONLY),
    )


def passage(pid, text, page=0):
    return EvidenceChunk(paper_id=pid, source_type=TextCoverage.FULL_TEXT,
                         page_index=page, text=text)


def settings(enabled=False):
    return Settings(
        watsonx_api_key='mock-only' if enabled else '',
        watsonx_project_id='mock-project' if enabled else '',
        watsonx_url='https://us-south.ml.cloud.ibm.com',
        watsonx_model_id='ibm/mock',
        allow_live_generation=enabled,
    )


def test_scoped_retrieval_ranks_eligible_before_top_k():
    index = CorpusIndex(enable_semantic=False)
    # Unrelated corpus has far more candidates than default top-k.
    index.add_chunks([
        passage('online', f'model accuracy model accuracy sample {i}') for i in range(40)
    ])
    index.add_chunks([passage('uploaded', 'Our model accuracy was measured on plant images.')])
    evidence = select_evidence(
        index, 'model accuracy', max_total=4, top_candidates=5,
        allowed_paper_ids={'uploaded'},
    )
    assert len(evidence) == 1
    assert evidence[0].paper_id == 'uploaded'


def test_empty_scoped_selection_cannot_search_everything():
    index = CorpusIndex(enable_semantic=False)
    index.add_chunks([passage('p1', 'Research methodology.')])
    assert select_evidence(index, 'Research', allowed_paper_ids=set()) == []


def test_search_off_uses_uploads_only_even_with_old_arxiv_data():
    idx = CorpusIndex(enable_semantic=False)
    upload = paper('upload')
    online = paper('online', SourceProvider.ARXIV)
    idx.add_chunks([
        passage(upload.paper_id, 'This uploaded study discusses disease classification.'),
        passage(online.paper_id, 'This online study also discusses disease classification.'),
    ])
    with patch('researchlens.agents.search_arxiv') as arxiv, patch('researchlens.agents.search_crossref') as crossref:
        result, library, _ = ResearchWorkflow(settings()).run(
            TaskType.ANSWER_QUESTION, 'disease classification',
            [upload, online], idx, search_first=False,
        )
    arxiv.assert_not_called()
    crossref.assert_not_called()
    assert result.paper_ids_used == ['upload']
    assert len(library) == 2  # existing records not deleted from library


def test_explicit_single_paper_excludes_other_uploads():
    idx = CorpusIndex(enable_semantic=False)
    p1, p2 = paper('p1'), paper('p2')
    idx.add_chunks([
        passage('p1', 'The first study used 1350 tomato images.'),
        passage('p2', 'The second study used 4775 soybean images.'),
    ])
    result, _, _ = ResearchWorkflow(settings()).run(
        TaskType.ANSWER_QUESTION, 'plant disease dataset images',
        [p1, p2], idx, search_first=False, selected_paper_ids=['p1'],
    )
    assert result.paper_ids_used == ['p1']
    assert all(idx.get_chunks_by_ids([cid])[0].paper_id == 'p1' for cid in result.evidence_chunk_ids)


def test_unknown_paper_id_causes_no_model_call():
    idx = CorpusIndex(enable_semantic=False)
    p1 = paper('p1')
    idx.add_chunks([passage('p1', 'model accuracy results')])
    client = MagicMock()
    with patch('researchlens.workflow.get_client', return_value=client):
        result, _, _ = ResearchWorkflow(settings(True)).run(
            TaskType.ANSWER_QUESTION, 'model accuracy', [p1], idx,
            search_first=False, selected_paper_ids=['not-a-paper'],
        )
    client.chat.assert_not_called()
    assert result.evidence_chunk_ids == []
    assert 'no relevant evidence' in result.answer_text.lower()


def test_unrelated_citation_is_invalid_and_warning_retained():
    idx = CorpusIndex(enable_semantic=False)
    p1, p2 = paper('p1'), paper('p2')
    c1 = passage('p1', 'The proposed method achieved 98.24% accuracy.')
    c2 = passage('p2', 'An unrelated study achieved 87% accuracy.')
    idx.add_chunks([c1, c2])
    fake_answer = f'The proposed method achieved 87% accuracy. [{c2.chunk_id[:8]}]'
    fake_claim = ClaimRecord(claim_text=fake_answer, evidence_chunk_ids=[c2.chunk_id])
    usage = UsageRecord(input_tokens=10, output_tokens=10, model_id='ibm/mock')
    client = MagicMock()
    client.action_calls_used = 1
    with patch('researchlens.workflow.get_client', return_value=client), patch(
        'researchlens.agents.WriteAgent.synthesize', return_value=(fake_answer, [fake_claim], usage)
    ):
        result, _, _ = ResearchWorkflow(settings(True)).run(
            TaskType.ANSWER_QUESTION, 'proposed method accuracy',
            [p1, p2], idx, search_first=False, selected_paper_ids=['p1'],
        )
    assert result.validation_state == ValidationState.INVALID
    assert any('unknown' in w.lower() for w in result.warnings)


def test_answer_without_citations_cannot_be_marked_valid():
    p = paper('p1')
    c = passage('p1', 'The system achieved 98.24% accuracy.')
    result = RunResult(task=TaskType.ANSWER_QUESTION, question='Accuracy?')
    result.evidence_chunk_ids = [c.chunk_id]
    result.paper_ids_used = ['p1']
    result.answer_text = 'The system achieved 98.24% accuracy.'
    verified = VerifyAgent().verify_result(result, [c], [p])
    assert verified.validation_state != ValidationState.VALID
    assert verified.warnings


def test_ingested_pages_are_split_without_losing_page_provenance():
    idx = CorpusIndex(enable_semantic=False)
    p = paper('p1')
    text = ' '.join('plant disease model sample images accuracy' for _ in range(80))
    index_uploaded_paper(p, [passage(p.paper_id, text, page=3)], idx)
    assert idx.chunk_count() > 1
    assert all(c.page_index == 3 for c in idx._chunks)
    assert all(len(c.text) < 1200 for c in idx._chunks)


def test_identical_text_across_papers_preserves_both_provenances():
    idx = CorpusIndex(enable_semantic=False)
    idx.add_chunks([passage('p1', 'Same acknowledgment text'), passage('p2', 'Same acknowledgment text')])
    assert {c.paper_id for c in idx._chunks} == {'p1', 'p2'}
