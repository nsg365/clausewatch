from clausewatch.ingest.chunking import _heading, chunk_pages
from clausewatch.ingest.pipeline import pretty_title
from clausewatch.taxonomy import tag_clauses
from clausewatch.textutil import quote_in_text


def test_quote_matching_tolerates_pdf_whitespace_and_punctuation():
    text = "Either Party may terminate this\nAgreement for any reason upon  providing ninety (90) days' notice."
    assert quote_in_text("may terminate this Agreement for any reason upon providing ninety (90) days’ notice", text)
    assert quote_in_text("Either Party may terminate ... ninety (90) days", text)
    assert not quote_in_text("may terminate this Agreement upon thirty (30) days notice", text)
    assert not quote_in_text("terminate", text)  # too short to count as evidence


def test_heading_detection():
    assert _heading("2.TERM, EXTENSION AND RENEWAL:\n") == "2.TERM, EXTENSION AND RENEWAL"
    assert _heading("12.1 Termination for Cause. Either party may terminate\n") == "12.1 Termination for Cause"
    assert _heading("ARTICLE 5 - INDEMNIFICATION\n") == "ARTICLE 5 - INDEMNIFICATION"
    assert _heading("the parties agree as follows\n") is None


def test_chunks_track_pages_and_sections():
    pages = ["1. Term\n" + "The term is five years. " * 30, "2. Termination\n" + "Either party may terminate. " * 30]
    chunks = chunk_pages(pages, chunk_size=300, overlap=40)
    assert {c.section for c in chunks} == {"1. Term", "2. Termination"}
    assert all(c.page_start == 1 for c in chunks if c.section == "1. Term")
    assert all(c.page_start == 2 for c in chunks if c.section == "2. Termination")
    assert all(not c.text[0].islower() for c in chunks)  # no mid-word starts


def test_tagger():
    assert "indemnification" in tag_clauses("Supplier shall indemnify and hold harmless Buyer.")
    assert "renewal" in tag_clauses("This Agreement shall automatically renew for successive one year terms.")
    assert "termination" in tag_clauses("text", section="10. TERMINATION")


def test_pretty_title():
    assert pretty_title("LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT") == "LIMEENERGYCO - Distributor Agreement"
    assert (
        pretty_title("KitovPharmaLtd_20190326_20-F_EX-4.15_11584449_EX-4.15_Manufacturing Agreement")
        == "KitovPharmaLtd - Manufacturing Agreement"
    )
