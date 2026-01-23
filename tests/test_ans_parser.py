"""Tests for ANS parser."""

import pytest
from pathlib import Path

from ans2py.ans_parser import ANSParser, parse_ans_file


@pytest.fixture
def mathfns_path():
    """Path to MATHFNS.ans test fixture."""
    return Path(__file__).parent / "fixtures" / "MATHFNS.ans"


def test_parse_header(mathfns_path):
    """Test that ANS header is parsed correctly."""
    parser = ANSParser(mathfns_path)
    ws = parser.parse()

    assert ws.version == "1.00"
    assert "APL2" in ws.name


def test_extract_symbols(mathfns_path):
    """Test that symbols are extracted."""
    parser = ANSParser(mathfns_path)
    ws = parser.parse()

    # Should find known function names
    assert "DOMINO" in ws.raw_symbols
    assert "EIGEN" in ws.raw_symbols
    assert "FFT" in ws.raw_symbols
    assert "FACTORIAL" in ws.raw_symbols


def test_list_functions(mathfns_path):
    """Test list_functions method."""
    parser = ANSParser(mathfns_path)
    functions = parser.list_functions()

    assert len(functions) > 0
    assert "DOMINO" in functions


def test_extract_all_apl_text(mathfns_path):
    """Test APL text extraction."""
    parser = ANSParser(mathfns_path)
    parser.parse()
    text = parser.extract_all_apl_text()

    # Should contain documentation
    assert "MATHFNS" in text
    assert "matrix inverse" in text.lower() or "DOMINO" in text

    # Should contain APL symbols
    assert "⍝" in text  # Comment symbol


def test_invalid_file():
    """Test handling of invalid file."""
    with pytest.raises(FileNotFoundError):
        parser = ANSParser("/nonexistent/file.ans")
        parser.parse()


def test_invalid_format(tmp_path):
    """Test handling of non-ANS file."""
    bad_file = tmp_path / "bad.ans"
    bad_file.write_bytes(b"NOT AN ANS FILE")

    with pytest.raises(ValueError, match="Invalid ANS file"):
        parser = ANSParser(bad_file)
        parser.parse()


def test_extract_function_signatures(mathfns_path):
    """Test function signature extraction."""
    parser = ANSParser(mathfns_path)
    parser.parse()
    sigs = parser.extract_function_signatures()

    # Check that known functions have signatures
    assert "DOMINO" in sigs
    assert "←" in sigs["DOMINO"]

    # Dyadic functions should have L in signature
    assert "L" in sigs.get("DOMINO", "")
    assert "L" in sigs.get("FACTORIAL", "")

    # Monadic functions like FFT should not have L
    assert "FFT" in sigs
    assert "L" not in sigs["FFT"]
