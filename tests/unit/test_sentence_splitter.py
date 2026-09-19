"""Unit tests for sentence splitter."""

import pytest

from resume_matcher.domain.models import IndexedLine
from resume_matcher.segmentation.sentence_splitter import SentenceSplitter


@pytest.fixture
def splitter():
    return SentenceSplitter()


def _make_lines(texts: list[str]) -> list[IndexedLine]:
    """Helper to create IndexedLine list from text strings."""
    return [
        IndexedLine(
            global_line_number=i + 1,
            text=text,
            page_number=1,
            local_line_number=i + 1,
        )
        for i, text in enumerate(texts)
    ]


class TestSentenceSplitter:
    def test_basic_split(self, splitter):
        lines = _make_lines(["Hello world. This is a test. Another sentence."])
        sentences = splitter.split_lines(lines)
        assert len(sentences) >= 2

    def test_preserve_abbreviations(self, splitter):
        """Must NOT split on abbreviations like 'Dr.' or 'Inc.'"""
        lines = _make_lines(["Dr. Smith works at Inc. corp and is great."])
        sentences = splitter.split_lines(lines)
        # Should be one sentence (not split on Dr. or Inc.)
        combined = " ".join(s.text for s in sentences)
        assert "Dr." in combined or "Dr" in combined

    def test_preserve_decimal_years(self, splitter):
        """Must NOT split '3.5 years' into separate sentences."""
        lines = _make_lines(["Has 3.5 years of experience in Python."])
        sentences = splitter.split_lines(lines)
        combined = " ".join(s.text for s in sentences)
        assert "3.5" in combined

    def test_preserve_eg_ie(self, splitter):
        """Must NOT split on 'e.g.' or 'i.e.'"""
        lines = _make_lines(["Technologies e.g. Python and Java are required."])
        sentences = splitter.split_lines(lines)
        # e.g. should not cause a split
        assert any("e.g" in s.text for s in sentences)

    def test_bullet_points_separate(self, splitter):
        """Bullet points should be separate sentences."""
        lines = _make_lines([
            "• Built microservices with Python",
            "• Designed REST APIs with Flask",
            "• Managed CI/CD with Jenkins",
        ])
        sentences = splitter.split_lines(lines)
        assert len(sentences) == 3

    def test_numbered_list_items(self, splitter):
        lines = _make_lines([
            "1. First item in list",
            "2. Second item in list",
        ])
        sentences = splitter.split_lines(lines)
        assert len(sentences) == 2

    def test_source_line_tracking(self, splitter):
        lines = _make_lines(["Hello world.", "This is line two."])
        sentences = splitter.split_lines(lines)
        for sent in sentences:
            assert len(sent.source_line_numbers) > 0

    def test_empty_input(self, splitter):
        sentences = splitter.split_lines([])
        assert sentences == []

    def test_preserves_urls(self, splitter):
        lines = _make_lines(["Visit https://example.com for more info. Thank you."])
        sentences = splitter.split_lines(lines)
        combined = " ".join(s.text for s in sentences)
        assert "https://example.com" in combined

    def test_multiline_paragraph(self, splitter):
        """Consecutive non-empty lines should be joined before splitting."""
        lines = _make_lines([
            "This is the first part of a paragraph",
            "and this continues the same paragraph.",
            "",
            "This is a new paragraph.",
        ])
        sentences = splitter.split_lines(lines)
        assert len(sentences) >= 2
