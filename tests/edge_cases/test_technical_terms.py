"""
Edge case tests: technical term preservation during text normalization and splitting.
"""

from __future__ import annotations

import pytest

from resume_matcher.preprocessing.cleaner import TextCleaner
from resume_matcher.preprocessing.normalizer import TextNormalizer
from resume_matcher.segmentation.sentence_splitter import SentenceSplitter


@pytest.mark.edge_case
class TestTechnicalTermsPreservation:
    """Verify tricky technical terms, symbols, and version identifiers are never corrupted."""

    @pytest.fixture
    def normalizer(self):
        return TextNormalizer()

    @pytest.fixture
    def splitter(self):
        return SentenceSplitter()

    @pytest.fixture
    def cleaner(self):
        return TextCleaner()

    def test_programming_languages_preserved(self, normalizer):
        """Ensure symbols in C++, C#, .NET, Node.js are preserved."""
        raw_text = "Proficient in C++, C#, .NET Core, and Node.js backend development."
        cleaned = normalizer.normalize_text(raw_text)
        assert "C++" in cleaned
        assert "C#" in cleaned
        assert ".NET" in cleaned
        assert "Node.js" in cleaned

    def test_web_and_cloud_technologies(self, normalizer):
        """Ensure Vue.js, CI/CD, TCP/IP, PL/SQL are preserved."""
        raw_text = "Experienced with Vue.js, CI/CD pipelines, TCP/IP networking, and PL/SQL queries."
        cleaned = normalizer.normalize_text(raw_text)
        assert "Vue.js" in cleaned
        assert "CI/CD" in cleaned
        assert "TCP/IP" in cleaned
        assert "PL/SQL" in cleaned

    def test_experience_and_gpa_formats(self, normalizer, splitter):
        """Ensure '5+ years' and '3.8/4.0 GPA' aren't split into fragmented sentences."""
        raw_text = "Has 5+ years of Python experience. Graduated with 3.8/4.0 GPA from university."
        cleaned = normalizer.normalize_text(raw_text)
        assert "5+ years" in cleaned
        assert "3.8/4.0" in cleaned

        from resume_matcher.domain.models import IndexedLine
        lines = [IndexedLine(global_line_number=1, text=cleaned, page_number=1, local_line_number=1)]
        sentences = splitter.split_lines(lines)
        assert len(sentences) == 2
        assert "5+ years" in sentences[0].text
        assert "3.8/4.0" in sentences[1].text

    def test_version_numbers_and_standards(self, normalizer, splitter):
        """Ensure OAuth 2.0 and ISO 27001 are preserved."""
        raw_text = "Implemented OAuth 2.0 authentication conforming to ISO 27001 standards."
        cleaned = normalizer.normalize_text(raw_text)
        assert "OAuth 2.0" in cleaned
        assert "ISO 27001" in cleaned
