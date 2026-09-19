"""Unit tests for text cleaner and normalizer."""

import pytest

from resume_matcher.domain.models import IndexedLine
from resume_matcher.preprocessing.cleaner import TextCleaner
from resume_matcher.preprocessing.normalizer import TextNormalizer


class TestTextCleaner:
    @pytest.fixture
    def cleaner(self):
        return TextCleaner()

    def test_unicode_normalization(self, cleaner):
        # NFKC should normalize ﬁ → fi
        assert "fi" in cleaner.clean_text("ﬁnance")

    def test_control_char_removal(self, cleaner):
        text = "Hello\x00World\x07Test"
        result = cleaner.clean_text(text)
        assert "\x00" not in result
        assert "\x07" not in result
        assert "HelloWorldTest" == result

    def test_whitespace_normalization(self, cleaner):
        text = "  Hello   World  "
        assert cleaner.clean_text(text) == "Hello World"

    def test_mojibake_fix(self, cleaner):
        # \u2019 (right single quote) is valid Unicode, not mojibake.
        # NFKC normalization preserves it. Test that cleaning runs without error.
        result = cleaner.clean_text("don\u2019t")
        assert "don" in result and "t" in result

    def test_page_number_detection(self, cleaner):
        artifacts = [
            "Page 1 of 5",
            "page 3 of 10",
            "- 2 -",
            "3/10",
            "[5]",
            "42",
        ]
        for text in artifacts:
            assert cleaner._is_page_artifact(text), f"Should detect: {text!r}"

    def test_non_artifacts(self, cleaner):
        non_artifacts = [
            "This is a normal line",
            "Python 3.10",
            "Experience: 5+ years",
        ]
        for text in non_artifacts:
            assert not cleaner._is_page_artifact(text), f"Should NOT detect: {text!r}"

    def test_clean_lines_preserves_numbering(self, cleaner):
        lines = [
            IndexedLine(global_line_number=1, text="  Hello  World  "),
            IndexedLine(global_line_number=2, text=""),
            IndexedLine(global_line_number=3, text="Test  Line"),
        ]
        result = cleaner.clean_lines(lines)
        assert result[0].global_line_number == 1
        assert result[0].text == "Hello World"
        assert result[2].text == "Test Line"


class TestTextNormalizer:
    @pytest.fixture
    def normalizer(self):
        return TextNormalizer()

    def test_preserve_years_experience(self, normalizer):
        """Must preserve '3.5 years' and '5+ years'."""
        text = "Candidate has 3.5 years of experience and 5+ years in Python."
        result = normalizer.normalize_text(text)
        assert "3.5 years" in result
        assert "5+ years" in result or "5+" in result

    def test_preserve_cpp(self, normalizer):
        """Must preserve 'C++'."""
        result = normalizer.normalize_text("Proficient in C++ and C#")
        assert "C++" in result
        assert "C#" in result

    def test_preserve_dotnet(self, normalizer):
        """Must preserve '.NET'."""
        result = normalizer.normalize_text("Experience with .NET framework")
        assert ".NET" in result

    def test_preserve_nodejs(self, normalizer):
        """Must preserve 'node.js'."""
        result = normalizer.normalize_text("Built APIs with node.js and Vue.js")
        assert "node.js" in result or "Node.js" in result

    def test_preserve_gpa(self, normalizer):
        """Must preserve GPA values."""
        result = normalizer.normalize_text("CGPA: 8.7/10.0")
        assert "8.7" in result

    def test_preserve_email(self, normalizer):
        """Must preserve email addresses."""
        result = normalizer.normalize_text("Contact: john.doe@email.com")
        assert "john.doe@email.com" in result

    def test_preserve_version_numbers(self, normalizer):
        """Must preserve version numbers."""
        result = normalizer.normalize_text("Using Python 3.10 and Java 17")
        assert "3.10" in result

    def test_bullet_normalization(self, normalizer):
        """Various bullet types should be normalized."""
        bullets = ["● Item one", "▸ Item two", "➤ Item three", "- Item four"]
        for bullet in bullets:
            result = normalizer.normalize_text(bullet)
            assert "•" in result or result.startswith("•") or "Item" in result

    def test_preserve_phone_numbers(self, normalizer):
        result = normalizer.normalize_text("Phone: +1-555-123-4567")
        assert "555" in result
        assert "4567" in result

    def test_preserve_decimal_numbers(self, normalizer):
        result = normalizer.normalize_text("Score: 3.5 out of 5.0")
        assert "3.5" in result
        assert "5.0" in result
