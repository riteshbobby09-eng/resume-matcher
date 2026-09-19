"""Unit tests for section detector and inferencer."""

import pytest

from resume_matcher.config import SectionDetectionConfig
from resume_matcher.domain.enums import DocumentType, SectionType
from resume_matcher.domain.models import Block, IndexedLine, Sentence
from resume_matcher.metadata.section_detector import SectionDetector
from resume_matcher.metadata.section_inferencer import SectionInferencer


class TestSectionDetector:
    @pytest.fixture
    def detector(self):
        return SectionDetector()

    def test_detect_all_caps_heading(self, detector):
        lines = [
            IndexedLine(global_line_number=1, text="WORK EXPERIENCE"),
            IndexedLine(global_line_number=2, text="Worked at TechCorp for 5 years"),
        ]
        sections = detector.detect_sections(lines)
        assert len(sections) >= 1
        assert sections[0].section_type == SectionType.EXPERIENCE

    def test_detect_colon_heading(self, detector):
        lines = [
            IndexedLine(global_line_number=1, text="Skills:"),
            IndexedLine(global_line_number=2, text="Python, Java, Docker"),
        ]
        sections = detector.detect_sections(lines)
        assert len(sections) >= 1
        assert sections[0].section_type == SectionType.SKILLS

    def test_detect_education_variants(self, detector):
        variants = [
            "EDUCATION",
            "Educational Background",
            "Academic Qualifications",
            "Education & Training",
        ]
        for heading in variants:
            lines = [IndexedLine(global_line_number=1, text=heading)]
            sections = detector.detect_sections(lines)
            assert len(sections) >= 1, f"Should detect: {heading}"
            assert sections[0].section_type == SectionType.EDUCATION

    def test_detect_skills_variants(self, detector):
        variants = [
            "Technical Skills",
            "SKILLS",
            "Core Competencies",
            "Tech Stack",
        ]
        for heading in variants:
            lines = [IndexedLine(global_line_number=1, text=heading)]
            sections = detector.detect_sections(lines)
            assert len(sections) >= 1, f"Should detect: {heading}"
            assert sections[0].section_type == SectionType.SKILLS

    def test_skip_long_lines(self, detector):
        lines = [
            IndexedLine(
                global_line_number=1,
                text="This is a very long line that should not be detected as a heading " * 3,
            ),
        ]
        sections = detector.detect_sections(lines)
        assert len(sections) == 0

    def test_confidence_values(self, detector):
        lines = [IndexedLine(global_line_number=1, text="WORK EXPERIENCE")]
        sections = detector.detect_sections(lines)
        assert sections[0].confidence >= 0.8


    def test_jd_sections(self, detector):
        """Detect JD-specific sections like Requirements and Responsibilities."""
        jd_headings = ["Requirements", "Key Responsibilities", "About Us"]
        for heading in jd_headings:
            lines = [IndexedLine(global_line_number=1, text=heading)]
            sections = detector.detect_sections(lines)
            assert len(sections) >= 1, f"Should detect: {heading}"


class TestSectionInferencer:
    @pytest.fixture
    def inferencer(self):
        return SectionInferencer(SectionDetectionConfig())

    def test_infer_from_neighbor(self, inferencer):
        blocks = [
            Block(
                text="Python, Java, Docker, Kubernetes",
                section=SectionType.SKILLS,
                source_line_numbers=[1],
                position_in_document=0.3,
            ),
            Block(
                text="React, Angular, Vue.js, TypeScript",
                section=SectionType.UNKNOWN,
                source_line_numbers=[2],
                position_in_document=0.35,
            ),
        ]
        result = inferencer.infer_sections(blocks)
        # Second block should be inferred as SKILLS (same as neighbor)
        assert result[1].section == SectionType.SKILLS

    def test_infer_education_from_content(self, inferencer):
        blocks = [
            Block(
                text="Bachelor of Technology in Computer Science from IIT Delhi, CGPA: 8.7",
                section=SectionType.UNKNOWN,
                source_line_numbers=[10],
                position_in_document=0.8,
            ),
        ]
        result = inferencer.infer_sections(blocks)
        assert result[0].section == SectionType.EDUCATION

    def test_infer_experience_from_dates(self, inferencer):
        blocks = [
            Block(
                text="Senior Developer at TechCorp, 2020-2023. Led team of 5 engineers. Worked at the company for 3 years.",
                section=SectionType.UNKNOWN,
                source_line_numbers=[5],
                position_in_document=0.5,
            ),
        ]
        result = inferencer.infer_sections(blocks)
        # Should infer EXPERIENCE from date range and employment keywords
        assert result[0].section == SectionType.EXPERIENCE

    def test_inference_has_reasoning(self, inferencer):
        blocks = [
            Block(
                text="john@email.com, +1-555-1234",
                section=SectionType.UNKNOWN,
                source_line_numbers=[1],
                position_in_document=0.05,
            ),
        ]
        inferencer.infer_sections(blocks)
        inference = blocks[0].metadata.get("inference", {})
        assert inference.get("reasoning")
        assert inference.get("confidence", 0) > 0

    def test_low_confidence_gets_review_flag(self, inferencer):
        blocks = [
            Block(
                text="Some ambiguous content here",
                section=SectionType.UNKNOWN,
                source_line_numbers=[50],
                position_in_document=0.5,
            ),
        ]
        inferencer.infer_sections(blocks)
        inference = blocks[0].metadata.get("inference", {})
        # Low-confidence inferences should be flagged for review
        if inference.get("confidence", 0) < 0.5:
            assert inference.get("review_flag") is True
