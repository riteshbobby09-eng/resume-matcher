"""Unit tests for scoring system."""

import pytest

from resume_matcher.config import ScoringWeights
from resume_matcher.domain.enums import MatchStrength, SectionType
from resume_matcher.domain.models import Block, MatchEvidence, MandatoryRequirement, ScoreComponent
from resume_matcher.exceptions import ScoringError
from resume_matcher.scoring.component_scorers import (
    EducationScorer,
    SemanticMatchScorer,
    SkillScorer,
    WorkExperienceScorer,
)
from resume_matcher.scoring.final_scorer import FinalScorer


class TestFinalScorer:
    @pytest.fixture
    def scorer(self):
        return FinalScorer(ScoringWeights())

    def test_exact_formula(self, scorer):
        """Verify: Final = Skill×0.30 + WorkExp×0.15 + Education×0.15 + Semantic×0.40"""
        components = [
            ScoreComponent(name="skill", raw_score=80.0, weight=0.30),
            ScoreComponent(name="work_experience", raw_score=70.0, weight=0.15),
            ScoreComponent(name="education", raw_score=90.0, weight=0.15),
            ScoreComponent(name="semantic_match", raw_score=75.0, weight=0.40),
        ]
        result = scorer.calculate(components)
        expected = 80 * 0.30 + 70 * 0.15 + 90 * 0.15 + 75 * 0.40
        assert abs(result.final_score - expected) < 0.1

    def test_all_zeros(self, scorer):
        components = [
            ScoreComponent(name="skill", raw_score=0.0, weight=0.30),
            ScoreComponent(name="work_experience", raw_score=0.0, weight=0.15),
            ScoreComponent(name="education", raw_score=0.0, weight=0.15),
            ScoreComponent(name="semantic_match", raw_score=0.0, weight=0.40),
        ]
        result = scorer.calculate(components)
        assert result.final_score == 0.0

    def test_all_perfect(self, scorer):
        components = [
            ScoreComponent(name="skill", raw_score=100.0, weight=0.30),
            ScoreComponent(name="work_experience", raw_score=100.0, weight=0.15),
            ScoreComponent(name="education", raw_score=100.0, weight=0.15),
            ScoreComponent(name="semantic_match", raw_score=100.0, weight=0.40),
        ]
        result = scorer.calculate(components)
        assert result.final_score == 100.0

    def test_missing_component_raises(self, scorer):
        components = [
            ScoreComponent(name="skill", raw_score=80.0, weight=0.30),
            ScoreComponent(name="work_experience", raw_score=70.0, weight=0.15),
        ]
        with pytest.raises(ScoringError, match="Missing"):
            scorer.calculate(components)

    def test_invalid_weight_sum(self):
        """Weights must sum to 1.0."""
        with pytest.raises(ValueError, match="sum to 1.0"):
            ScoringWeights(skill=0.50, work_experience=0.20, education=0.20, semantic_match=0.20)

    def test_score_out_of_range_raises(self, scorer):
        components = [
            ScoreComponent(name="skill", raw_score=150.0, weight=0.30),  # Invalid!
            ScoreComponent(name="work_experience", raw_score=70.0, weight=0.15),
            ScoreComponent(name="education", raw_score=90.0, weight=0.15),
            ScoreComponent(name="semantic_match", raw_score=75.0, weight=0.40),
        ]
        with pytest.raises(ScoringError, match="outside 0–100"):
            scorer.calculate(components)

    def test_mandatory_requirements_tracked(self, scorer):
        components = [
            ScoreComponent(name="skill", raw_score=80.0, weight=0.30),
            ScoreComponent(name="work_experience", raw_score=70.0, weight=0.15),
            ScoreComponent(name="education", raw_score=90.0, weight=0.15),
            ScoreComponent(name="semantic_match", raw_score=75.0, weight=0.40),
        ]
        mandatory = [
            MandatoryRequirement(requirement_text="5+ years Python", is_met=True),
            MandatoryRequirement(requirement_text="AWS experience", is_met=False),
        ]
        result = scorer.calculate(components, mandatory_requirements=mandatory)
        assert not result.all_mandatory_met
        assert len(result.mandatory_requirements) == 2

    def test_review_flags_on_zero_score(self, scorer):
        components = [
            ScoreComponent(name="skill", raw_score=80.0, weight=0.30),
            ScoreComponent(name="work_experience", raw_score=0.0, weight=0.15),  # Zero!
            ScoreComponent(name="education", raw_score=90.0, weight=0.15),
            ScoreComponent(name="semantic_match", raw_score=75.0, weight=0.40),
        ]
        result = scorer.calculate(components)
        assert any("Zero score" in flag for flag in result.review_flags)


class TestSkillScorer:
    def test_full_match(self):
        scorer = SkillScorer()
        jd_blocks = [
            Block(block_id="jd1", text="Python", section=SectionType.SKILLS),
            Block(block_id="jd2", text="Docker", section=SectionType.SKILLS),
        ]
        evidence = [
            MatchEvidence(jd_block_id="jd1", similarity_score=0.9, match_strength=MatchStrength.STRONG),
            MatchEvidence(jd_block_id="jd2", similarity_score=0.85, match_strength=MatchStrength.STRONG),
        ]
        result = scorer.score(evidence, jd_blocks, weight=0.30)
        assert result.raw_score > 0
        assert result.weight == 0.30
        assert result.name == "skill"

    def test_no_skill_blocks(self):
        scorer = SkillScorer()
        jd_blocks = [
            Block(block_id="jd1", text="Summary", section=SectionType.SUMMARY),
        ]
        result = scorer.score([], jd_blocks, weight=0.30)
        assert result.raw_score == 0.0

    def test_partial_match_has_missing(self):
        scorer = SkillScorer()
        jd_blocks = [
            Block(block_id="jd1", text="Python", section=SectionType.SKILLS),
            Block(block_id="jd2", text="Kubernetes", section=SectionType.SKILLS),
        ]
        evidence = [
            MatchEvidence(jd_block_id="jd1", similarity_score=0.9, match_strength=MatchStrength.STRONG),
        ]
        result = scorer.score(evidence, jd_blocks, weight=0.30)
        assert len(result.missing_evidence) > 0
