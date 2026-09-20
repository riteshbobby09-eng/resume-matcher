"""Unit tests for config loading."""

import os
import pytest

from resume_matcher.config import load_config, ScoringWeights


class TestConfig:
    def test_default_config_loads(self):
        config = load_config()
        assert config.embedding.model_name == "BAAI/bge-base-en-v1.5"
        assert config.embedding.dimension == 768

    def test_scoring_weights(self):
        config = load_config()
        w = config.scoring.weights
        assert w.skill == 0.15
        assert w.work_experience == 0.35
        assert w.education == 0.15
        assert w.content_score == 0.35
        total = w.skill + w.work_experience + w.education + w.content_score
        assert abs(total - 1.0) < 1e-6

    def test_invalid_weights_raise(self):
        with pytest.raises(ValueError):
            ScoringWeights(skill=0.50, work_experience=0.30, education=0.20, semantic_match=0.20)

    def test_env_override(self):
        os.environ["RESUME_MATCHER_LOG_LEVEL"] = "DEBUG"
        try:
            config = load_config()
            assert config.logging.level == "DEBUG"
        finally:
            del os.environ["RESUME_MATCHER_LOG_LEVEL"]

    def test_test_config_override(self):
        config = load_config(
            config_path="config/default.yaml",
            override_path="config/test.yaml",
        )
        assert config.processing.max_file_size_mb == 5  # Overridden
        assert config.embedding.model_name == "BAAI/bge-base-en-v1.5"  # Not overridden
