"""
Report exporters for Resume Matcher.

Supports versioned JSON export with stable field ordering.
HTML/PDF exporters are stubbed for future implementation.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

from resume_matcher.exceptions import ExportError
from resume_matcher.reporting.report_builder import RecruiterReport

logger = logging.getLogger(__name__)


class BaseExporter(ABC):
    """Abstract base for report exporters."""

    @abstractmethod
    def export(self, report: RecruiterReport, output_path: str | Path) -> Path:
        """Export report to a file. Returns the output path."""
        ...


class JsonExporter(BaseExporter):
    """
    Exports reports as versioned JSON files.

    Features:
    - Schema version in header
    - Pretty-printed for readability
    - UTF-8 encoded
    - Stable field ordering for diff-friendliness
    """

    def __init__(self, schema_version: str = "1.0.0") -> None:
        self._schema_version = schema_version

    def export(self, report: RecruiterReport, output_path: str | Path) -> Path:
        """
        Export a recruiter report as JSON.

        Args:
            report: RecruiterReport to export.
            output_path: Path for the output JSON file.

        Returns:
            Path to the created file.

        Raises:
            ExportError: If file writing fails.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "_schema_version": self._schema_version,
            "_report_id": report.report_id,
            "processing_details": report.processing_details,
            "candidate_score": report.candidate_score,
            "skill_evidence": report.skill_evidence,
            "experience_evidence": report.experience_evidence,
            "education_evidence": report.education_evidence,
            "semantic_evidence": report.semantic_evidence,
            "strongest_matches": report.strongest_matches,
            "weakest_matches": report.weakest_matches,
            "mandatory_requirements": report.mandatory_requirements,
            "missing_requirements": report.missing_requirements,
            "review_flags": report.review_flags,
            "data_quality_warnings": report.data_quality_warnings,
            "recruiter_summary": report.recruiter_summary,
        }

        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False, sort_keys=False)
        except OSError as exc:
            raise ExportError(
                f"Failed to write report to {path}",
                details={"path": str(path), "error": str(exc)},
            ) from exc

        logger.info("Report exported to: %s (%d bytes)", path, path.stat().st_size)
        return path

    def export_as_string(self, report: RecruiterReport) -> str:
        """Export report as a JSON string."""
        data = {
            "_schema_version": self._schema_version,
            "_report_id": report.report_id,
            "processing_details": report.processing_details,
            "candidate_score": report.candidate_score,
            "recruiter_summary": report.recruiter_summary,
        }
        return json.dumps(data, indent=2, ensure_ascii=False)


class HtmlExporter(BaseExporter):
    """HTML report exporter (stub for future implementation)."""

    def export(self, report: RecruiterReport, output_path: str | Path) -> Path:
        raise NotImplementedError("HTML export is planned for a future version")


class PdfExporter(BaseExporter):
    """PDF report exporter (stub for future implementation)."""

    def export(self, report: RecruiterReport, output_path: str | Path) -> Path:
        raise NotImplementedError("PDF export is planned for a future version")
