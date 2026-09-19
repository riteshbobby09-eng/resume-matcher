"""
Pipeline service for Resume Matcher.

End-to-end coordinator: ingest → match → score → report → export.
This is the top-level service that scripts and APIs call.
"""

from __future__ import annotations

import logging
from pathlib import Path

from resume_matcher.config import AppConfig
from resume_matcher.domain.enums import DocumentType
from resume_matcher.reporting.exporters import JsonExporter
from resume_matcher.reporting.report_builder import RecruiterReport, ReportBuilder
from resume_matcher.repositories.document_repo import InMemoryDocumentRepository
from resume_matcher.repositories.result_repo import InMemoryResultRepository
from resume_matcher.services.ingestion_service import IngestionService
from resume_matcher.services.matching_service import MatchingService

logger = logging.getLogger(__name__)


class PipelineService:
    """
    End-to-end pipeline coordinator.

    Orchestrates the complete flow from files to reports.
    Uses dependency injection for all components.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._ingestion = IngestionService(config)
        self._matching = MatchingService(config)
        self._report_builder = ReportBuilder(config)
        self._exporter = JsonExporter(config.reporting.schema_version)
        self._doc_repo = InMemoryDocumentRepository()
        self._result_repo = InMemoryResultRepository()

    def process_single(
        self,
        jd_path: str | Path,
        resume_path: str | Path,
        output_dir: str | Path = "output",
    ) -> Path:
        """
        Process a single JD–resume pair end-to-end.

        Args:
            jd_path: Path to the JD file (PDF or DOCX).
            resume_path: Path to the resume file (PDF or DOCX).
            output_dir: Directory for JSON report output.

        Returns:
            Path to the generated JSON report.
        """
        logger.info("Pipeline: %s vs %s", Path(jd_path).name, Path(resume_path).name)

        # 1. Ingest
        jd_doc = self._ingestion.ingest(jd_path, DocumentType.JD)
        resume_doc = self._ingestion.ingest(resume_path, DocumentType.RESUME)

        # Store in repositories
        self._doc_repo.save(jd_doc.meta.document_id, jd_doc)
        self._doc_repo.save(resume_doc.meta.document_id, resume_doc)

        # 2. Match and score
        candidate_score, evidence = self._matching.match_and_score(jd_doc, resume_doc)

        # Store result
        result_id = f"{candidate_score.jd_id}_{candidate_score.candidate_id}"
        self._result_repo.save(result_id, candidate_score)

        # 3. Build report
        report = self._report_builder.build_report(
            jd_doc, resume_doc, candidate_score, evidence
        )

        # 4. Export
        output_path = Path(output_dir) / (
            f"report_{jd_doc.meta.sanitized_filename}"
            f"_vs_{resume_doc.meta.sanitized_filename}.json"
        ).replace(" ", "_")

        exported = self._exporter.export(report, output_path)

        logger.info(
            "Pipeline complete: score=%.2f, report=%s",
            candidate_score.final_score,
            exported,
        )
        return exported

    def process_batch(
        self,
        jd_path: str | Path,
        resume_paths: list[str | Path],
        output_dir: str | Path = "output",
    ) -> list[Path]:
        """
        Process one JD against multiple resumes.

        Args:
            jd_path: Path to the JD file.
            resume_paths: Paths to resume files.
            output_dir: Directory for reports.

        Returns:
            List of paths to generated reports.
        """
        logger.info("Batch: %d resumes against %s", len(resume_paths), Path(jd_path).name)

        # Ingest JD once
        jd_doc = self._ingestion.ingest(jd_path, DocumentType.JD)
        self._doc_repo.save(jd_doc.meta.document_id, jd_doc)

        reports: list[Path] = []
        for idx, resume_path in enumerate(resume_paths, 1):
            try:
                logger.info("Processing resume %d/%d: %s", idx, len(resume_paths), Path(resume_path).name)

                resume_doc = self._ingestion.ingest(resume_path, DocumentType.RESUME)
                self._doc_repo.save(resume_doc.meta.document_id, resume_doc)

                candidate_score, evidence = self._matching.match_and_score(
                    jd_doc, resume_doc
                )
                result_id = f"{candidate_score.jd_id}_{candidate_score.candidate_id}"
                self._result_repo.save(result_id, candidate_score)

                report = self._report_builder.build_report(
                    jd_doc, resume_doc, candidate_score, evidence
                )

                output_path = (
                    Path(output_dir)
                    / f"report_{idx:03d}_{resume_doc.meta.sanitized_filename}.json"
                )
                exported = self._exporter.export(report, output_path)
                reports.append(exported)

                logger.info(
                    "Resume %d/%d done: score=%.2f",
                    idx,
                    len(resume_paths),
                    candidate_score.final_score,
                )
            except Exception as exc:
                logger.error(
                    "Failed to process resume %d (%s): %s",
                    idx,
                    Path(resume_path).name,
                    exc,
                )
                continue

        logger.info("Batch complete: %d/%d reports generated", len(reports), len(resume_paths))
        return reports
