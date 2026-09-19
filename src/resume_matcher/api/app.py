"""
FastAPI application for Resume Matcher.

Thin API layer — all business logic lives in services.
"""

from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from resume_matcher.config import AppConfig, load_config, setup_logging
from resume_matcher.domain.enums import DocumentType
from resume_matcher.services.ingestion_service import IngestionService
from resume_matcher.services.matching_service import MatchingService
from resume_matcher.reporting.report_builder import ReportBuilder
from resume_matcher.reporting.exporters import JsonExporter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(config: AppConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if config is None:
        config = load_config()
    setup_logging(config)

    app = FastAPI(
        title="Resume Matcher API",
        description="Deterministic, explainable Resume-JD matching system",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store config and services on app state
    app.state.config = config
    app.state.ingestion = IngestionService(config)
    app.state.matching = MatchingService(config)
    app.state.report_builder = ReportBuilder(config)
    app.state.exporter = JsonExporter(config.reporting.schema_version)

    # Register routes
    _register_routes(app)

    # Serve static dashboard UI
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return app


def _register_routes(app: FastAPI) -> None:
    """Register all API routes."""

    @app.get("/", include_in_schema=False)
    async def root():
        """Serve the HR dashboard UI."""
        static_dir = Path(__file__).parent / "static"
        return FileResponse(str(static_dir / "index.html"))

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": "0.1.0"}

    @app.post("/match")
    async def match_resume(
        jd_file: Annotated[UploadFile, File(description="Job Description (PDF/DOCX)")],
        resume_file: Annotated[UploadFile, File(description="Resume (PDF/DOCX)")],
    ):
        """
        Match a single resume against a single JD.

        Accepts PDF or DOCX files for both JD and resume.
        Returns the full recruiter report as JSON.
        """
        config: AppConfig = app.state.config
        ingestion: IngestionService = app.state.ingestion
        matching: MatchingService = app.state.matching
        report_builder: ReportBuilder = app.state.report_builder

        # Validate file extensions
        for f, label in [(jd_file, "JD"), (resume_file, "Resume")]:
            if f.filename:
                ext = Path(f.filename).suffix.lower()
                if ext not in (".pdf", ".docx"):
                    raise HTTPException(
                        status_code=400,
                        detail=f"{label} file must be PDF or DOCX, got '{ext}'",
                    )

        # Save uploaded files to temp directory
        with tempfile.TemporaryDirectory(prefix="resume_matcher_") as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Save JD
            jd_filename = jd_file.filename or f"jd_{uuid.uuid4().hex[:8]}.pdf"
            jd_path = tmp_path / jd_filename
            jd_content = await jd_file.read()
            jd_path.write_bytes(jd_content)

            # Save resume
            resume_filename = resume_file.filename or f"resume_{uuid.uuid4().hex[:8]}.pdf"
            resume_path = tmp_path / resume_filename
            resume_content = await resume_file.read()
            resume_path.write_bytes(resume_content)

            try:
                # 1. Ingest both documents
                jd_doc = ingestion.ingest(jd_path, DocumentType.JD)
                resume_doc = ingestion.ingest(resume_path, DocumentType.RESUME)

                # 2. Match and score
                candidate_score, evidence = matching.match_and_score(jd_doc, resume_doc)

                # 3. Build report
                report = report_builder.build_report(
                    jd_doc, resume_doc, candidate_score, evidence
                )

                # 4. Convert to JSON-serializable dict
                exporter: JsonExporter = app.state.exporter
                report_dict = {
                    "_schema_version": exporter._schema_version,
                    "_report_id": report.report_id,
                    "processing_details": report.processing_details,
                    "candidate_score": report.candidate_score,
                    "comparison_summary": getattr(report, "comparison_summary", {}),
                    "comparison_parameters": getattr(report, "comparison_parameters", []),
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

                return JSONResponse(content=report_dict)

            except Exception as exc:
                logger.exception("Matching failed: %s", exc)
                raise HTTPException(
                    status_code=500,
                    detail=f"Processing failed: {str(exc)}",
                )


# Create the default app instance for uvicorn
app = create_app()
