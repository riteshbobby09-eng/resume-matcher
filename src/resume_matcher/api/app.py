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
from pydantic import BaseModel

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

    @app.post("/schedule-interview")
    async def schedule_interview(req: InterviewInviteRequest):
        """
        Schedule an MS Teams interview session and send invite to both
        candidate and interviewer.
        """
        import os
        import smtplib
        from datetime import datetime, timedelta
        from email import encoders
        from email.mime.base import MIMEBase
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        # Generate Teams link if not provided
        teams_link = req.teams_link.strip()
        if not teams_link:
            meeting_id = uuid.uuid4().hex[:12]
            teams_link = f"https://teams.microsoft.com/l/meetup-join/19%3ameeting_{meeting_id}%40thread.v2/0"

        # Parse start & end datetime
        try:
            clean_dt = req.interview_datetime.replace("Z", "+00:00")
            dt_start = datetime.fromisoformat(clean_dt)
        except Exception:
            dt_start = datetime.utcnow() + timedelta(days=1, hours=10)

        dt_end = dt_start + timedelta(minutes=req.duration_minutes)

        # Build RFC 5545 iCalendar content (.ics)
        dt_stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        dt_start_str = dt_start.strftime("%Y%m%dT%H%M%SZ")
        dt_end_str = dt_end.strftime("%Y%m%dT%H%M%SZ")
        uid = f"interview-{uuid.uuid4().hex[:12]}@resumematcher.ai"

        summary = f"MS Teams Interview: {req.job_title}"
        description = (
            f"MS Teams Interview for {req.job_title}\\n\\n"
            f"Candidate Score: {req.candidate_score:.1f}/100\\n"
            f"Join MS Teams Meeting: {teams_link}\\n\\n"
            f"Notes: {req.notes or 'None'}"
        )

        ics_content = (
            "BEGIN:VCALENDAR\r\n"
            "VERSION:2.0\r\n"
            "PRODID:-//Resume Matcher//Interview Scheduler//EN\r\n"
            "CALSCALE:GREGORIAN\r\n"
            "METHOD:REQUEST\r\n"
            "BEGIN:VEVENT\r\n"
            f"UID:{uid}\r\n"
            f"DTSTAMP:{dt_stamp}\r\n"
            f"DTSTART:{dt_start_str}\r\n"
            f"DTEND:{dt_end_str}\r\n"
            f"SUMMARY:{summary}\r\n"
            f"DESCRIPTION:{description}\r\n"
            f"LOCATION:{teams_link}\r\n"
            f"ORGANIZER;CN=Recruiter:mailto:{req.interviewer_email}\r\n"
            f"ATTENDEE;ROLE=REQ-PARTICIPANT;PARTSTAT=NEEDS-ACTION;CN=Candidate:mailto:{req.candidate_email}\r\n"
            f"ATTENDEE;ROLE=REQ-PARTICIPANT;PARTSTAT=NEEDS-ACTION;CN=Interviewer:mailto:{req.interviewer_email}\r\n"
            "STATUS:CONFIRMED\r\n"
            "END:VEVENT\r\n"
            "END:VCALENDAR\r\n"
        )

        # Check for SMTP configuration in environment
        smtp_host = os.getenv("SMTP_HOST")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_user = os.getenv("SMTP_USER")
        smtp_password = os.getenv("SMTP_PASSWORD")
        smtp_sent = False
        smtp_error = None

        if smtp_host and smtp_user and smtp_password:
            try:
                msg = MIMEMultipart()
                msg["From"] = smtp_user
                msg["To"] = req.candidate_email
                msg["Cc"] = req.interviewer_email
                msg["Subject"] = f"Interview Invitation: {req.job_title} via MS Teams"

                body = (
                    f"Hello,\n\n"
                    f"You are invited to an interview for the {req.job_title} position.\n\n"
                    f"Date & Time: {dt_start.strftime('%A, %B %d, %Y at %I:%M %p')}\n"
                    f"Duration: {req.duration_minutes} minutes\n"
                    f"Meeting Link: {teams_link}\n\n"
                    f"Candidate Match Score: {req.candidate_score:.1f}/100\n"
                    f"Notes: {req.notes or 'None'}\n\n"
                    f"Best regards,\nRecruitment Team"
                )
                msg.attach(MIMEText(body, "plain"))

                part = MIMEBase("text", "calendar", method="REQUEST", name="invite.ics")
                part.set_payload(ics_content.encode("utf-8"))
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", 'attachment; filename="invite.ics"')
                msg.attach(part)

                with smtplib.SMTP(smtp_host, smtp_port) as server:
                    server.starttls()
                    server.login(smtp_user, smtp_password)
                    server.sendmail(smtp_user, [req.candidate_email, req.interviewer_email], msg.as_string())
                smtp_sent = True
                logger.info("Interview invite email sent successfully via SMTP to %s and %s", req.candidate_email, req.interviewer_email)
            except Exception as e:
                logger.warning("SMTP sending failed: %s", e)
                smtp_error = str(e)

        return {
            "status": "success",
            "message": "Interview invite generated successfully",
            "smtp_sent": smtp_sent,
            "smtp_error": smtp_error,
            "teams_link": teams_link,
            "interview_datetime": dt_start.isoformat(),
            "candidate_email": req.candidate_email,
            "interviewer_email": req.interviewer_email,
            "ics_content": ics_content,
        }


class InterviewInviteRequest(BaseModel):
    candidate_email: str
    interviewer_email: str
    job_title: str = "Candidate Position"
    interview_datetime: str = ""
    duration_minutes: int = 45
    teams_link: str = ""
    candidate_score: float = 0.0
    notes: str = ""


# Create the default app instance for uvicorn
app = create_app()

