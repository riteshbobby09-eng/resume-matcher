"""
Domain enumerations for Resume Matcher.

These enums define the controlled vocabulary used throughout the system.
They are pure value types with no I/O or business logic.
"""

from __future__ import annotations

from enum import Enum, unique


@unique
class DocumentType(str, Enum):
    """Whether a document is a Job Description or a Resume."""

    JD = "jd"
    RESUME = "resume"


@unique
class FileFormat(str, Enum):
    """Supported file formats for document ingestion."""

    PDF = "pdf"
    DOCX = "docx"


@unique
class SectionType(str, Enum):
    """
    Canonical section types found in JDs and resumes.

    These are NOT role-specific — they represent structural sections
    common across all domains.
    """

    SKILLS = "skills"
    EXPERIENCE = "experience"
    EDUCATION = "education"
    SUMMARY = "summary"
    PROJECTS = "projects"
    CERTIFICATIONS = "certifications"
    ACHIEVEMENTS = "achievements"
    CONTACT = "contact"
    OBJECTIVE = "objective"
    RESPONSIBILITIES = "responsibilities"
    REQUIREMENTS = "requirements"
    ABOUT = "about"
    CONTENT = "content"  # Catch-all for final report scoring
    OTHER = "other"
    UNKNOWN = "unknown"


# The 4 sections shown in the final recruiter report
REPORT_SECTIONS = {
    SectionType.SKILLS,
    SectionType.EDUCATION,
    SectionType.EXPERIENCE,
    SectionType.CONTENT,
}


@unique
class MatchStrength(str, Enum):
    """Categorization of how strong a match is between a JD and resume block."""

    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NONE = "none"


@unique
class ValidationStatus(str, Enum):
    """Outcome of a validation check."""

    VALID = "valid"
    WARNING = "warning"
    ERROR = "error"


@unique
class ProcessingStage(str, Enum):
    """Stages in the document processing pipeline — used for traceability."""

    INGESTED = "ingested"
    EXTRACTED = "extracted"
    CLEANED = "cleaned"
    INDEXED = "indexed"
    SEGMENTED = "segmented"
    BLOCKS_CREATED = "blocks_created"
    SECTIONS_DETECTED = "sections_detected"
    METADATA_ASSIGNED = "metadata_assigned"
    VALIDATED = "validated"
    EMBEDDED = "embedded"
    MATCHED = "matched"
    SCORED = "scored"
    REPORTED = "reported"
