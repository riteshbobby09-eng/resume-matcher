"""
Section detector for Resume Matcher.

Detects section headings using structural patterns — NOT role-specific keywords.

Detection signals:
- ALL CAPS lines (e.g., "EDUCATION", "WORK EXPERIENCE")
- Colon-terminated lines (e.g., "Skills:", "Responsibilities:")
- Short lines that match known section aliases
- Lines with formatting indicators (bold markers from extraction)

Section alias mapping is generic: maps common phrasings to canonical SectionType
without being specific to any role or technology.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
import re

from resume_matcher.domain.enums import SectionType
from resume_matcher.domain.models import IndexedLine

logger = logging.getLogger(__name__)

# Generic section aliases → canonical SectionType
# Comprehensive aliases covering all common phrasings found in resumes and JDs
_SECTION_ALIASES: dict[str, SectionType] = {
    # ──────────────── SKILLS (30+ aliases) ────────────────
    "skills": SectionType.SKILLS,
    "technical skills": SectionType.SKILLS,
    "core skills": SectionType.SKILLS,
    "key skills": SectionType.SKILLS,
    "tech stack": SectionType.SKILLS,
    "technologies": SectionType.SKILLS,
    "tools & technologies": SectionType.SKILLS,
    "tools and technologies": SectionType.SKILLS,
    "competencies": SectionType.SKILLS,
    "core competencies": SectionType.SKILLS,
    "areas of expertise": SectionType.SKILLS,
    "proficiencies": SectionType.SKILLS,
    "technical proficiencies": SectionType.SKILLS,
    "skill set": SectionType.SKILLS,
    "skillset": SectionType.SKILLS,
    "technical expertise": SectionType.SKILLS,
    "programming languages": SectionType.SKILLS,
    "languages and tools": SectionType.SKILLS,
    "software skills": SectionType.SKILLS,
    "it skills": SectionType.SKILLS,
    "technical competencies": SectionType.SKILLS,
    "computer skills": SectionType.SKILLS,
    "professional skills": SectionType.SKILLS,
    "relevant skills": SectionType.SKILLS,
    "functional skills": SectionType.SKILLS,
    "tools": SectionType.SKILLS,
    "frameworks": SectionType.SKILLS,
    "platforms": SectionType.SKILLS,
    "databases": SectionType.SKILLS,
    "devops tools": SectionType.SKILLS,
    "cloud technologies": SectionType.SKILLS,
    "data tools": SectionType.SKILLS,
    "soft skills": SectionType.SKILLS,
    "interpersonal skills": SectionType.SKILLS,
    "technical knowledge": SectionType.SKILLS,
    "domain expertise": SectionType.SKILLS,
    "technology stack": SectionType.SKILLS,
    "tech skills": SectionType.SKILLS,
    "key technologies": SectionType.SKILLS,
    "skills & abilities": SectionType.SKILLS,
    "skills and abilities": SectionType.SKILLS,
    "skills & expertise": SectionType.SKILLS,
    "skills and expertise": SectionType.SKILLS,
    "expertise": SectionType.SKILLS,
    "areas of knowledge": SectionType.SKILLS,
    # ──────────────── EXPERIENCE (25+ aliases) ────────────────
    "experience": SectionType.EXPERIENCE,
    "work experience": SectionType.EXPERIENCE,
    "professional experience": SectionType.EXPERIENCE,
    "employment history": SectionType.EXPERIENCE,
    "work history": SectionType.EXPERIENCE,
    "career history": SectionType.EXPERIENCE,
    "professional background": SectionType.EXPERIENCE,
    "employment": SectionType.EXPERIENCE,
    "relevant experience": SectionType.EXPERIENCE,
    "industry experience": SectionType.EXPERIENCE,
    "job history": SectionType.EXPERIENCE,
    "positions held": SectionType.EXPERIENCE,
    "professional history": SectionType.EXPERIENCE,
    "working experience": SectionType.EXPERIENCE,
    "career experience": SectionType.EXPERIENCE,
    "internship experience": SectionType.EXPERIENCE,
    "internships": SectionType.EXPERIENCE,
    "past employment": SectionType.EXPERIENCE,
    "previous roles": SectionType.EXPERIENCE,
    "previous positions": SectionType.EXPERIENCE,
    "roles": SectionType.EXPERIENCE,
    "career profile": SectionType.EXPERIENCE,
    "career path": SectionType.EXPERIENCE,
    "work record": SectionType.EXPERIENCE,
    "summary of experience": SectionType.EXPERIENCE,
    "experience summary": SectionType.EXPERIENCE,
    "employment record": SectionType.EXPERIENCE,
    "job experience": SectionType.EXPERIENCE,
    "work details": SectionType.EXPERIENCE,
    "professional roles": SectionType.EXPERIENCE,
    "where i have worked": SectionType.EXPERIENCE,
    "where i worked": SectionType.EXPERIENCE,
    # ──────────────── EDUCATION (25+ aliases) ────────────────
    "education": SectionType.EDUCATION,
    "educational background": SectionType.EDUCATION,
    "academic background": SectionType.EDUCATION,
    "academic qualifications": SectionType.EDUCATION,
    "qualifications": SectionType.EDUCATION,
    "educational qualifications": SectionType.EDUCATION,
    "degrees": SectionType.EDUCATION,
    "academic details": SectionType.EDUCATION,
    "education & training": SectionType.EDUCATION,
    "education and training": SectionType.EDUCATION,
    "academic record": SectionType.EDUCATION,
    "educational history": SectionType.EDUCATION,
    "academic history": SectionType.EDUCATION,
    "scholastic record": SectionType.EDUCATION,
    "academic credentials": SectionType.EDUCATION,
    "my academic credentials": SectionType.EDUCATION,
    "educational credentials": SectionType.EDUCATION,
    "studies": SectionType.EDUCATION,
    "academic profile": SectionType.EDUCATION,
    "degree details": SectionType.EDUCATION,
    "university education": SectionType.EDUCATION,
    "college education": SectionType.EDUCATION,
    "school education": SectionType.EDUCATION,
    "coursework": SectionType.EDUCATION,
    "relevant coursework": SectionType.EDUCATION,
    "academic achievements": SectionType.EDUCATION,
    "educational details": SectionType.EDUCATION,
    "scholastic details": SectionType.EDUCATION,
    "academic summary": SectionType.EDUCATION,
    # ──────────────── SUMMARY ────────────────
    "summary": SectionType.SUMMARY,
    "professional summary": SectionType.SUMMARY,
    "executive summary": SectionType.SUMMARY,
    "career summary": SectionType.SUMMARY,
    "profile": SectionType.SUMMARY,
    "profile summary": SectionType.SUMMARY,
    "what i bring to the table": SectionType.SUMMARY,
    "about me": SectionType.SUMMARY,
    "personal statement": SectionType.SUMMARY,
    "overview": SectionType.SUMMARY,
    "introduction": SectionType.SUMMARY,
    # ──────────────── OBJECTIVE ────────────────
    "career objective": SectionType.OBJECTIVE,
    "objective": SectionType.OBJECTIVE,
    "career goal": SectionType.OBJECTIVE,
    "professional objective": SectionType.OBJECTIVE,
    # ──────────────── PROJECTS ────────────────
    "projects": SectionType.PROJECTS,
    "project experience": SectionType.PROJECTS,
    "key projects": SectionType.PROJECTS,
    "personal projects": SectionType.PROJECTS,
    "academic projects": SectionType.PROJECTS,
    "notable projects": SectionType.PROJECTS,
    "side projects": SectionType.PROJECTS,
    "major projects": SectionType.PROJECTS,
    # ──────────────── CERTIFICATIONS ────────────────
    "certifications": SectionType.CERTIFICATIONS,
    "certificates": SectionType.CERTIFICATIONS,
    "professional certifications": SectionType.CERTIFICATIONS,
    "licenses & certifications": SectionType.CERTIFICATIONS,
    "licenses and certifications": SectionType.CERTIFICATIONS,
    "training & certifications": SectionType.CERTIFICATIONS,
    "training and certifications": SectionType.CERTIFICATIONS,
    "certification": SectionType.CERTIFICATIONS,
    "professional development": SectionType.CERTIFICATIONS,
    "training": SectionType.CERTIFICATIONS,
    # ──────────────── ACHIEVEMENTS ────────────────
    "achievements": SectionType.ACHIEVEMENTS,
    "accomplishments": SectionType.ACHIEVEMENTS,
    "awards": SectionType.ACHIEVEMENTS,
    "awards & honors": SectionType.ACHIEVEMENTS,
    "awards and honors": SectionType.ACHIEVEMENTS,
    "honors": SectionType.ACHIEVEMENTS,
    "recognition": SectionType.ACHIEVEMENTS,
    "key achievements": SectionType.ACHIEVEMENTS,
    # ──────────────── CONTACT ────────────────
    "contact": SectionType.CONTACT,
    "contact information": SectionType.CONTACT,
    "contact details": SectionType.CONTACT,
    "personal information": SectionType.CONTACT,
    "personal details": SectionType.CONTACT,
    # ──────────────── RESPONSIBILITIES (JDs) ────────────────
    "responsibilities": SectionType.RESPONSIBILITIES,
    "key responsibilities": SectionType.RESPONSIBILITIES,
    "roles and responsibilities": SectionType.RESPONSIBILITIES,
    "roles & responsibilities": SectionType.RESPONSIBILITIES,
    "job responsibilities": SectionType.RESPONSIBILITIES,
    "duties": SectionType.RESPONSIBILITIES,
    "duties and responsibilities": SectionType.RESPONSIBILITIES,
    # ──────────────── REQUIREMENTS (JDs) ────────────────
    "requirements": SectionType.REQUIREMENTS,
    "job requirements": SectionType.REQUIREMENTS,
    "minimum requirements": SectionType.REQUIREMENTS,
    "required qualifications": SectionType.REQUIREMENTS,
    "desired qualifications": SectionType.REQUIREMENTS,
    "preferred qualifications": SectionType.REQUIREMENTS,
    "must have": SectionType.REQUIREMENTS,
    "nice to have": SectionType.REQUIREMENTS,
    "eligibility": SectionType.REQUIREMENTS,
    "minimum qualifications": SectionType.REQUIREMENTS,
    "basic qualifications": SectionType.REQUIREMENTS,
    # ──────────────── ABOUT (JDs) ────────────────
    "about us": SectionType.ABOUT,
    "about the company": SectionType.ABOUT,
    "company overview": SectionType.ABOUT,
    "about the role": SectionType.ABOUT,
    "job description": SectionType.ABOUT,
    "role overview": SectionType.ABOUT,
    "position overview": SectionType.ABOUT,
    "job overview": SectionType.ABOUT,
    # ──────────────── OTHER ────────────────
    "references": SectionType.OTHER,
    "hobbies": SectionType.OTHER,
    "interests": SectionType.OTHER,
    "hobbies & interests": SectionType.OTHER,
    "extracurricular": SectionType.OTHER,
    "volunteer": SectionType.OTHER,
    "volunteer experience": SectionType.OTHER,
    "publications": SectionType.OTHER,
    "languages": SectionType.OTHER,
    "additional information": SectionType.OTHER,
    "extra curricular activities": SectionType.OTHER,
    "activities": SectionType.OTHER,
    "memberships": SectionType.OTHER,
}


@dataclass
class DetectedSection:
    """A section heading detected in the document."""

    section_type: SectionType
    heading_text: str
    line_number: int
    confidence: float  # 0.0 – 1.0
    detection_method: str  # "exact_match", "alias_match", "pattern_match"




class SectionDetector:
    """
    Detects section headings in document lines.

    Uses structural patterns (not content keywords) to identify headings:
    1. Exact match against alias dictionary
    2. ALL CAPS short lines
    3. Colon-terminated short lines
    """

    # Maximum length for a line to be considered a heading
    MAX_HEADING_LENGTH = 80

    def detect_heading(self, text: str) -> SectionType | None:
        """Convenience method to detect section type from a heading string."""
        detected = self._try_detect(text, 1)
        return detected.section_type if detected else None

    def detect_sections(self, lines: list[IndexedLine]) -> list[DetectedSection]:
        """
        Scan lines for section headings.

        Args:
            lines: Indexed lines from the document.

        Returns:
            List of detected sections in document order.
        """
        sections: list[DetectedSection] = []

        for line in lines:
            if line.is_empty or not line.text.strip():
                continue

            text = line.text.strip()

            # Skip very long lines — unlikely to be headings
            if len(text) > self.MAX_HEADING_LENGTH:
                continue

            detected = self._try_detect(text, line.global_line_number)
            if detected:
                sections.append(detected)

        logger.debug("Detected %d section headings", len(sections))
        return sections

    def _try_detect(self, text: str, line_number: int) -> DetectedSection | None:
        """Try to detect a section heading from a line of text."""
        # Key-value lines like "CGPA: 8.5/10.0", "Email: ...", "Phone: ..." are not headings
        if ":" in text and not text.rstrip().endswith(":"):
            return None

        # Clean text for matching
        clean = self._normalize_for_matching(text)

        # 1. Exact alias match (highest confidence)
        if clean in _SECTION_ALIASES:
            return DetectedSection(
                section_type=_SECTION_ALIASES[clean],
                heading_text=text,
                line_number=line_number,
                confidence=0.95,
                detection_method="alias_match",
            )

        # 2. Colon-terminated alias match: "Skills:" → "skills"
        if clean.endswith(":"):
            base = clean[:-1].strip()
            if base in _SECTION_ALIASES:
                return DetectedSection(
                    section_type=_SECTION_ALIASES[base],
                    heading_text=text,
                    line_number=line_number,
                    confidence=0.90,
                    detection_method="alias_colon_match",
                )

        # 3. ALL CAPS line that matches an alias
        if text.isupper() and len(text.split()) <= 6:
            lower = text.lower().strip()
            if lower in _SECTION_ALIASES:
                return DetectedSection(
                    section_type=_SECTION_ALIASES[lower],
                    heading_text=text,
                    line_number=line_number,
                    confidence=0.90,
                    detection_method="caps_alias_match",
                )

        # 4. ALL CAPS short line (could be a heading we don't have an alias for)
        if text.isupper() and 2 <= len(text.split()) <= 5 and len(text) <= 50:
            return DetectedSection(
                section_type=SectionType.OTHER,
                heading_text=text,
                line_number=line_number,
                confidence=0.60,
                detection_method="caps_pattern",
            )

        # 5. Colon-terminated short line (potential heading)
        if text.endswith(":") and len(text.split()) <= 5 and len(text) <= 50:
            return DetectedSection(
                section_type=SectionType.OTHER,
                heading_text=text,
                line_number=line_number,
                confidence=0.55,
                detection_method="colon_pattern",
            )

        return None

    def _normalize_for_matching(self, text: str) -> str:
        """Normalize text for alias dictionary lookup."""
        # Strip, lowercase, remove trailing colon, collapse whitespace
        result = text.strip().lower()
        result = re.sub(r"\s+", " ", result)
        # Remove trailing punctuation (except colon, handled separately)
        result = result.rstrip(".-;")
        return result
