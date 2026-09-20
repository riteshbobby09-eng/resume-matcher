"""
Section inferencer for Resume Matcher.

For blocks where no explicit heading was detected, infers the section using:
1. Surrounding content — what section are the neighboring blocks in?
2. Document position — education typically near end, summary near top
3. Content signals — dates + company patterns → EXPERIENCE, degrees → EDUCATION
4. Generic aliases — common phrasings mapped to canonical sections

Returns SectionInference with: section_name, confidence, reasoning, review_flag.
"""

from __future__ import annotations

import logging
import re

from resume_matcher.config import SectionDetectionConfig
from resume_matcher.domain.enums import SectionType
from resume_matcher.domain.models import Block, SectionInference

logger = logging.getLogger(__name__)

# Content signal patterns (NOT role-specific — structural patterns only)
_CONTENT_SIGNALS: list[tuple[re.Pattern[str], SectionType, str]] = [
    # Experience signals: date ranges, company-like patterns, action verbs, job titles
    (
        re.compile(
            r"(?:\b\d{4}\s*[-–—]\s*(?:\d{4}|present|current|till date)\b)",
            re.IGNORECASE,
        ),
        SectionType.EXPERIENCE,
        "Contains date range pattern (e.g., 2020-2023)",
    ),
    (
        re.compile(
            r"\b(?:worked at|worked with|employed at|joined|role|position|designation|"
            r"responsibilities|duties|client|employer|tenure)\b",
            re.IGNORECASE,
        ),
        SectionType.EXPERIENCE,
        "Contains employment-related keywords",
    ),
    (
        re.compile(
            r"\b(?:software engineer|developer|architect|team lead|tech lead|engineering manager|"
            r"product manager|data scientist|data engineer|devops engineer|consultant|"
            r"analyst|specialist|administrator|intern|associate|director|vp|vice president)\b",
            re.IGNORECASE,
        ),
        SectionType.EXPERIENCE,
        "Contains job title patterns",
    ),
    (
        re.compile(
            r"^\s*[-*•]?\s*(?:managed|spearheaded|architected|developed|implemented|designed|"
            r"coordinated|collaborated|delivered|engineered|optimized|resolved|maintained|"
            r"deployed|monitored|increased|reduced|achieved|drove|trained|mentored|led)\b",
            re.IGNORECASE | re.MULTILINE,
        ),
        SectionType.EXPERIENCE,
        "Starts with action verb typical of experience bullets",
    ),
    # Education signals: degree names, university patterns, majors, graduation years
    (
        re.compile(
            r"\b(?:bachelor|master|ph\.?d|diploma|degree|b\.?tech|m\.?tech|"
            r"b\.?sc|m\.?sc|b\.?e|m\.?e|mba|bba|bca|mca|b\.?a|m\.?a|"
            r"b\.?com|m\.?com|associate degree|doctorate|post graduate|undergraduate)\b",
            re.IGNORECASE,
        ),
        SectionType.EDUCATION,
        "Contains degree/qualification names",
    ),
    (
        re.compile(
            r"\b(?:university|college|institute|school|academy|campus|"
            r"cgpa|gpa|percentage|grade|marks|graduated|class of|batch of|passout)\b",
            re.IGNORECASE,
        ),
        SectionType.EDUCATION,
        "Contains educational institution or grading keywords",
    ),
    (
        re.compile(
            r"\b(?:computer science|information technology|electronics|electrical|mechanical|"
            r"civil engineering|data science|artificial intelligence|business administration|"
            r"mathematics|statistics|physics)\b",
            re.IGNORECASE,
        ),
        SectionType.EDUCATION,
        "Contains academic major or field of study",
    ),
    # Skills signals: comma-separated technical terms, pipe-separated lists, tech keywords
    (
        re.compile(r"(?:[\w\+\#\.]+\s*[,|/]\s*){3,}"),  # 3+ comma/pipe/slash-separated items
        SectionType.SKILLS,
        "Contains comma/pipe-separated list (likely skills)",
    ),
    (
        re.compile(
            r"\b(?:python|java|c\+\+|javascript|typescript|golang|rust|ruby|php|swift|kotlin|"
            r"react|angular|vue|next\.?js|node\.?js|express|django|fastapi|flask|spring boot|"
            r"docker|kubernetes|aws|azure|gcp|terraform|git|linux|sql|postgresql|mysql|mongodb|"
            r"redis|elasticsearch|kafka|graphql|rest api|ci/cd|html|css|tailwind)\b",
            re.IGNORECASE,
        ),
        SectionType.SKILLS,
        "Contains specific technical skill keywords",
    ),
    # Contact signals: email, phone, address patterns
    (
        re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
        SectionType.CONTACT,
        "Contains email address",
    ),
    (
        re.compile(r"(?:\+?\d{1,3}[\s\-]?)?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}"),
        SectionType.CONTACT,
        "Contains phone number pattern",
    ),
    (
        re.compile(r"\b(?:linkedin\.com/in/|github\.com/|[a-z0-9]+\.github\.io)\b", re.IGNORECASE),
        SectionType.CONTACT,
        "Contains profile URL pattern",
    ),
    # Certification signals
    (
        re.compile(
            r"\b(?:certified|certification|certificate|credential|license|"
            r"aws certified|google certified|microsoft certified|pmp|"
            r"scrum master|itil|cka|ckad|cissp)\b",
            re.IGNORECASE,
        ),
        SectionType.CERTIFICATIONS,
        "Contains certification-related keywords",
    ),
    # Project signals
    (
        re.compile(
            r"\b(?:project|built|developed|implemented|created|designed|"
            r"github|repository|demo|prototype|tech stack used)\b",
            re.IGNORECASE,
        ),
        SectionType.PROJECTS,
        "Contains project-related keywords",
    ),
    # Achievement signals
    (
        re.compile(
            r"\b(?:award|recognition|achievement|honor|prize|medal|"
            r"top performer|outstanding|excellence|hackathon winner|rank)\b",
            re.IGNORECASE,
        ),
        SectionType.ACHIEVEMENTS,
        "Contains achievement-related keywords",
    ),
    # Summary signals
    (
        re.compile(
            r"\b(?:experienced professional|results-driven|proven track record|"
            r"seeking an opportunity|dynamic and motivated|passionate engineer)\b",
            re.IGNORECASE,
        ),
        SectionType.SUMMARY,
        "Contains summary / objective phrasing",
    ),
]

# Document position expectations (for resumes)
_POSITION_PRIORS: dict[SectionType, tuple[float, float]] = {
    SectionType.CONTACT: (0.0, 0.15),
    SectionType.SUMMARY: (0.0, 0.25),
    SectionType.OBJECTIVE: (0.0, 0.25),
    SectionType.SKILLS: (0.1, 0.6),
    SectionType.EXPERIENCE: (0.2, 0.9),
    SectionType.EDUCATION: (0.5, 1.0),
    SectionType.PROJECTS: (0.3, 0.9),
    SectionType.CERTIFICATIONS: (0.6, 1.0),
    SectionType.ACHIEVEMENTS: (0.5, 1.0),
    SectionType.CONTENT: (0.0, 1.0),
}


class SectionInferencer:
    """
    Infers sections for blocks that have no detected heading.

    Uses multiple signals with confidence scoring and provides
    human-readable reasoning for each inference.
    """

    def __init__(self, config: SectionDetectionConfig) -> None:
        self._min_confidence = config.min_confidence
        self._flag_low_confidence = config.flag_low_confidence

    def infer_sections(self, blocks: list[Block]) -> list[Block]:
        """
        Infer sections for blocks that are still UNKNOWN.

        Modifies blocks in place, setting .section and .metadata["inference"].

        Args:
            blocks: Blocks with some sections already detected.

        Returns:
            Same blocks list with inferred sections.
        """
        for i, block in enumerate(blocks):
            if block.section == SectionType.OTHER:
                # Inspect OTHER blocks for strong structural signals
                reclassified = False
                for pattern, sec_type, _ in _CONTENT_SIGNALS:
                    if sec_type in (SectionType.EXPERIENCE, SectionType.EDUCATION, SectionType.SKILLS) and pattern.search(block.text):
                        block.section = sec_type
                        block.metadata["reclassified_from_other"] = True
                        reclassified = True
                        break
                if reclassified:
                    continue

            if block.section not in (SectionType.UNKNOWN, SectionType.OTHER):
                # Even if section is known, check for secondary section signals (e.g. skills in experience)
                for pattern, sec_type, _ in _CONTENT_SIGNALS:
                    if sec_type != block.section and pattern.search(block.text):
                        block.metadata["secondary_section"] = sec_type.value
                        break
                continue  # Already has a detected section

            inference = self._infer_block_section(block, blocks, i)

            if inference.confidence >= self._min_confidence:
                block.section = inference.section
            else:
                # Keep as UNKNOWN but still attach inference info
                inference.review_flag = True

            block.metadata["inference"] = {
                "section": inference.section.value,
                "confidence": inference.confidence,
                "reasoning": inference.reasoning,
                "review_flag": inference.review_flag,
                "evidence": inference.evidence,
            }

        inferred_count = sum(
            1 for b in blocks
            if b.metadata.get("inference") and b.section != SectionType.UNKNOWN
        )
        logger.debug(
            "Inferred sections for %d of %d blocks",
            inferred_count,
            len(blocks),
        )
        return blocks

    def _infer_block_section(
        self,
        block: Block,
        all_blocks: list[Block],
        block_index: int,
    ) -> SectionInference:
        """
        Infer section for a single block using multiple signals.

        Combines:
        1. Surrounding context (neighbors' sections)
        2. Content signal matching
        3. Document position priors
        """
        candidates: dict[SectionType, float] = {}
        evidence: list[str] = []

        # 1. Surrounding context
        neighbor_section, neighbor_conf, neighbor_reason = self._check_neighbors(
            all_blocks, block_index
        )
        if neighbor_section != SectionType.UNKNOWN:
            candidates[neighbor_section] = candidates.get(neighbor_section, 0) + neighbor_conf
            evidence.append(neighbor_reason)

        # 2. Content signals
        for pattern, section, reason in _CONTENT_SIGNALS:
            if pattern.search(block.text):
                candidates[section] = candidates.get(section, 0) + 0.3
                evidence.append(reason)

        # 3. Document position priors
        pos = block.position_in_document
        for section, (low, high) in _POSITION_PRIORS.items():
            if low <= pos <= high and section in candidates:
                candidates[section] = candidates[section] + 0.1
                evidence.append(
                    f"Document position ({pos:.2f}) is within expected range "
                    f"for {section.value} ({low:.1f}–{high:.1f})"
                )

        # Pick the best candidate
        if not candidates:
            return SectionInference(
                section=SectionType.UNKNOWN,
                confidence=0.0,
                reasoning="No signals matched for section inference",
                review_flag=True,
                evidence=evidence,
            )

        best_section = max(candidates, key=candidates.get)  # type: ignore[arg-type]
        raw_confidence = candidates[best_section]
        # Normalize confidence to 0–1 range
        confidence = min(raw_confidence, 1.0)

        reasoning_parts = [f"Inferred as {best_section.value} (confidence: {confidence:.2f})"]
        reasoning_parts.extend(f"  - {e}" for e in evidence)

        return SectionInference(
            section=best_section,
            confidence=confidence,
            reasoning="\n".join(reasoning_parts),
            review_flag=(confidence < self._min_confidence) and self._flag_low_confidence,
            evidence=evidence,
        )

    def _check_neighbors(
        self,
        blocks: list[Block],
        index: int,
    ) -> tuple[SectionType, float, str]:
        """
        Check what section the neighboring blocks belong to.

        If the previous block has a known section, this block likely
        belongs to the same section (especially if close in the document).
        """
        # Check previous blocks (up to 3 back)
        for offset in range(1, min(4, index + 1)):
            prev = blocks[index - offset]
            if prev.section != SectionType.UNKNOWN:
                # Closer neighbors get higher confidence
                conf = 0.4 - (offset - 1) * 0.1
                return (
                    prev.section,
                    conf,
                    f"Previous block ({offset} back) is in section '{prev.section.value}'",
                )

        # Check next blocks
        for offset in range(1, min(4, len(blocks) - index)):
            nxt = blocks[index + offset]
            if nxt.section != SectionType.UNKNOWN:
                conf = 0.3 - (offset - 1) * 0.1
                return (
                    nxt.section,
                    max(conf, 0.1),
                    f"Next block ({offset} forward) is in section '{nxt.section.value}'",
                )

        return SectionType.UNKNOWN, 0.0, ""
