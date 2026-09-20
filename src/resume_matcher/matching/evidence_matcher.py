"""
Evidence matcher for Resume Matcher.

For each JD block, retrieves the most similar resume blocks via FAISS
and constructs MatchEvidence with full audit trail:
- matched text, similarity scores, source sections, line numbers
- match strength classification, missing evidence, uncertainty flags
- human-readable explanations
"""

from __future__ import annotations

import logging

import re
import numpy as np

from resume_matcher.config import RetrievalConfig, ScoringThresholds
from resume_matcher.domain.enums import DocumentType, MatchStrength, SectionType
from resume_matcher.domain.models import Block, ComparisonParameter, MandatoryRequirement, MatchEvidence
from resume_matcher.embeddings.bge_embedder import BgeEmbedder
from resume_matcher.metadata.section_detector import _SECTION_ALIASES
from resume_matcher.retrieval.faiss_index import FaissIndex
from resume_matcher.retrieval.metadata_filter import MetadataFilter

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "and", "or", "the", "in", "with", "for", "to", "of", "a", "an", "is", "at", "by",
    "from", "on", "as", "experience", "years", "skills", "knowledge", "strong",
    "proficient", "responsible", "working", "work", "role", "job", "etc", "including",
    "using", "must", "have", "required", "requirement", "requirements", "responsibilities",
    "about", "us", "overview", "seeking", "platform", "we", "are", "our", "you", "will",
    "degree", "field", "related", "candidate", "ability", "abilities", "demonstrated"
}


def _extract_terms(text: str) -> set[str]:
    """Extract content terms excluding stopwords for domain grounding."""
    tokens = re.findall(r"[a-zA-Z0-9\+\#]+", text.lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 1}


class EvidenceMatcher:
    """
    Matches JD blocks against resume blocks using FAISS retrieval.

    Produces MatchEvidence objects with complete traceability
    and extracts structured comparison parameters based on JD requirements.
    """

    def __init__(
        self,
        embedder: BgeEmbedder,
        retrieval_config: RetrievalConfig,
        thresholds: ScoringThresholds,
    ) -> None:
        self._embedder = embedder
        self._retrieval_config = retrieval_config
        self._thresholds = thresholds
        self._metadata_filter = MetadataFilter(retrieval_config)

    def _compute_effective_match(
        self, dense_sim: float, jd_text: str, res_text: str, jd_sec: SectionType
    ) -> tuple[float, MatchStrength]:
        """
        Compute domain-grounded and calibrated similarity score.

        Eliminates embedding distributional noise (baseline ~0.45-0.58) and verifies
        lexical overlap for technical requirements.
        """
        jd_terms = _extract_terms(jd_text)
        res_terms = _extract_terms(res_text)
        overlap = jd_terms.intersection(res_terms)

        # Domain grounding: if technical section and 0 lexical overlap, dense_sim < 0.70 is noise
        if jd_sec in (SectionType.SKILLS, SectionType.REQUIREMENTS):
            if not overlap and dense_sim < 0.70:
                return 0.0, MatchStrength.NONE
            raw = dense_sim
        elif jd_sec in (SectionType.EXPERIENCE, SectionType.RESPONSIBILITIES):
            if dense_sim < 0.52 and not overlap:
                return 0.0, MatchStrength.NONE
            raw = dense_sim
        elif jd_sec == SectionType.EDUCATION:
            if dense_sim < 0.55 and not overlap:
                return 0.0, MatchStrength.NONE
            raw = dense_sim
        else:
            if dense_sim < 0.48 and not overlap:
                return 0.0, MatchStrength.NONE
            raw = dense_sim

        # Calibrate: 0.45 noise floor -> 0.0; 0.85 -> 1.0
        calibrated = min(1.0, max(0.0, (raw - 0.45) / 0.40))
        if calibrated < self._thresholds.cosine_floor:
            return 0.0, MatchStrength.NONE
        elif calibrated >= self._thresholds.strong_match:
            return round(calibrated, 4), MatchStrength.STRONG
        elif calibrated >= self._thresholds.moderate_match:
            return round(calibrated, 4), MatchStrength.MODERATE
        else:
            return round(calibrated, 4), MatchStrength.WEAK

    def match(
        self,
        jd_blocks: list[Block],
        resume_blocks: list[Block],
    ) -> list[MatchEvidence]:
        """
        Match JD blocks against resume blocks.

        Filters out non-qualification resume content (e.g. contact info)
        and non-requirement JD content (e.g. company overview) for high accuracy.

        Args:
            jd_blocks: Blocks from the job description.
            resume_blocks: Blocks from the resume.

        Returns:
            List of MatchEvidence objects.
        """
        if not jd_blocks or not resume_blocks:
            return []

        # Filter candidate resume blocks: exclude CONTACT sections
        valid_resume_blocks = [
            b for b in resume_blocks if b.section != SectionType.CONTACT
        ] or resume_blocks

        # Embed resume blocks and build FAISS index
        resume_embeddings = self._embedder.embed_blocks(valid_resume_blocks)
        resume_metadata = [
            {
                "block_id": b.block_id,
                "section": b.section.value,
                "document_type": b.document_type.value,
                "text": b.text,
                "line_numbers": b.source_line_numbers,
            }
            for b in valid_resume_blocks
        ]

        faiss_index = FaissIndex(self._retrieval_config, self._embedder.dimension)
        faiss_index.build(resume_embeddings, resume_metadata)

        # Embed JD blocks and search
        jd_embeddings = self._embedder.embed_blocks(jd_blocks)
        search_results = faiss_index.search(jd_embeddings)

        # Build match evidence
        all_evidence: list[MatchEvidence] = []

        for jd_idx, jd_block in enumerate(jd_blocks):
            # Skip non-requirement sections like company overview unless mandatory keywords are present
            if (
                jd_block.section in (SectionType.ABOUT, SectionType.CONTACT)
                and not jd_block.metadata.get("is_mandatory")
                and not self._has_requirement_content(jd_block.text)
            ):
                continue

            results = search_results[jd_idx] if jd_idx < len(search_results) else []

            # Apply metadata filtering
            filtered = self._metadata_filter.filter_results(
                results,
                query_section=jd_block.section,
                required_doc_type=DocumentType.RESUME,
            )

            if not filtered:
                # No match found → missing evidence
                all_evidence.append(
                    MatchEvidence(
                        jd_block_id=jd_block.block_id,
                        jd_text=jd_block.text,
                        jd_section=jd_block.section,
                        jd_line_numbers=jd_block.source_line_numbers,
                        match_strength=MatchStrength.NONE,
                        explanation=f"No matching resume content found for: {jd_block.text[:100]}",
                    )
                )
                continue

            # Create evidence for each match
            for result in filtered:
                meta = result.metadata
                dense_score = result.similarity_score
                score, strength = self._compute_effective_match(
                    dense_score, jd_block.text, meta.get("text", ""), jd_block.section
                )

                evidence = MatchEvidence(
                    jd_block_id=jd_block.block_id,
                    resume_block_id=meta.get("block_id", ""),
                    jd_text=jd_block.text,
                    resume_text=meta.get("text", ""),
                    similarity_score=score,
                    jd_section=jd_block.section,
                    resume_section=SectionType(meta.get("section", "unknown")),
                    jd_line_numbers=jd_block.source_line_numbers,
                    resume_line_numbers=meta.get("line_numbers", []),
                    match_strength=strength,
                    explanation=self._build_explanation(
                        jd_block, meta, score, strength
                    ),
                    uncertainty=(strength == MatchStrength.WEAK),
                )
                all_evidence.append(evidence)

        logger.info(
            "Matched %d JD blocks → %d evidence items",
            len(jd_blocks),
            len(all_evidence),
        )
        return all_evidence

    def extract_comparison_parameters(
        self,
        jd_blocks: list[Block],
        evidence: list[MatchEvidence],
        mandatory_reqs: list[MandatoryRequirement] | None = None,
        resume_blocks: list[Block] | None = None,
    ) -> list[ComparisonParameter]:
        """
        Extract comparison parameters based on JD requirements and evaluate
        whether each parameter is found in the candidate's resume with pinpoint accuracy.

        Filters out non-requirement sections (ABOUT, CONTACT, pure headings) from JD,
        pinpoints the exact matching sentence and line numbers in the resume,
        and accurately classifies each requirement as FOUND, PARTIAL, or NOT_FOUND.
        """
        mandatory_block_ids = {r.jd_block_id for r in (mandatory_reqs or [])}
        resume_blocks_by_id = {b.block_id: b for b in (resume_blocks or [])}

        # Filter JD blocks to only actual requirements
        filtered_jd_blocks: list[Block] = []
        for b in jd_blocks:
            text_clean = b.text.strip()
            text_lower = text_clean.lower().rstrip(":")

            # Skip pure section heading blocks (e.g. "Education", "Requirements")
            if len(text_clean.split()) <= 2 or text_lower in _SECTION_ALIASES:
                continue

            # Skip company marketing / intro lines
            if text_lower.startswith(("about us", "role overview", "company overview", "who we are")):
                continue

            if b.section in (
                SectionType.REQUIREMENTS,
                SectionType.SKILLS,
                SectionType.RESPONSIBILITIES,
                SectionType.EDUCATION,
                SectionType.EXPERIENCE,
                SectionType.CERTIFICATIONS,
            ):
                filtered_jd_blocks.append(b)
            elif (
                b.block_id in mandatory_block_ids
                or b.metadata.get("is_mandatory")
                or self._has_requirement_content(b.text)
            ):
                if b.section not in (SectionType.ABOUT, SectionType.CONTACT):
                    filtered_jd_blocks.append(b)

        # Fallback if all were somehow filtered
        if not filtered_jd_blocks:
            filtered_jd_blocks = [
                b for b in jd_blocks
                if b.section not in (SectionType.ABOUT, SectionType.CONTACT)
                and len(b.text.strip().split()) > 2
            ] or jd_blocks

        parameters: list[ComparisonParameter] = []

        for block in filtered_jd_blocks:
            is_mandatory = (
                block.block_id in mandatory_block_ids
                or block.metadata.get("is_mandatory", False)
            )

            # Determine category
            if is_mandatory:
                category = "mandatory"
            elif block.section == SectionType.SKILLS:
                category = "skill"
            elif block.section in (SectionType.EXPERIENCE, SectionType.RESPONSIBILITIES):
                category = "experience"
            elif block.section in (SectionType.EDUCATION, SectionType.CERTIFICATIONS):
                category = "education"
            elif block.section == SectionType.RESPONSIBILITIES:
                category = "responsibility"
            else:
                category = "requirement"

            # Find matching evidence for this JD block
            block_evidence = [
                e for e in evidence
                if e.jd_block_id == block.block_id
                and e.match_strength != MatchStrength.NONE
                and e.resume_section != SectionType.CONTACT
            ]

            if block_evidence:
                best = max(block_evidence, key=lambda e: e.similarity_score)
                sim = best.similarity_score
                matched_text = best.resume_text
                matched_lines = best.resume_line_numbers

                # Pinpoint exact matching sentence inside the matched resume block
                matched_block = resume_blocks_by_id.get(best.resume_block_id)
                if matched_block and len(matched_block.sentences) > 1:
                    candidates = [
                        s for s in matched_block.sentences
                        if len(s.text.strip().split()) > 2
                        and s.text.strip().lower().rstrip(":") not in _SECTION_ALIASES
                    ] or matched_block.sentences

                    if len(candidates) == 1:
                        matched_text = candidates[0].text
                        matched_lines = candidates[0].source_line_numbers
                    elif len(candidates) > 1:
                        try:
                            req_emb = self._embedder.embed_texts([block.text])
                            sent_embs = self._embedder.embed_texts([s.text for s in candidates])
                            sent_sims = (sent_embs @ req_emb.T).flatten()
                            best_sent_idx = int(np.argmax(sent_sims))
                            matched_text = candidates[best_sent_idx].text
                            matched_lines = candidates[best_sent_idx].source_line_numbers
                            raw_sent_sim = float(sent_sims[best_sent_idx])
                            calibrated_sent_sim, _ = self._compute_effective_match(
                                raw_sent_sim, block.text, matched_text, block.section
                            )
                            # Use higher of sentence or block similarity for accuracy
                            sim = max(sim, calibrated_sent_sim)
                        except Exception as e:
                            logger.debug("Sentence pinpointing fallback: %s", e)

                # Accurate classification of whether found in resume
                if sim >= 0.65:
                    status = "found"
                    status_desc = "Found in resume with strong evidence"
                elif sim >= 0.40:
                    status = "partial"
                    status_desc = "Partially found in resume"
                else:
                    status = "not_found"
                    status_desc = "Low similarity / requirement not met"

                confidence = round(min(sim * 100.0, 100.0), 1)

                param = ComparisonParameter(
                    parameter_id=f"param_{len(parameters)+1:03d}",
                    category=category,
                    requirement_text=block.text,
                    is_mandatory=is_mandatory,
                    status=status,
                    confidence_score=confidence,
                    similarity_score=round(sim, 4),
                    matched_resume_text=matched_text if status != "not_found" else "",
                    matched_resume_lines=matched_lines if status != "not_found" else [],
                    source_jd_lines=block.source_line_numbers,
                    match_strength=best.match_strength if status != "not_found" else MatchStrength.NONE,
                    explanation=(
                        f"{status_desc} (score: {confidence}%). {best.explanation}"
                        if status != "not_found"
                        else f"Requirement not met in resume (score {confidence}% below threshold)."
                    ),
                    jd_section=block.section,
                    resume_section=best.resume_section if status != "not_found" else SectionType.UNKNOWN,
                )
            else:
                param = ComparisonParameter(
                    parameter_id=f"param_{len(parameters)+1:03d}",
                    category=category,
                    requirement_text=block.text,
                    is_mandatory=is_mandatory,
                    status="not_found",
                    confidence_score=0.0,
                    similarity_score=0.0,
                    matched_resume_text="",
                    matched_resume_lines=[],
                    source_jd_lines=block.source_line_numbers,
                    match_strength=MatchStrength.NONE,
                    explanation=f"No matching evidence found in resume for requirement: '{block.text[:100]}...'",
                    jd_section=block.section,
                    resume_section=SectionType.UNKNOWN,
                )

            parameters.append(param)

        return parameters

    def _has_requirement_content(self, text: str) -> bool:
        """Check if text contains requirement-related terminology."""
        keywords = (
            "experience", "skill", "proficien", "knowledge", "degree",
            "bachelor", "master", "docker", "python", "cloud", "aws",
            "gcp", "kubernetes", "database", "sql", "microservice",
            "architecture", "develop", "manage", "lead", "years",
            "must", "required", "essential", "familiar", "preferred",
        )
        lower = text.lower()
        return any(k in lower for k in keywords)

    def _classify_strength(self, score: float) -> MatchStrength:
        """Classify match strength based on similarity score thresholds."""
        if score >= self._thresholds.strong_match:
            return MatchStrength.STRONG
        elif score >= self._thresholds.moderate_match:
            return MatchStrength.MODERATE
        elif score >= self._thresholds.weak_match:
            return MatchStrength.WEAK
        else:
            return MatchStrength.NONE

    def _build_explanation(
        self,
        jd_block: Block,
        resume_meta: dict,
        score: float,
        strength: MatchStrength,
    ) -> str:
        """Build a human-readable explanation for the match."""
        parts = [
            f"Match strength: {strength.value} (similarity: {score:.3f})",
            f"JD section: {jd_block.section.value}",
            f"Resume section: {resume_meta.get('section', 'unknown')}",
        ]
        if jd_block.section.value == resume_meta.get("section"):
            parts.append("✓ Same section match (boosted)")
        else:
            parts.append("△ Cross-section match")
        return " | ".join(parts)
