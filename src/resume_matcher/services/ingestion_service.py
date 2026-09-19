"""
Ingestion service for Resume Matcher.

Orchestrates the full ingestion pipeline:
File → Validate → Extract → Clean → Normalize → Index → Segment → Block → Detect Sections → Infer Sections → Assign Context → Quality Check → ProcessedDocument
"""

from __future__ import annotations

import logging
from pathlib import Path

from resume_matcher.config import AppConfig
from resume_matcher.domain.enums import DocumentType, FileFormat, ProcessingStage
from resume_matcher.domain.models import DocumentMeta, ProcessedDocument
from resume_matcher.ingestion.docx_extractor import DocxExtractor
from resume_matcher.ingestion.pdf_extractor import PdfExtractor
from resume_matcher.ingestion.validator import FileValidator
from resume_matcher.metadata.context_assigner import ContextAssigner
from resume_matcher.metadata.section_detector import SectionDetector
from resume_matcher.metadata.section_inferencer import SectionInferencer
from resume_matcher.preprocessing.cleaner import TextCleaner
from resume_matcher.preprocessing.normalizer import TextNormalizer
from resume_matcher.segmentation.block_builder import BlockBuilder
from resume_matcher.segmentation.line_indexer import LineIndexer
from resume_matcher.segmentation.sentence_splitter import SentenceSplitter
from resume_matcher.validation.quality_checker import QualityChecker

logger = logging.getLogger(__name__)


class IngestionService:
    """
    Orchestrates file ingestion: from raw file to ProcessedDocument.

    Uses dependency injection — all components are passed in.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._validator = FileValidator(config.processing)
        self._pdf_extractor = PdfExtractor()
        self._docx_extractor = DocxExtractor()
        self._cleaner = TextCleaner()
        self._normalizer = TextNormalizer()
        self._line_indexer = LineIndexer()
        self._sentence_splitter = SentenceSplitter()
        self._block_builder = BlockBuilder(config.segmentation)
        self._section_detector = SectionDetector()
        self._section_inferencer = SectionInferencer(config.section_detection)
        self._context_assigner = ContextAssigner()
        self._quality_checker = QualityChecker()

    def ingest(
        self,
        file_path: str | Path,
        document_type: DocumentType,
    ) -> ProcessedDocument:
        """
        Ingest a document file into a ProcessedDocument.

        Full pipeline:
        1. Validate file
        2. Extract text (PDF or DOCX)
        3. Clean and normalize text
        4. Index lines
        5. Split into sentences
        6. Build blocks
        7. Detect section headings
        8. Assign sections to blocks
        9. Infer sections for unassigned blocks
        10. Quality check

        Args:
            file_path: Path to the PDF or DOCX file.
            document_type: Whether this is a JD or resume.

        Returns:
            ProcessedDocument ready for embedding.
        """
        path = Path(file_path)
        logger.info("Ingesting: %s (%s)", path.name, document_type.value)

        # 1. Validate
        validation = self._validator.validate(path)
        if not validation.is_valid:
            logger.error("Validation failed: %s", validation.messages)
            raise ValueError(f"File validation failed: {'; '.join(validation.messages)}")

        # Create document metadata
        meta = DocumentMeta(
            source_filename=path.name,
            sanitized_filename=validation.sanitized_filename,
            document_type=document_type,
            file_format=validation.file_format,
            file_size_bytes=validation.file_size_bytes,
            processing_stage=ProcessingStage.INGESTED,
        )

        # 2. Extract text
        if validation.file_format == FileFormat.PDF:
            lines = self._pdf_extractor.extract(path, meta)
        else:
            lines = self._docx_extractor.extract(path, meta)
        meta.processing_stage = ProcessingStage.EXTRACTED

        # 3. Clean and normalize
        lines = self._cleaner.clean_lines(lines)
        lines = self._cleaner.remove_artifacts(lines)
        lines = self._normalizer.normalize_lines(lines)
        meta.processing_stage = ProcessingStage.CLEANED

        # 4. Index lines
        lines = self._line_indexer.index_lines(lines)
        content_lines = self._line_indexer.get_content_lines(lines)
        meta.processing_stage = ProcessingStage.INDEXED

        # Store raw text
        raw_text = "\n".join(line.text for line in content_lines)

        # 5. Split into sentences
        sentences = self._sentence_splitter.split_lines(
            content_lines, document_id=meta.document_id
        )
        meta.processing_stage = ProcessingStage.SEGMENTED

        # 6. Detect section headings
        detected_sections = self._section_detector.detect_sections(lines)
        meta.processing_stage = ProcessingStage.SECTIONS_DETECTED
        section_lines = {s.line_number for s in detected_sections}

        # 7. Build blocks respecting section boundaries
        blocks = self._block_builder.build_blocks(
            sentences,
            document_id=meta.document_id,
            document_type=document_type,
            total_lines=len(lines),
            section_break_lines=section_lines,
        )
        meta.processing_stage = ProcessingStage.BLOCKS_CREATED

        # 8. Assign sections from detections
        blocks = self._context_assigner.assign_sections_from_detection(
            blocks, detected_sections, len(lines)
        )

        # 9. Infer sections for unassigned blocks
        blocks = self._section_inferencer.infer_sections(blocks)

        # 10. Assign document metadata
        blocks = self._context_assigner.assign_document_metadata(blocks, meta)
        meta.processing_stage = ProcessingStage.METADATA_ASSIGNED

        # Build ProcessedDocument
        doc = ProcessedDocument(
            meta=meta,
            lines=lines,
            sentences=sentences,
            blocks=blocks,
            raw_text=raw_text,
        )

        # 11. Quality check
        doc.quality_report = self._quality_checker.check(doc)
        meta.processing_stage = ProcessingStage.VALIDATED

        logger.info(
            "Ingestion complete: %s — %d lines, %d sentences, %d blocks, %d sections",
            meta.sanitized_filename,
            len(lines),
            len(sentences),
            len(blocks),
            len(detected_sections),
        )

        return doc
