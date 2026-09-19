"""
PDF text extractor using PyMuPDF.

Extracts text with line-level granularity, preserving:
- Page numbers
- Line numbers (local per page, global across document)
- Reading order (top-to-bottom, left-to-right for multi-column)

Uses page.get_text("dict") for structured extraction.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pymupdf  # PyMuPDF

from resume_matcher.domain.models import DocumentMeta, IndexedLine
from resume_matcher.exceptions import CorruptedDocumentError, ExtractionError

logger = logging.getLogger(__name__)


class PdfExtractor:
    """
    Extracts text from PDF files using PyMuPDF.

    Produces a list of IndexedLine objects, each with a global line number,
    page number, and the raw text content.
    """

    def extract(self, file_path: str | Path, meta: DocumentMeta) -> list[IndexedLine]:
        """
        Extract text from a PDF file.

        Args:
            file_path: Path to the PDF file.
            meta: Document metadata (populated with page_count after extraction).

        Returns:
            List of IndexedLine objects preserving source traceability.

        Raises:
            ExtractionError: If the PDF cannot be opened or read.
            CorruptedDocumentError: If the PDF structure is damaged.
        """
        path = Path(file_path)
        try:
            doc = pymupdf.open(str(path))
        except Exception as exc:
            raise CorruptedDocumentError(
                f"Cannot open PDF: {meta.sanitized_filename}",
                details={"path": str(path), "error": str(exc)},
            ) from exc

        meta.page_count = len(doc)
        lines: list[IndexedLine] = []
        global_line_num = 0

        try:
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                page_lines = self._extract_page_lines(page, page_idx + 1)

                for local_num, text in enumerate(page_lines, start=1):
                    global_line_num += 1
                    lines.append(
                        IndexedLine(
                            global_line_number=global_line_num,
                            text=text,
                            page_number=page_idx + 1,
                            local_line_number=local_num,
                            source_filename=meta.source_filename,
                            document_id=meta.document_id,
                            is_empty=(text.strip() == ""),
                        )
                    )
        except Exception as exc:
            raise ExtractionError(
                f"Error extracting page from PDF: {meta.sanitized_filename}",
                details={"page": page_idx + 1, "error": str(exc)},
            ) from exc
        finally:
            doc.close()

        logger.info(
            "PDF extracted: %s — %d pages, %d lines",
            meta.sanitized_filename,
            meta.page_count,
            len(lines),
        )
        return lines

    def _extract_page_lines(self, page: pymupdf.Page, page_number: int) -> list[str]:
        """
        Extract lines from a single page using structured dict output.

        Sorts blocks and lines spatially (top-to-bottom, left-to-right)
        to handle multi-column layouts correctly.
        """
        try:
            page_dict = page.get_text("dict", sort=True)
        except Exception:
            # Fallback to simple text extraction
            logger.warning("Structured extraction failed for page %d, using fallback", page_number)
            text = page.get_text("text", sort=True)
            return text.split("\n") if text else []

        lines: list[str] = []
        blocks = page_dict.get("blocks", [])

        # Sort blocks by vertical position (y0), then horizontal (x0)
        text_blocks = [b for b in blocks if b.get("type") == 0]  # type 0 = text
        text_blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))

        for block in text_blocks:
            block_lines = block.get("lines", [])
            # Sort lines within block by vertical position
            block_lines.sort(key=lambda ln: (ln["bbox"][1], ln["bbox"][0]))

            for line_data in block_lines:
                spans = line_data.get("spans", [])
                line_text = "".join(span.get("text", "") for span in spans)
                if line_text:  # Include non-empty lines
                    lines.append(line_text)

        return lines
