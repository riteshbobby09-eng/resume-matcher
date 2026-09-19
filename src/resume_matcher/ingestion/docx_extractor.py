"""
DOCX text extractor using python-docx.

Extracts text from:
- Paragraphs (preserving order)
- Table cells (concatenated row-wise)

Produces IndexedLine objects with global line numbers.
"""

from __future__ import annotations

import logging
from pathlib import Path

from docx import Document as DocxDocument
from docx.opc.exceptions import PackageNotFoundError
from docx.table import Table

from resume_matcher.domain.models import DocumentMeta, IndexedLine
from resume_matcher.exceptions import CorruptedDocumentError, ExtractionError

logger = logging.getLogger(__name__)


class DocxExtractor:
    """
    Extracts text from DOCX files using python-docx.

    Handles both paragraphs and tables, preserving document order.
    """

    def extract(self, file_path: str | Path, meta: DocumentMeta) -> list[IndexedLine]:
        """
        Extract text from a DOCX file.

        Args:
            file_path: Path to the DOCX file.
            meta: Document metadata.

        Returns:
            List of IndexedLine objects.

        Raises:
            ExtractionError: If the DOCX cannot be read.
            CorruptedDocumentError: If the DOCX structure is damaged.
        """
        path = Path(file_path)
        try:
            doc = DocxDocument(str(path))
        except PackageNotFoundError as exc:
            raise CorruptedDocumentError(
                f"Cannot open DOCX: {meta.sanitized_filename}",
                details={"path": str(path), "error": str(exc)},
            ) from exc
        except Exception as exc:
            raise ExtractionError(
                f"Error opening DOCX: {meta.sanitized_filename}",
                details={"path": str(path), "error": str(exc)},
            ) from exc

        meta.page_count = 1  # DOCX doesn't have reliable page counts without rendering

        lines: list[IndexedLine] = []
        global_line_num = 0

        try:
            # Walk through the document body in order.
            # The body contains paragraphs and tables interleaved.
            for element in doc.element.body:
                tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag

                if tag == "p":
                    # It's a paragraph
                    text = element.text or ""
                    # Also collect text from child runs
                    full_text = self._extract_paragraph_text(element)
                    global_line_num += 1
                    lines.append(
                        IndexedLine(
                            global_line_number=global_line_num,
                            text=full_text,
                            page_number=1,
                            local_line_number=global_line_num,
                            source_filename=meta.source_filename,
                            document_id=meta.document_id,
                            is_empty=(full_text.strip() == ""),
                        )
                    )

                elif tag == "tbl":
                    # It's a table — extract rows as lines
                    table_lines = self._extract_table_lines(element)
                    for tline in table_lines:
                        global_line_num += 1
                        lines.append(
                            IndexedLine(
                                global_line_number=global_line_num,
                                text=tline,
                                page_number=1,
                                local_line_number=global_line_num,
                                source_filename=meta.source_filename,
                                document_id=meta.document_id,
                                is_empty=(tline.strip() == ""),
                            )
                        )
        except Exception as exc:
            raise ExtractionError(
                f"Error extracting content from DOCX: {meta.sanitized_filename}",
                details={"error": str(exc)},
            ) from exc

        logger.info(
            "DOCX extracted: %s — %d lines",
            meta.sanitized_filename,
            len(lines),
        )
        return lines

    def _extract_paragraph_text(self, paragraph_element) -> str:
        """
        Extract full text from a paragraph XML element.

        Concatenates text from all runs (w:r/w:t elements).
        """
        # Use itertext to get all text content
        texts: list[str] = []
        ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        for t_elem in paragraph_element.iter(f"{ns}t"):
            if t_elem.text:
                texts.append(t_elem.text)
        return "".join(texts)

    def _extract_table_lines(self, table_element) -> list[str]:
        """
        Extract text from a table XML element.

        Each row becomes a single line with cells separated by " | ".
        """
        ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        lines: list[str] = []

        for row in table_element.iter(f"{ns}tr"):
            cells: list[str] = []
            for cell in row.iter(f"{ns}tc"):
                cell_texts: list[str] = []
                for t_elem in cell.iter(f"{ns}t"):
                    if t_elem.text:
                        cell_texts.append(t_elem.text)
                cells.append(" ".join(cell_texts).strip())
            row_text = " | ".join(cells)
            if row_text.strip():
                lines.append(row_text)

        return lines
