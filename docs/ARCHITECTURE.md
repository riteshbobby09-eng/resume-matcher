# Architecture

## Data Flow

```
Document (PDF/DOCX)
    │
    ├─→ FileValidator          → pass/fail + messages
    ├─→ PdfExtractor/DocxExt   → List[IndexedLine]     (line_num, page, text)
    ├─→ TextCleaner            → List[IndexedLine]     (cleaned text)
    ├─→ TextNormalizer         → List[IndexedLine]     (preserved tech terms)
    ├─→ LineIndexer            → List[IndexedLine]     (global numbering)
    ├─→ SentenceSplitter       → List[Sentence]        (rule-based splitting)
    ├─→ BlockBuilder           → List[Block]           (meaningful units)
    ├─→ SectionDetector        → List[DetectedSection] (headings found)
    ├─→ ContextAssigner        → List[Block]           (sections assigned)
    ├─→ SectionInferencer      → List[Block]           (gaps filled)
    ├─→ QualityChecker         → QualityReport         (validation)
    ├─→ BgeEmbedder            → ndarray (N×768)       (vectors)
    ├─→ FaissIndex (IndexFlatIP)                       (cosine search)
    ├─→ EvidenceMatcher        → List[MatchEvidence]   (matches)
    ├─→ MandatoryHandler       → mandatory status
    ├─→ ComponentScorers ×4    → ScoreComponents
    ├─→ FinalScorer            → CandidateScore
    └─→ ReportBuilder          → RecruiterReport → JSON
```

## Traceability Chain

Every score can be traced back to the original line in the source document:

```
Document → IndexedLine → Sentence → Block → MatchEvidence → ScoreComponent → CandidateScore → Report
```

## Module Responsibilities

| Module | What it does | What it does NOT do |
|--------|-------------|-------------------|
| `ingestion/` | File validation, raw text extraction | No cleaning, no analysis |
| `preprocessing/` | Encoding fixes, whitespace, term preservation | No sentence splitting |
| `segmentation/` | Line numbering, sentence splitting, block grouping | No section detection |
| `metadata/` | Section detection, inference with confidence | No scoring |
| `embeddings/` | BGE vector generation | No retrieval or matching |
| `retrieval/` | FAISS indexing, similarity search, metadata filtering | No scoring |
| `matching/` | Evidence construction, mandatory tracking | No score calculation |
| `scoring/` | 4-component scoring, weight validation | No report generation |
| `reporting/` | Report assembly, JSON export | No business logic |
| `services/` | Pipeline orchestration | No direct I/O |
| `api/` | HTTP endpoints, file uploads | No business logic |

## Key Design Decisions

1. **FAISS IndexFlatIP** with L2-normalized vectors = cosine similarity
2. **Rule-based sentence splitting** — no spaCy, no ML models for segmentation
3. **Section inference** uses surrounding context + content signals + position priors
4. **No hardcoded skill lists** — section detection uses structural patterns only
5. **Singleton model loading** — BGE model loaded once and reused
6. **In-memory repositories** — designed for easy PostgreSQL swap
