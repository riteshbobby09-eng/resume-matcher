# Resume Matcher

**Deterministic, explainable Resume-JD matching system** using BGE embeddings and FAISS retrieval.

No LLMs, no RAG, no spaCy — just clean, auditable, rule-based processing with semantic matching.

## Quick Start

### 1. Setup

```bash
# Clone and enter project
cd resume_matcher

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Download the BGE model (~440MB, one-time)
python -m scripts.download_model
```

### 2. Run via CLI

```bash
# Single resume vs JD
python -m scripts.run_pipeline --jd path/to/jd.pdf --resume path/to/resume.pdf

# Multiple resumes vs JD
python -m scripts.run_pipeline --jd path/to/jd.pdf --resume-dir path/to/resumes/

# Custom output directory
python -m scripts.run_pipeline --jd jd.pdf --resume resume.pdf --output ./reports/
```

### 3. Run via FastAPI

```bash
# Start the API server
uvicorn resume_matcher.api.app:app --reload --port 8000

# Match via curl
curl -X POST http://localhost:8000/match \
  -F "jd_file=@path/to/jd.pdf" \
  -F "resume_file=@path/to/resume.pdf"
```

### 4. Run Tests

```bash
pytest tests/ -v                    # All tests
pytest tests/unit/ -v               # Unit tests only
pytest tests/ -v --cov=src/resume_matcher  # With coverage
```

## Scoring Formula

```
Final Score = Skill × 0.30 + Work Experience × 0.15 + Education × 0.15 + Semantic Match × 0.40
```

All scores on 0–100 scale.

## Architecture

```
PDF/DOCX → Validate → Extract → Clean → Normalize → Index → Segment → Block
    → Detect Sections → Infer Sections → Quality Check
    → Embed (BGE) → Index (FAISS) → Match → Score → Report → JSON
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for full details.

## Project Structure

```
src/resume_matcher/
├── domain/          # Pure data models and enums
├── schemas/         # Pydantic validation schemas
├── ingestion/       # File validation, PDF/DOCX extraction
├── preprocessing/   # Text cleaning and normalization
├── segmentation/    # Line indexing, sentence splitting, block building
├── metadata/        # Section detection, inference, context assignment
├── validation/      # Quality checks
├── embeddings/      # BGE embedding generation
├── retrieval/       # FAISS indexing and search
├── matching/        # JD-resume evidence matching
├── scoring/         # 4-component scoring system
├── reporting/       # Report generation and export
├── repositories/    # Data access abstractions
├── services/        # Business logic orchestration
└── api/             # FastAPI routes
```

## Technologies Used

| Technology | Purpose |
|-----------|---------|
| PyMuPDF | PDF text extraction |
| python-docx | DOCX text extraction |
| Pydantic v2 | Schema validation |
| Sentence Transformers | BGE embedding model |
| FAISS | Vector similarity search |
| FastAPI | REST API |
| pytest | Testing |
| YAML | Configuration |
