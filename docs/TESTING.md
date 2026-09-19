# Testing Guide — Resume Matcher

The Resume Matcher project maintains a comprehensive test suite of **99 automated tests** covering unit components, end-to-end integration flows, edge cases, and performance benchmarks.

---

## Test Organization

```text
tests/
├── conftest.py                   # Shared session fixtures and synthetic PDF/DOCX generators
├── benchmark.py                  # Full-system performance and scalability benchmark suite
├── unit/                         # Fast isolated unit tests (73 tests)
│   ├── test_cleaner.py
│   ├── test_config.py
│   ├── test_docx_extractor.py
│   ├── test_pdf_extractor.py
│   ├── test_scoring.py
│   ├── test_section_detector.py
│   ├── test_sentence_splitter.py
│   └── test_validator.py
├── integration/                  # End-to-end pipeline tests (7 tests)
│   ├── test_ingestion_pipeline.py
│   ├── test_matching_pipeline.py
│   └── test_end_to_end.py
└── edge_cases/                   # Corrupted, empty, and technical term tests (12 tests)
    ├── test_corrupted_docs.py
    ├── test_empty_files.py
    └── test_technical_terms.py
```

---

## Running Tests

### 1. Run All Standard Tests (Fast)

```bash
pytest tests/unit tests/integration tests/edge_cases -v
```

### 2. Run Only Unit Tests

```bash
pytest tests/unit -v
```

### 3. Run Integration Tests

```bash
pytest tests/integration -v
```

### 4. Run Edge Case Tests

```bash
pytest tests/edge_cases -v
```

### 5. Run Full Benchmark Test Suite

```bash
# Standalone CLI mode with rich formatted console report:
python tests/benchmark.py --iterations 5

# Pytest mode with strict SLA assertions:
pytest tests/benchmark.py -v
```

### 6. Generate Code Coverage Report

```bash
pytest tests/unit tests/integration tests/edge_cases --cov=resume_matcher --cov-report=term-missing
```
