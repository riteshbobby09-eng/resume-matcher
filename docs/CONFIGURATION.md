# Configuration Guide — Resume Matcher

The Resume Matcher system uses a strictly typed, layered configuration architecture powered by Pydantic and dataclasses.

---

## Configuration Hierarchy & Precedence

Configuration values are resolved in the following priority order (highest to lowest):

1. **Environment Variables** (prefixed with `RESUME_MATCHER_`)
2. **Custom YAML Overrides** (`--config /path/to/custom.yaml`)
3. **Default YAML** (`config/default.yaml`)
4. **Code Defaults** (defined in `resume_matcher.config`)

---

## Configuration Schema

### 1. File Processing (`processing`)

Controls file limits, supported formats, and batching.

```yaml
processing:
  max_file_size_mb: 25          # Maximum allowed file size in MB
  allowed_extensions:           # Accepted file types
    - .pdf
    - .docx
  batch_size: 32                # Chunk size for batch processing
  max_lines_per_document: 10000 # Hard cap on document lines to prevent DoS
```

### 2. Embedding Model (`embedding`)

Controls the SentenceTransformers BGE vector embedding engine.

```yaml
embedding:
  model_name: "BAAI/bge-base-en-v1.5"  # Pre-trained BGE embedding model
  dimension: 768                       # Output vector dimensionality
  normalize: true                      # L2-normalize vectors for cosine similarity
  batch_size: 32                       # Inference batch size
  show_progress: false                 # Disable CLI progress bars in service mode
```

### 3. FAISS Retrieval (`retrieval`)

Controls vector indexing and semantic nearest neighbor retrieval.

```yaml
retrieval:
  top_k: 20                     # Maximum candidate blocks retrieved per query
  similarity_threshold: 0.45    # Minimum cosine similarity threshold (0.0 to 1.0)
  section_match_boost: 0.05     # Additive score bonus for matching within identical sections
```

### 4. Scoring Weights (`scoring.weights`)

Defines the mathematical contribution of each component to the final candidate score. **Weights must sum to exactly 1.0**.

$$\text{Final Score} = (\text{Skill} \times 0.30) + (\text{WorkExp} \times 0.15) + (\text{Education} \times 0.15) + (\text{Semantic} \times 0.40)$$

```yaml
scoring:
  weights:
    skill: 0.30                 # Skills match contribution (30%)
    work_experience: 0.15       # Work experience match contribution (15%)
    education: 0.15             # Education match contribution (15%)
    semantic_match: 0.40        # Overall semantic similarity contribution (40%)
```

### 5. Scoring Thresholds (`scoring.thresholds`)

Categorizes match strength based on cosine similarity scores:

```yaml
scoring:
  thresholds:
    strong_match: 0.75          # Cosine similarity >= 0.75 is a strong match
    moderate_match: 0.55        # Cosine similarity >= 0.55 is a moderate match
    weak_match: 0.35            # Cosine similarity >= 0.35 is a weak match
    mandatory_threshold: 0.50   # Threshold to mark a mandatory requirement as satisfied
```

---

## Environment Variable Overrides

Any setting can be overridden via environment variables using double underscores (`__`) for nested keys:

| Environment Variable | Target Setting | Example Value |
|---|---|---|
| `RESUME_MATCHER_PROCESSING__MAX_FILE_SIZE_MB` | `processing.max_file_size_mb` | `50` |
| `RESUME_MATCHER_EMBEDDING__MODEL_NAME` | `embedding.model_name` | `BAAI/bge-base-en-v1.5` |
| `RESUME_MATCHER_EMBEDDING__BATCH_SIZE` | `embedding.batch_size` | `64` |
| `RESUME_MATCHER_RETRIEVAL__TOP_K` | `retrieval.top_k` | `30` |
| `RESUME_MATCHER_RETRIEVAL__SIMILARITY_THRESHOLD` | `retrieval.similarity_threshold` | `0.50` |
| `RESUME_MATCHER_LOGGING__LEVEL` | `logging.level` | `DEBUG` |
