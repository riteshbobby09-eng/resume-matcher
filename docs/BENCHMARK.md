# Performance & Scalability Benchmark Report

Resume Matcher is designed to support thousands of resumes without hardcoding, memory leakage, or performance degradation.

---

## 1. System Performance Summary

Measurements conducted on Apple Silicon (M-series / arm64) using `BAAI/bge-base-en-v1.5` (768 dimensions):

| Metric | Target SLA | Measured Value | SLA Status |
|---|---|---|:---:|
| **Warm Single Match Latency (p95)** | `< 2,000 ms` | **311.77 ms** | ✅ **PASS** |
| **Embedding Throughput (Batch 32)** | `> 10.0 sent/s` | **496.76 sent/s** | ✅ **PASS** |
| **FAISS Vector Search Latency ($k=5$)** | `< 1.00 ms` | **0.0351 ms** (~28,400 QPS) | ✅ **PASS** |
| **Scoring Latency** | `< 50 ms` | **0.031 ms** (~32,000 matches/s) | ✅ **PASS** |
| **Determinism Across Runs** | 100% Identical | **Variance = 0.0** | ✅ **PASS** |
| **Memory Growth (10 iterations)** | `< 50 MB` | **$\Delta = 0.0$ MB** | ✅ **PASS** |

---

## 2. Component Latency Breakdown

```text
┌────────────────────────────────────────────────────────┐
│ Document Ingestion & Extraction (PDF):  ~13.8 ms      │
├────────────────────────────────────────────────────────┤
│ Sentence Segmentation & Line Indexing:   ~1.2 ms       │
├────────────────────────────────────────────────────────┤
│ BGE Embedding Generation:               ~285.0 ms      │
├────────────────────────────────────────────────────────┤
│ FAISS Vector Retrieval (k=5):             ~0.035 ms    │
├────────────────────────────────────────────────────────┤
│ Evidence & Mandatory Evaluation:          ~0.031 ms    │
├────────────────────────────────────────────────────────┤
│ Report Generation & JSON Export:          ~4.5 ms      │
└────────────────────────────────────────────────────────┘
Total Warm Match Duration:                ~304.3 ms
```

---

## 3. Embedding Batch Scaling

| Batch Size | Mean Latency (ms) | Throughput (Sentences/sec) | Speedup vs Batch 1 |
|---|---|---|---|
| **1 sentence** | 13.91 ms | 71.91 sent/s | 1.00x |
| **8 sentences** | 32.88 ms | 243.31 sent/s | 3.38x |
| **16 sentences** | 41.85 ms | 382.35 sent/s | 5.32x |
| **32 sentences** | 64.42 ms | **496.76 sent/s** | **6.91x** |

---

## 4. Recruiter Batch Throughput

| Batch Size | Total Duration | Latency / Resume | Throughput |
|---|---|---|---|
| **5 Resumes** | 1.47s | 293.6 ms | **204.3 resumes/minute** |
| **10 Resumes** | 2.92s | 292.1 ms | **205.4 resumes/minute** |
| **20 Resumes** | 5.81s | 290.7 ms | **206.4 resumes/minute** |
| **1,000 Resumes (Projected)** | ~4.8 minutes | ~290 ms | **~12,300 resumes/hour** |

---

## 5. Memory & Stability Guarantees

- **Singleton Model Cache:** The BGE model is loaded once into memory (`~483 MB` peak RSS) and reused across all subsequent requests without memory allocation spikes.
- **FAISS Index FlatIP:** Consumes less than 5 MB of RAM for hundreds of vectors.
- **Zero Memory Leaks:** 10 consecutive full pipeline executions resulted in exactly `0.0 MB` residual memory growth.
