"""
Comprehensive Performance and Scalability Benchmark Suite for Resume Matcher.

This module benchmarks every layer of the system:
1. Document Ingestion & Preprocessing (PDF & DOCX parsing, cleaning, segmentation)
2. Semantic Embeddings (BGE model latency, throughput, batch scaling)
3. FAISS Retrieval & Vector Indexing (Index creation, top-K search latency, QPS)
4. Multi-Component Scoring & Evidence Matching (Skills, Experience, Education, Semantic)
5. End-to-End Pipeline (Cold start, warm execution, batch scaling)
6. Memory Footprint & Resource Stability (Baseline, peak RSS, leak detection)
7. Determinism & Auditability Verification (Score consistency across runs)

Usage:
    # Run as a standalone benchmark with full console report:
    python -m tests.benchmark
    python tests/benchmark.py --iterations 5 --output-dir output/benchmarks

    # Run via pytest:
    pytest tests/benchmark.py -v
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import resource
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from resume_matcher.config import AppConfig, load_config
from resume_matcher.domain.enums import DocumentType
from resume_matcher.domain.models import Block, ProcessedDocument
from resume_matcher.embeddings.bge_embedder import BgeEmbedder
from resume_matcher.matching.evidence_matcher import EvidenceMatcher
from resume_matcher.matching.mandatory_handler import MandatoryHandler
from resume_matcher.retrieval.faiss_index import FaissIndex
from resume_matcher.scoring.component_scorers import (
    EducationScorer,
    SemanticMatchScorer,
    SkillScorer,
    WorkExperienceScorer,
)
from resume_matcher.scoring.final_scorer import FinalScorer
from resume_matcher.services.ingestion_service import IngestionService
from resume_matcher.services.matching_service import MatchingService
from resume_matcher.services.pipeline_service import PipelineService

# ---------------------------------------------------------------------------
# Helpers & Data Models
# ---------------------------------------------------------------------------


def get_current_memory_mb() -> float:
    """Return current process peak RSS in MB."""
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    if platform.system() == "Darwin":
        # macOS returns bytes
        return rusage.ru_maxrss / (1024 * 1024)
    # Linux returns kilobytes
    return rusage.ru_maxrss / 1024


@dataclass
class BenchmarkStat:
    """Statistical summary for a benchmark metric."""

    name: str
    unit: str
    iterations: int
    mean: float
    median: float
    p95: float
    p99: float
    min_val: float
    max_val: float
    std_dev: float
    throughput: float | None = None
    throughput_unit: str | None = None

    @classmethod
    def from_measurements(
        cls,
        name: str,
        unit: str,
        values: list[float],
        item_count_per_run: int = 1,
        throughput_unit: str = "items/s",
    ) -> BenchmarkStat:
        if not values:
            return cls(name, unit, 0, 0, 0, 0, 0, 0, 0, 0)
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        mean_val = statistics.mean(sorted_vals)
        med_val = statistics.median(sorted_vals)
        p95_val = sorted_vals[int(np.ceil(0.95 * n)) - 1]
        p99_val = sorted_vals[int(np.ceil(0.99 * n)) - 1]
        min_v = sorted_vals[0]
        max_v = sorted_vals[-1]
        std_v = statistics.stdev(sorted_vals) if n > 1 else 0.0

        throughput = None
        if mean_val > 0:
            # throughput = items per second (assuming unit is seconds or milliseconds)
            if unit == "ms":
                throughput = (item_count_per_run / (mean_val / 1000.0))
            elif unit == "s":
                throughput = item_count_per_run / mean_val

        return cls(
            name=name,
            unit=unit,
            iterations=n,
            mean=round(mean_val, 4),
            median=round(med_val, 4),
            p95=round(p95_val, 4),
            p99=round(p99_val, 4),
            min_val=round(min_v, 4),
            max_val=round(max_v, 4),
            std_dev=round(std_v, 4),
            throughput=round(throughput, 2) if throughput else None,
            throughput_unit=throughput_unit if throughput else None,
        )


@dataclass
class BenchmarkSuiteResult:
    """Complete results for the benchmark run."""

    system_info: dict[str, Any]
    benchmarks: dict[str, Any] = field(default_factory=dict)
    memory_profile: dict[str, Any] = field(default_factory=dict)
    sla_verdicts: dict[str, bool] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Benchmark Implementations
# ---------------------------------------------------------------------------


class BenchmarkRunner:
    """Executes and logs all system benchmarks."""

    def __init__(self, config: AppConfig | None = None, iterations: int = 5):
        self.config = config or load_config()
        self.iterations = iterations
        self.sample_jd_path = Path("data/sample_jd.pdf")
        self.sample_resume_path = Path("data/sample_resume.pdf")

        # Verify sample data exists
        if not self.sample_jd_path.exists() or not self.sample_resume_path.exists():
            from scripts.create_samples import create_sample_jd, create_sample_resume
            create_sample_jd(str(self.sample_jd_path))
            create_sample_resume(str(self.sample_resume_path))

    def get_system_info(self) -> dict[str, Any]:
        """Gather host environment information."""
        device = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
        except Exception:
            pass

        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_architecture": platform.machine(),
            "processor": platform.processor(),
            "embedding_model": self.config.embedding.model_name,
            "embedding_dimension": self.config.embedding.dimension,
            "device": device,
            "iterations": self.iterations,
        }

    # --- 1. Ingestion Benchmark ---
    def benchmark_ingestion(self) -> dict[str, Any]:
        """Benchmark PDF parsing, text cleaning, and block segmentation."""
        ingestion = IngestionService(self.config)
        jd_times: list[float] = []
        resume_times: list[float] = []

        # Warmup
        ingestion.ingest(self.sample_jd_path, DocumentType.JD)
        ingestion.ingest(self.sample_resume_path, DocumentType.RESUME)

        for _ in range(self.iterations):
            # JD Ingestion
            t0 = time.perf_counter()
            jd_doc = ingestion.ingest(self.sample_jd_path, DocumentType.JD)
            jd_times.append((time.perf_counter() - t0) * 1000.0)

            # Resume Ingestion
            t0 = time.perf_counter()
            resume_doc = ingestion.ingest(self.sample_resume_path, DocumentType.RESUME)
            resume_times.append((time.perf_counter() - t0) * 1000.0)

        jd_stat = BenchmarkStat.from_measurements(
            "JD Ingestion (PDF -> Blocks)", "ms", jd_times, 1, "docs/s"
        )
        resume_stat = BenchmarkStat.from_measurements(
            "Resume Ingestion (PDF -> Blocks)", "ms", resume_times, 1, "docs/s"
        )

        return {
            "jd_ingestion": asdict(jd_stat),
            "resume_ingestion": asdict(resume_stat),
            "jd_blocks_count": len(jd_doc.blocks),
            "resume_blocks_count": len(resume_doc.blocks),
        }

    # --- 2. Embedding Benchmark ---
    def benchmark_embedding(self) -> dict[str, Any]:
        """Benchmark sentence-transformers BGE embedding latency and throughput."""
        embedder = BgeEmbedder(self.config.embedding)

        # Sample texts representing resume/JD bullets
        sample_sentences = [
            "Designed and implemented high-throughput microservices using Python and Kafka.",
            "5+ years of software development experience with cloud platforms.",
            "Proficient in PostgreSQL database optimization and Redis caching layer.",
            "Mentored junior engineers and conducted technical code reviews.",
            "Deployed scalable containerized applications to Kubernetes clusters on AWS.",
            "Master's degree in Computer Science or related engineering discipline.",
            "Deep understanding of distributed systems and event-driven architecture.",
            "Implemented automated CI/CD deployment pipelines using GitHub Actions.",
        ]

        # Warmup
        _ = embedder.embed_texts(["Warmup query"])
        _ = embedder.embed_texts(sample_sentences)

        # Single Query Latency
        query_times: list[float] = []
        vec = None
        for _ in range(self.iterations * 2):
            t0 = time.perf_counter()
            vec = embedder.embed_texts(["Proficient in Python and microservices architecture"])[0]
            query_times.append((time.perf_counter() - t0) * 1000.0)

        single_query_stat = BenchmarkStat.from_measurements(
            "Single Query Embedding", "ms", query_times, 1, "queries/s"
        )

        # Batch Encoding Throughput across batch sizes
        batch_results: dict[str, Any] = {}
        for batch_size in [1, 8, 16, 32]:
            texts = (sample_sentences * ((batch_size // len(sample_sentences)) + 1))[:batch_size]
            batch_times: list[float] = []

            for _ in range(self.iterations):
                t0 = time.perf_counter()
                vecs = embedder.embed_texts(texts)
                batch_times.append((time.perf_counter() - t0) * 1000.0)

            stat = BenchmarkStat.from_measurements(
                f"Batch Encode (size={batch_size})", "ms", batch_times, batch_size, "sentences/s"
            )
            batch_results[f"batch_{batch_size}"] = asdict(stat)

        return {
            "single_query": asdict(single_query_stat),
            "batch_scaling": batch_results,
            "vector_dimension": len(vec),
            "normalized_check": bool(np.isclose(np.linalg.norm(vec), 1.0, atol=1e-4)),
        }

    # --- 3. FAISS Retrieval Benchmark ---
    def benchmark_faiss(self) -> dict[str, Any]:
        """Benchmark FAISS index creation, vector addition, and top-K search."""
        dim = self.config.embedding.dimension
        retrieval_cfg = self.config.retrieval
        indexer = FaissIndex(retrieval_cfg, dimension=dim)

        # Generate synthetic normalized vectors representing candidate resume blocks
        num_vectors = 500
        rng = np.random.default_rng(42)
        raw_vectors = rng.standard_normal((num_vectors, dim)).astype(np.float32)
        faiss_vectors = raw_vectors / np.linalg.norm(raw_vectors, axis=1, keepdims=True)

        # Index Build Time
        t0 = time.perf_counter()
        indexer.build(faiss_vectors, [{"id": f"doc_{i}"} for i in range(num_vectors)])
        build_time_ms = (time.perf_counter() - t0) * 1000.0

        # Query Search Latency across top-K
        search_results: dict[str, Any] = {}
        query_vectors = faiss_vectors[:1]  # shape (1, dim)

        for k in [1, 3, 5, 10]:
            k_times: list[float] = []
            for _ in range(self.iterations * 10):
                t0 = time.perf_counter()
                results = indexer.search(query_vectors, top_k=k)
                k_times.append((time.perf_counter() - t0) * 1000.0)

            stat = BenchmarkStat.from_measurements(
                f"FAISS Search (k={k}, index_size={num_vectors})",
                "ms",
                k_times,
                1,
                "searches/s",
            )
            search_results[f"top_{k}"] = asdict(stat)

        return {
            "index_build_ms": round(build_time_ms, 3),
            "index_size": num_vectors,
            "search_latency": search_results,
        }

    # --- 4. Scoring & Evidence Matcher Benchmark ---
    def benchmark_scoring(self) -> dict[str, Any]:
        """Benchmark evidence matching, mandatory requirement checks, and scorers."""
        ingestion = IngestionService(self.config)
        jd_doc = ingestion.ingest(self.sample_jd_path, DocumentType.JD)
        resume_doc = ingestion.ingest(self.sample_resume_path, DocumentType.RESUME)

        embedder = BgeEmbedder(self.config.embedding)
        evidence_matcher = EvidenceMatcher(
            embedder, self.config.retrieval, self.config.scoring.thresholds
        )
        mandatory_handler = MandatoryHandler()
        skill_scorer = SkillScorer()
        exp_scorer = WorkExperienceScorer()
        edu_scorer = EducationScorer()
        semantic_scorer = SemanticMatchScorer()
        final_scorer = FinalScorer(self.config.scoring.weights)

        # Pre-compute evidence once for isolated scoring benchmark
        evidence = evidence_matcher.match(jd_doc.blocks, resume_doc.blocks)
        mandatory_blocks = mandatory_handler.detect_mandatory_blocks(jd_doc.blocks)

        scoring_times: list[float] = []
        for _ in range(self.iterations * 10):
            t0 = time.perf_counter()
            mandatory_reqs = mandatory_handler.evaluate_mandatory(mandatory_blocks, evidence)
            weights = self.config.scoring.weights
            comp1 = skill_scorer.score(evidence, jd_doc.blocks, weights.skill)
            comp2 = exp_scorer.score(evidence, jd_doc.blocks, weights.work_experience)
            comp3 = edu_scorer.score(evidence, jd_doc.blocks, weights.education)
            comp4 = semantic_scorer.score(evidence, jd_doc.blocks, weights.semantic_match)
            candidate_score = final_scorer.calculate(
                components=[comp1, comp2, comp3, comp4],
                mandatory_requirements=mandatory_reqs,
                candidate_id=resume_doc.meta.document_id,
                jd_id=jd_doc.meta.document_id,
            )
            scoring_times.append((time.perf_counter() - t0) * 1000.0)

        stat = BenchmarkStat.from_measurements(
            "Scoring & Evaluation", "ms", scoring_times, 1, "matches/s"
        )

        return {
            "scoring_stat": asdict(stat),
            "final_score": candidate_score.final_score,
            "evidence_count": len(evidence),
            "mandatory_count": len(mandatory_reqs),
        }

    # --- 5. End-to-End Pipeline Benchmark ---
    def benchmark_pipeline(self) -> dict[str, Any]:
        """Benchmark complete pipeline execution: cold start, warm match, and batch throughput."""
        output_dir = Path("output/benchmark_run")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Cold Start Pipeline initialization
        t0 = time.perf_counter()
        pipeline = PipelineService(self.config)
        first_report = pipeline.process_single(
            self.sample_jd_path, self.sample_resume_path, output_dir
        )
        cold_start_time = (time.perf_counter() - t0) * 1000.0

        # Warm Pipeline Single Matches
        warm_times: list[float] = []
        for _ in range(self.iterations):
            t0 = time.perf_counter()
            pipeline.process_single(self.sample_jd_path, self.sample_resume_path, output_dir)
            warm_times.append((time.perf_counter() - t0) * 1000.0)

        single_stat = BenchmarkStat.from_measurements(
            "Pipeline Single Match (Warm)", "ms", warm_times, 1, "resumes/s"
        )

        # Batch Scaling Simulation (1 JD vs N Resumes)
        # Using the sample resume replicated N times to simulate high-throughput recruiter load
        batch_results: dict[str, Any] = {}
        for count in [5, 10, 20]:
            resume_list = [self.sample_resume_path] * count
            t0 = time.perf_counter()
            reports = pipeline.process_batch(self.sample_jd_path, resume_list, output_dir)
            total_duration_sec = time.perf_counter() - t0
            throughput_per_sec = count / total_duration_sec
            throughput_per_min = throughput_per_sec * 60.0

            batch_results[f"{count}_resumes"] = {
                "count": count,
                "total_time_seconds": round(total_duration_sec, 3),
                "resumes_per_second": round(throughput_per_sec, 2),
                "resumes_per_minute": round(throughput_per_min, 1),
                "avg_per_resume_ms": round((total_duration_sec / count) * 1000.0, 2),
            }

        return {
            "cold_start_ms": round(cold_start_time, 2),
            "warm_single_match": asdict(single_stat),
            "batch_scaling": batch_results,
        }

    # --- 6. Determinism & Stability Benchmark ---
    def benchmark_determinism(self) -> dict[str, Any]:
        """Verify that repeated matching yields 100% deterministic, auditable scores."""
        output_dir = Path("output/benchmark_determinism")
        output_dir.mkdir(parents=True, exist_ok=True)
        pipeline = PipelineService(self.config)

        scores: list[float] = []
        component_scores: dict[str, list[float]] = {
            "skill": [],
            "work_experience": [],
            "education": [],
            "semantic_match": [],
        }

        for _ in range(5):
            report_path = pipeline.process_single(
                self.sample_jd_path, self.sample_resume_path, output_dir
            )
            with open(report_path) as f:
                data = json.load(f)
            cand_score = data["candidate_score"]
            scores.append(cand_score["final_score"])
            for comp in cand_score["components"]:
                component_scores[comp["name"]].append(comp["raw_score"])

        score_std_dev = statistics.stdev(scores) if len(scores) > 1 else 0.0
        is_deterministic = bool(score_std_dev == 0.0 and len(set(scores)) == 1)

        return {
            "is_deterministic": is_deterministic,
            "score_variance": score_std_dev,
            "final_score": scores[0] if scores else None,
            "sample_scores": scores,
            "component_consistency": {
                name: bool(len(set(vals)) == 1) for name, vals in component_scores.items()
            },
        }

    # --- 7. Memory & Resource Profile ---
    def benchmark_memory(self) -> dict[str, Any]:
        """Profile memory footprint and assess leak resistance across iterations."""
        mem_start = get_current_memory_mb()
        pipeline = PipelineService(self.config)

        # Run multiple iterations to test for memory bloat
        output_dir = Path("output/benchmark_mem")
        output_dir.mkdir(parents=True, exist_ok=True)

        for _ in range(10):
            pipeline.process_single(self.sample_jd_path, self.sample_resume_path, output_dir)

        mem_end = get_current_memory_mb()
        mem_delta = mem_end - mem_start

        return {
            "baseline_rss_mb": round(mem_start, 2),
            "peak_rss_mb": round(mem_end, 2),
            "memory_delta_mb": round(mem_delta, 2),
            "leak_suspected": bool(mem_delta > 50.0),  # Alert if > 50MB unexplained growth
        }

    # --- Full Suite Execution ---
    def run_all(self) -> BenchmarkSuiteResult:
        """Run all benchmark modules and compile report."""
        print("\n" + "=" * 80)
        print("  RESUME MATCHER — COMPREHENSIVE PERFORMANCE & SCALABILITY BENCHMARK")
        print("=" * 80)

        sys_info = self.get_system_info()
        print(f"Platform:      {sys_info['platform']}")
        print(f"Python:        {sys_info['python_version']}")
        print(f"Architecture:  {sys_info['cpu_architecture']}")
        print(f"Model:         {sys_info['embedding_model']} ({sys_info['embedding_dimension']}d)")
        print(f"Iterations:    {self.iterations}")
        print("-" * 80)

        results = BenchmarkSuiteResult(system_info=sys_info)

        print("\n[1/6] Benchmarking Document Ingestion & Segmentation...")
        results.benchmarks["ingestion"] = self.benchmark_ingestion()
        jd_ing = results.benchmarks["ingestion"]["jd_ingestion"]
        res_ing = results.benchmarks["ingestion"]["resume_ingestion"]
        print(f"  -> JD Ingestion:     {jd_ing['mean']:.2f} ms (p95: {jd_ing['p95']:.2f} ms, {jd_ing['throughput']} docs/s)")
        print(f"  -> Resume Ingestion: {res_ing['mean']:.2f} ms (p95: {res_ing['p95']:.2f} ms, {res_ing['throughput']} docs/s)")

        print("\n[2/6] Benchmarking BGE Embedding Generation...")
        results.benchmarks["embedding"] = self.benchmark_embedding()
        emb = results.benchmarks["embedding"]
        print(f"  -> Single Query:     {emb['single_query']['mean']:.2f} ms (p95: {emb['single_query']['p95']:.2f} ms)")
        for b_name, b_val in emb["batch_scaling"].items():
            print(f"  -> {b_val['name']}: {b_val['mean']:.2f} ms -> {b_val['throughput']} sentences/s")

        print("\n[3/6] Benchmarking FAISS Vector Indexing & Retrieval...")
        results.benchmarks["faiss"] = self.benchmark_faiss()
        f_res = results.benchmarks["faiss"]
        print(f"  -> Index Build (500 vectors): {f_res['index_build_ms']:.2f} ms")
        for k_name, k_val in f_res["search_latency"].items():
            print(f"  -> {k_val['name']}: {k_val['mean']:.4f} ms -> {k_val['throughput']} queries/s")

        print("\n[4/6] Benchmarking Multi-Component Scoring Engine...")
        results.benchmarks["scoring"] = self.benchmark_scoring()
        sc = results.benchmarks["scoring"]["scoring_stat"]
        print(f"  -> Scoring Latency:  {sc['mean']:.2f} ms (p95: {sc['p95']:.2f} ms, {sc['throughput']} matches/s)")
        print(f"  -> Final Score:      {results.benchmarks['scoring']['final_score']:.2f}")

        print("\n[5/6] Benchmarking End-to-End Pipeline & Batch Scaling...")
        results.benchmarks["pipeline"] = self.benchmark_pipeline()
        pipe = results.benchmarks["pipeline"]
        print(f"  -> Cold Start:       {pipe['cold_start_ms']:.2f} ms")
        print(f"  -> Warm Single:      {pipe['warm_single_match']['mean']:.2f} ms (p95: {pipe['warm_single_match']['p95']:.2f} ms)")
        for b_key, b_info in pipe["batch_scaling"].items():
            print(f"  -> Batch ({b_info['count']} resumes): {b_info['total_time_seconds']:.2f}s -> {b_info['resumes_per_minute']} resumes/min ({b_info['avg_per_resume_ms']:.1f} ms/resume)")

        print("\n[6/6] Benchmarking Determinism & Memory Footprint...")
        results.benchmarks["determinism"] = self.benchmark_determinism()
        results.memory_profile = self.benchmark_memory()
        det = results.benchmarks["determinism"]
        mem = results.memory_profile
        print(f"  -> Determinism:      {'PASS (100% exact)' if det['is_deterministic'] else 'FAIL'}")
        print(f"  -> Peak Memory RSS:  {mem['peak_rss_mb']:.1f} MB (Delta: {mem['memory_delta_mb']:.1f} MB)")

        # SLA Verification
        sla_pass = True
        warm_p95 = pipe["warm_single_match"]["p95"]
        sla_verdicts = {
            "Warm Single Match < 2000ms": warm_p95 < 2000.0,
            "Embedding Throughput > 10 sent/s": emb["batch_scaling"]["batch_32"]["throughput"] > 10.0,
            "FAISS Search Latency < 1.0ms": f_res["search_latency"]["top_5"]["mean"] < 1.0,
            "100% Deterministic Scores": det["is_deterministic"],
            "No Memory Leak (< 50MB delta)": not mem["leak_suspected"],
        }
        results.sla_verdicts = sla_verdicts

        print("\n" + "=" * 80)
        print("  PERFORMANCE SLA VERIFICATION")
        print("=" * 80)
        for criterion, passed in sla_verdicts.items():
            badge = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {badge:10} | {criterion}")
            if not passed:
                sla_pass = False

        print("=" * 80)
        overall_status = "ALL SLAS PASSED" if sla_pass else "ONE OR MORE SLAS FAILED"
        print(f"  OVERALL VERDICT: {overall_status}")
        print("=" * 80 + "\n")

        return results


# ---------------------------------------------------------------------------
# Report Exporters
# ---------------------------------------------------------------------------


def save_benchmark_reports(results: BenchmarkSuiteResult, output_dir: Path) -> tuple[Path, Path]:
    """Save benchmark results to JSON and Markdown reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "benchmark_results.json"
    md_path = output_dir / "benchmark_report.md"

    # Export JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(asdict(results), f, indent=2)

    # Export Markdown
    b = results.benchmarks
    sys_i = results.system_info
    mem = results.memory_profile

    md_content = f"""# Resume Matcher — System Benchmark Report

**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Host Platform:** {sys_i['platform']} ({sys_i['cpu_architecture']})
**Python:** {sys_i['python_version']}
**Embedding Model:** `{sys_i['embedding_model']}` ({sys_i['embedding_dimension']} dimensions)
**Device:** `{sys_i['device']}`

---

## 1. Executive Summary & SLA Verdicts

| SLA Criterion | Target | Measured | Status |
|---|---|---|---|
| **Warm Single Match Latency (p95)** | < 2,000 ms | **{b['pipeline']['warm_single_match']['p95']} ms** | {'✅ PASS' if results.sla_verdicts.get('Warm Single Match < 2000ms') else '❌ FAIL'} |
| **Embedding Throughput (Batch 32)** | > 10.0 sent/s | **{b['embedding']['batch_scaling']['batch_32']['throughput']} sent/s** | {'✅ PASS' if results.sla_verdicts.get('Embedding Throughput > 10 sent/s') else '❌ FAIL'} |
| **FAISS Vector Search Latency (k=5)** | < 1.00 ms | **{b['faiss']['search_latency']['top_5']['mean']} ms** | {'✅ PASS' if results.sla_verdicts.get('FAISS Search Latency < 1.0ms') else '❌ FAIL'} |
| **Score Determinism & Auditability** | 100% Identical | **Variance = {b['determinism']['score_variance']}** | {'✅ PASS' if results.sla_verdicts.get('100% Deterministic Scores') else '❌ FAIL'} |
| **Memory Leak Resistance** | < 50 MB growth | **Delta = {mem.get('memory_delta_mb', 0)} MB** | {'✅ PASS' if results.sla_verdicts.get('No Memory Leak (< 50MB delta)') else '❌ FAIL'} |

---

## 2. Component Latency Breakdown

| Pipeline Stage | Mean (ms) | Median (ms) | p95 (ms) | Min (ms) | Max (ms) | Throughput |
|---|---|---|---|---|---|---|
| **JD Ingestion (PDF)** | {b['ingestion']['jd_ingestion']['mean']} | {b['ingestion']['jd_ingestion']['median']} | {b['ingestion']['jd_ingestion']['p95']} | {b['ingestion']['jd_ingestion']['min_val']} | {b['ingestion']['jd_ingestion']['max_val']} | {b['ingestion']['jd_ingestion']['throughput']} docs/s |
| **Resume Ingestion (PDF)** | {b['ingestion']['resume_ingestion']['mean']} | {b['ingestion']['resume_ingestion']['median']} | {b['ingestion']['resume_ingestion']['p95']} | {b['ingestion']['resume_ingestion']['min_val']} | {b['ingestion']['resume_ingestion']['max_val']} | {b['ingestion']['resume_ingestion']['throughput']} docs/s |
| **Single Sentence Embedding** | {b['embedding']['single_query']['mean']} | {b['embedding']['single_query']['median']} | {b['embedding']['single_query']['p95']} | {b['embedding']['single_query']['min_val']} | {b['embedding']['single_query']['max_val']} | {b['embedding']['single_query']['throughput']} queries/s |
| **FAISS Top-5 Search (500 vectors)** | {b['faiss']['search_latency']['top_5']['mean']} | {b['faiss']['search_latency']['top_5']['median']} | {b['faiss']['search_latency']['top_5']['p95']} | {b['faiss']['search_latency']['top_5']['min_val']} | {b['faiss']['search_latency']['top_5']['max_val']} | {b['faiss']['search_latency']['top_5']['throughput']} searches/s |
| **Scoring & Evidence Matcher** | {b['scoring']['scoring_stat']['mean']} | {b['scoring']['scoring_stat']['median']} | {b['scoring']['scoring_stat']['p95']} | {b['scoring']['scoring_stat']['min_val']} | {b['scoring']['scoring_stat']['max_val']} | {b['scoring']['scoring_stat']['throughput']} matches/s |
| **End-to-End Single Match (Warm)** | {b['pipeline']['warm_single_match']['mean']} | {b['pipeline']['warm_single_match']['median']} | {b['pipeline']['warm_single_match']['p95']} | {b['pipeline']['warm_single_match']['min_val']} | {b['pipeline']['warm_single_match']['max_val']} | {b['pipeline']['warm_single_match']['throughput']} resumes/s |

---

## 3. Embedding Batch Scaling

| Batch Size | Mean Latency (ms) | Throughput (Sentences/sec) | Speedup vs Batch 1 |
|---|---|---|---|
| **1** | {b['embedding']['batch_scaling']['batch_1']['mean']} ms | {b['embedding']['batch_scaling']['batch_1']['throughput']} sent/s | 1.0x |
| **8** | {b['embedding']['batch_scaling']['batch_8']['mean']} ms | {b['embedding']['batch_scaling']['batch_8']['throughput']} sent/s | {round(b['embedding']['batch_scaling']['batch_8']['throughput'] / max(b['embedding']['batch_scaling']['batch_1']['throughput'], 0.001), 2)}x |
| **16** | {b['embedding']['batch_scaling']['batch_16']['mean']} ms | {b['embedding']['batch_scaling']['batch_16']['throughput']} sent/s | {round(b['embedding']['batch_scaling']['batch_16']['throughput'] / max(b['embedding']['batch_scaling']['batch_1']['throughput'], 0.001), 2)}x |
| **32** | {b['embedding']['batch_scaling']['batch_32']['mean']} ms | {b['embedding']['batch_scaling']['batch_32']['throughput']} sent/s | {round(b['embedding']['batch_scaling']['batch_32']['throughput'] / max(b['embedding']['batch_scaling']['batch_1']['throughput'], 0.001), 2)}x |

---

## 4. Batch Pipeline Throughput (Recruiter Scale)

| Batch Size | Total Duration | Latency / Resume | Throughput (Resumes/Min) |
|---|---|---|---|
| **5 Resumes** | {b['pipeline']['batch_scaling']['5_resumes']['total_time_seconds']}s | {b['pipeline']['batch_scaling']['5_resumes']['avg_per_resume_ms']} ms | **{b['pipeline']['batch_scaling']['5_resumes']['resumes_per_minute']}** resumes/min |
| **10 Resumes** | {b['pipeline']['batch_scaling']['10_resumes']['total_time_seconds']}s | {b['pipeline']['batch_scaling']['10_resumes']['avg_per_resume_ms']} ms | **{b['pipeline']['batch_scaling']['10_resumes']['resumes_per_minute']}** resumes/min |
| **20 Resumes** | {b['pipeline']['batch_scaling']['20_resumes']['total_time_seconds']}s | {b['pipeline']['batch_scaling']['20_resumes']['avg_per_resume_ms']} ms | **{b['pipeline']['batch_scaling']['20_resumes']['resumes_per_minute']}** resumes/min |

---

## 5. Memory & Stability Profile

- **Process Baseline RSS:** `{mem.get('baseline_rss_mb')} MB`
- **Peak RSS:** `{mem.get('peak_rss_mb')} MB`
- **Memory Growth over 10 iterations:** `{mem.get('memory_delta_mb')} MB`
- **Leak Warning:** `{'⚠️ Suspected' if mem.get('leak_suspected') else '✅ None detected'}`
- **Score Determinism:** `{'✅ 100% Identical Across Runs' if b['determinism']['is_deterministic'] else '❌ Inconsistent'}`
"""

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return json_path, md_path


# ---------------------------------------------------------------------------
# Pytest Integration Test Functions
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def benchmark_runner() -> BenchmarkRunner:
    """Session benchmark runner with 3 iterations for fast test suite."""
    return BenchmarkRunner(iterations=3)


@pytest.mark.slow
def test_benchmark_ingestion_performance(benchmark_runner: BenchmarkRunner):
    """Assert PDF ingestion runs within SLA (< 200ms)."""
    res = benchmark_runner.benchmark_ingestion()
    assert res["jd_ingestion"]["mean"] < 200.0, f"JD ingestion too slow: {res['jd_ingestion']['mean']}ms"
    assert res["resume_ingestion"]["mean"] < 200.0, f"Resume ingestion too slow: {res['resume_ingestion']['mean']}ms"


@pytest.mark.slow
def test_benchmark_embedding_performance(benchmark_runner: BenchmarkRunner):
    """Assert BGE embedding runs within SLA (> 5 sentences/s)."""
    res = benchmark_runner.benchmark_embedding()
    assert res["vector_dimension"] == 768
    assert res["normalized_check"] is True
    assert res["batch_scaling"]["batch_8"]["throughput"] > 5.0


@pytest.mark.slow
def test_benchmark_faiss_performance(benchmark_runner: BenchmarkRunner):
    """Assert FAISS top-5 search latency is sub-millisecond (< 5ms on CPU)."""
    res = benchmark_runner.benchmark_faiss()
    assert res["search_latency"]["top_5"]["mean"] < 5.0


@pytest.mark.slow
def test_benchmark_scoring_performance(benchmark_runner: BenchmarkRunner):
    """Assert scoring logic runs in under 50ms."""
    res = benchmark_runner.benchmark_scoring()
    assert res["scoring_stat"]["mean"] < 50.0
    assert 0.0 <= res["final_score"] <= 100.0


@pytest.mark.slow
def test_benchmark_pipeline_single_performance(benchmark_runner: BenchmarkRunner):
    """Assert warm end-to-end match is under 2.0 seconds."""
    res = benchmark_runner.benchmark_pipeline()
    assert res["warm_single_match"]["mean"] < 2000.0, (
        f"Warm match exceeded 2000ms SLA: {res['warm_single_match']['mean']}ms"
    )


def test_benchmark_determinism(benchmark_runner: BenchmarkRunner):
    """Assert deterministic, auditable output with 0 variance."""
    res = benchmark_runner.benchmark_determinism()
    assert res["is_deterministic"] is True
    assert res["score_variance"] == 0.0


def test_benchmark_memory_stability(benchmark_runner: BenchmarkRunner):
    """Assert no memory leak over repeated executions."""
    res = benchmark_runner.benchmark_memory()
    assert not res["leak_suspected"], f"Memory leak detected: delta={res['memory_delta_mb']}MB"


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Run Resume Matcher Performance Benchmark Suite")
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of iterations for statistical benchmarks (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/benchmarks",
        help="Directory to save JSON and Markdown reports (default: output/benchmarks)",
    )
    args = parser.parse_args()

    runner = BenchmarkRunner(iterations=args.iterations)
    results = runner.run_all()

    out_dir = Path(args.output_dir)
    json_path, md_path = save_benchmark_reports(results, out_dir)
    print(f"📊 Benchmark JSON Report saved: {json_path}")
    print(f"📄 Benchmark Markdown Report saved: {md_path}\n")


if __name__ == "__main__":
    main()
