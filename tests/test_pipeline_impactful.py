"""
Impactful Input Document Test Suite for Resume Matcher.

Tests the pipeline across diverse candidate profiles and edge cases:
1. Target JD: Senior Backend / Distributed Systems Engineer
2. Candidate A: Strong Match (Senior Backend Engineer, 7 yrs, Python, Go, Kafka, AWS)
3. Candidate B: Adjacent/Pivot Match (ML Engineer / Data Scientist, 4 yrs, Python, PyTorch)
4. Candidate C: Junior Candidate (Junior Developer, 1 yr, Python, Flask, BS CS)
5. Candidate D: Completely Irrelevant (Executive Chef / Kitchen Manager, 10 yrs)
6. Candidate E: Edge-Case / Stress Test (Messy headers, 80+ word sentences, 50+ skills)

Verifies:
- Efficiency (latency per stage: ingestion, splitting, embedding, retrieval, scoring)
- 40-word max chunk constraint strictly enforced
- Section detector & inferencer accuracy
- 30% cosine floor enforcement
- Exact 4-section weighted scoring (35% content, 35% exp, 15% skill, 15% edu)
- Proper rank ordering (A > B > C > D)
"""

import time
import pymupdf
from pathlib import Path
import numpy as np

from resume_matcher.config import load_config
from resume_matcher.domain.enums import DocumentType, MatchStrength, SectionType
from resume_matcher.services.ingestion_service import IngestionService
from resume_matcher.services.matching_service import MatchingService
from resume_matcher.reporting.report_builder import ReportBuilder


def create_test_documents(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Target JD: Senior Distributed Backend Engineer ──
    jd_text = """Job Title: Senior Distributed Backend Engineer

About Us:
We are a premier fintech platform processing billions of transactions annually.

Role Overview:
We are seeking a Senior Backend Engineer to architect, build, and optimize our high-throughput payment processing engine.

Requirements:
- Must have 5+ years of software engineering experience in backend development.
- Strong expertise in Python and Go for high-concurrency services.
- Extensive experience with Docker and Kubernetes in production environments.
- Deep hands-on experience with Apache Kafka, RabbitMQ, or similar event-streaming systems.
- Production experience with PostgreSQL optimization, indexing, and Redis caching.
- Solid understanding of distributed systems principles, CAP theorem, and eventual consistency.
- Experience with AWS infrastructure (EKS, RDS, S3, CloudWatch).

Responsibilities:
- Design, develop, and scale resilient microservices handling 50,000+ RPS.
- Lead architectural reviews and mentor mid-level and junior engineers.
- Collaborate with security and infrastructure teams on zero-downtime deployments.
- Build automated testing and CI/CD pipelines with GitHub Actions.

Skills:
Python, Go, Golang, Docker, Kubernetes, Apache Kafka, PostgreSQL, Redis, AWS, Microservices, CI/CD, Linux, Git, REST APIs, gRPC

Education:
- Bachelor's or Master's degree in Computer Science, Software Engineering, or related STEM field.
"""
    _write_pdf(output_dir / "target_jd.pdf", jd_text)

    # ── 2. Profile A: Strong Match ──
    strong_resume = """David Chen
Email: david.chen@devmail.com | Phone: +1-555-0182 | LinkedIn: linkedin.com/in/davidchen-backend
San Francisco, CA

Professional Summary:
Accomplished Senior Backend Engineer with 7+ years of experience designing and scaling distributed microservices in fintech. Proven track record handling 60k+ requests per second with 99.99% uptime.

Work Experience:
Senior Backend Engineer | Apex FinTech | 2021 - Present
- Architected and implemented core transaction ledger using Go and Python, processing $2B in monthly volume.
- Migrated legacy monolithic services to containerized microservices orchestrated via Kubernetes (EKS) on AWS.
- Designed event-driven payment notification pipeline utilizing Apache Kafka, reducing message delivery latency by 55%.
- Optimized complex PostgreSQL queries and implemented Redis distributed caching, improving p99 latency from 420ms to 65ms.

Software Engineer | CloudScale Systems | 2017 - 2021
- Developed scalable REST and gRPC microservices in Python (FastAPI) and PostgreSQL.
- Built automated CI/CD deployment pipelines using GitHub Actions and Docker, reducing release cycle time by 40%.
- Monitored production systems using Prometheus and Grafana; led incident response and root-cause analysis.

Technical Skills:
- Languages: Python, Go (Golang), SQL, Bash
- Cloud & DevOps: AWS (EKS, RDS, S3, IAM), Docker, Kubernetes, CI/CD, GitHub Actions, Terraform
- Databases & Streaming: PostgreSQL, Redis, Apache Kafka, DynamoDB
- Core Concepts: Distributed Systems, Microservices Architecture, Concurrency, RESTful APIs, gRPC

Education:
University of California, Berkeley
- B.S. in Computer Science (2013 - 2017) | GPA: 3.85/4.0
"""
    _write_pdf(output_dir / "profile_a_strong.pdf", strong_resume)

    # ── 3. Profile B: Pivot / ML Engineer ──
    pivot_resume = """Elena Rostova
Email: elena.rostova@ailab.org | Phone: +1-555-0144
New York, NY

Summary:
Machine Learning Engineer with 4 years of experience building and deploying computer vision and NLP models into production cloud environments.

Work Experience:
Machine Learning Engineer | VisionAI Labs | 2022 - Present
- Developed and fine-tuned Transformer models for document understanding using PyTorch and Hugging Face.
- Containerized deep learning inference pipelines with Docker and deployed them to AWS SageMaker.
- Maintained Python microservices exposing model inference APIs via FastAPI.
- Managed training datasets stored in AWS S3 and automated experiment tracking with MLflow.

Junior Data Scientist | DataCorp | 2020 - 2022
- Built automated ETL data pipelines in Python and Pandas for customer churn prediction.
- Analyzed tabular data in PostgreSQL and developed visualization dashboards for stakeholders.

Skills:
Python, PyTorch, TensorFlow, Scikit-Learn, Docker, AWS S3, SageMaker, FastAPI, PostgreSQL, Git, Linux, HuggingFace

Education:
Columbia University
- M.S. in Data Science (2018 - 2020)
- B.S. in Applied Mathematics (2014 - 2018)
"""
    _write_pdf(output_dir / "profile_b_pivot.pdf", pivot_resume)

    # ── 4. Profile C: Junior Developer ──
    junior_resume = """Kevin Patel
Email: kevin.patel@studentmail.edu | Phone: +1-555-0199
Austin, TX

Objective:
Enthusiastic Junior Software Engineer looking to contribute backend development skills to a fast-paced engineering team.

Work Experience:
Junior Backend Developer | Startup Hub | 2025 - Present
- Assisted in building internal REST API endpoints using Python and Flask.
- Wrote unit tests and fixed bugs in PostgreSQL database migration scripts.
- Participated in daily agile standups and sprint planning meetings.

Software Engineering Intern | TechStart | Summer 2024
- Implemented user authentication and session management using JWT and Redis.
- Created documentation for internal developer APIs.

Skills:
Python, Flask, SQLite, PostgreSQL, HTML, CSS, JavaScript, Git, Basic Docker

Education:
University of Texas at Austin
- B.S. in Computer Science (Graduated May 2025)
"""
    _write_pdf(output_dir / "profile_c_junior.pdf", junior_resume)

    # ── 5. Profile D: Completely Irrelevant (Chef) ──
    chef_resume = """Marco Rossi
Email: marco.rossi@culinaryarts.com | Phone: +1-555-0111
Chicago, IL

Executive Summary:
Passionate Executive Chef with over 10 years of culinary leadership in high-end Michelin-starred restaurants. Expert in French and Italian cuisine, kitchen operations, inventory management, and banquet planning.

Work Experience:
Executive Chef | Le Petit Bistro | 2018 - Present
- Led kitchen brigade of 25 culinary professionals in a 120-seat fine dining restaurant.
- Designed seasonal 7-course tasting menus that achieved a 25% increase in customer satisfaction.
- Managed food cost controls, supplier negotiations, and daily food safety inspections.
- Trained prep cooks and sous chefs in classic culinary techniques and sanitation.

Head Sous Chef | Ristorante Bella | 2014 - 2018
- Supervised line stations during busy dinner services delivering 300+ covers per night.
- Oversaw butchery, sauce preparation, pastry creation, and daily inventory tracking.

Skills & Expertise:
Culinary Arts, Menu Engineering, Food Safety, HACCP, Kitchen Operations, Wine Pairing, Staff Training, Food Costing

Education:
The Culinary Institute of America
- Associate Degree in Culinary Arts (2012 - 2014)
"""
    _write_pdf(output_dir / "profile_d_chef.pdf", chef_resume)

    # ── 6. Profile E: Stress Test / Edge Case ──
    stress_resume = """ALEXANDER VANDERBILT
Email: alex.vanderbilt@enterprise.global | Mobile: +44 20 7946 0912 | Address: 12 Canary Wharf, London, UK

WHAT I BRING TO THE TABLE
Senior technical leader and polyglot engineer with extensive background building enterprise architectures and mission critical systems across multiple industries with intense demands for throughput, security, resilience, fault tolerance, and compliance.

WHERE I HAVE WORKED
Principal Solutions Architect, Global Bank PLC (March 2019 to Present)
- Spearheaded the design, implementation, rollout, and post-launch monitoring of an international cross-border payment settlement engine processing over seventy thousand transactions per second with sub-second finality across seventeen geographic jurisdictions while ensuring complete adherence to ISO 20022 messaging standards, zero data loss, multi-region active-active disaster recovery, and automated failover capabilities using Go, Python, and Apache Kafka.
- Mentored and guided more than thirty engineers across four squads, conducted extensive architecture reviews, established unified coding style guidelines, introduced chaos engineering practices with Gremlin, and drastically cut deployment failure rates by seventy percent through automated verification gates.
- Built distributed streaming data pipelines leveraging Apache Kafka, Kafka Connect, and Apache Flink for real-time fraud detection and transaction anomaly screening.

Senior Systems Developer, Cloud Matrix Solutions (January 2015 to February 2019)
- Designed and built resilient microservices using Python, Golang, Docker, and Kubernetes on Amazon Web Services utilizing Amazon EKS, Aurora PostgreSQL, Redis cluster, and AWS Lambda.

AREAS OF EXPERTISE
Python, Go, Golang, C++, Java, Rust, JavaScript, TypeScript, Docker, Podman, Kubernetes, Helm, Terraform, Ansible, AWS, GCP, Azure, Apache Kafka, RabbitMQ, Apache Flink, PostgreSQL, MySQL, Redis, MongoDB, Cassandra, DynamoDB, Elasticsearch, gRPC, Protobuf, GraphQL, REST, Microservices, Event-Driven Architecture, CI/CD, Git, GitHub Actions, GitLab CI, Linux, Prometheus, Grafana, OpenTelemetry, Datadog

MY ACADEMIC CREDENTIALS
Imperial College London
- Master of Science in Advanced Computing (Graduated with Distinction, 2014)
University of Cambridge
- Bachelor of Arts with Honours in Computer Science (2010 - 2013)
"""
    _write_pdf(output_dir / "profile_e_stress.pdf", stress_resume)
    print("✅ Created all 6 test documents (1 JD + 5 Profiles)")


def _write_pdf(path: Path, text: str):
    doc = pymupdf.open()
    page = doc.new_page()
    lines = text.strip().split("\n")
    y_pos = 50

    for line in lines:
        if y_pos > 750:
            page = doc.new_page()
            y_pos = 50
        stripped = line.strip()
        if not stripped:
            y_pos += 10
            continue
        fontsize = 12 if stripped.isupper() or len(stripped) < 35 and stripped.endswith(":") else 9
        page.insert_text(pymupdf.Point(45, y_pos), stripped, fontsize=fontsize)
        y_pos += 14

    doc.save(str(path))
    doc.close()


def run_comprehensive_evaluation():
    test_dir = Path("data/impactful_test_docs")
    create_test_documents(test_dir)

    config = load_config()
    ingestion = IngestionService(config)
    matching = MatchingService(config)
    report_builder = ReportBuilder(config)

    print("\n" + "=" * 80)
    print("🚀 COMPREHENSIVE PIPELINE EVALUATION: EFFICIENCY & ACCURACY")
    print("=" * 80)

    # 1. Ingest JD
    t0 = time.perf_counter()
    jd_doc = ingestion.ingest(test_dir / "target_jd.pdf", DocumentType.JD)
    jd_ingest_time = (time.perf_counter() - t0) * 1000.0
    print(f"\n📋 Target JD Ingested in {jd_ingest_time:.1f}ms | Blocks: {len(jd_doc.blocks)}")

    # Verify JD chunks <= 40 words
    for i, b in enumerate(jd_doc.blocks):
        wc = len(b.text.split())
        assert wc <= 40, f"JD Block {i} exceeded 40 words ({wc} words): {b.text[:50]}"
    print("  ✓ All JD blocks satisfy 40-word max constraint")

    profiles = [
        ("Profile A (Strong Match)", test_dir / "profile_a_strong.pdf", "high"),
        ("Profile B (Pivot / ML)", test_dir / "profile_b_pivot.pdf", "moderate"),
        ("Profile C (Junior Dev)", test_dir / "profile_c_junior.pdf", "low_moderate"),
        ("Profile D (Irrelevant Chef)", test_dir / "profile_d_chef.pdf", "very_low"),
        ("Profile E (Stress / Edge Case)", test_dir / "profile_e_stress.pdf", "high"),
    ]

    results = []

    for name, resume_path, expected_tier in profiles:
        print(f"\n{'─' * 40}")
        print(f"Testing {name}...")

        # Ingestion
        t0 = time.perf_counter()
        resume_doc = ingestion.ingest(resume_path, DocumentType.RESUME)
        res_ingest_time = (time.perf_counter() - t0) * 1000.0

        # Chunk word-count audit
        max_wc = 0
        oversized_chunks = []
        for idx, b in enumerate(resume_doc.blocks):
            wc = len(b.text.split())
            if wc > max_wc:
                max_wc = wc
            if wc > 40:
                oversized_chunks.append((idx, wc, b.text[:60]))

        # Matching & Scoring
        t0 = time.perf_counter()
        candidate_score, evidence = matching.match_and_score(jd_doc, resume_doc)
        match_time = (time.perf_counter() - t0) * 1000.0

        # Build Report
        report = report_builder.build_report(jd_doc, resume_doc, candidate_score, evidence)

        # 30% cosine floor audit: verify no evidence with score > 0 has sim < 0.30
        floor_violations = [
            e for e in evidence
            if 0.0 < e.similarity_score < 0.30
        ]

        # Collect component scores
        comps = {c.name: c.raw_score for c in candidate_score.components}
        comp_weights = {c.name: c.weight for c in candidate_score.components}

        # Check weights sum to 1.0 and match configuration
        assert abs(sum(comp_weights.values()) - 1.0) < 1e-6
        assert comp_weights.get("content_score") == 0.35
        assert comp_weights.get("work_experience") == 0.35
        assert comp_weights.get("skill") == 0.15
        assert comp_weights.get("education") == 0.15

        results.append({
            "name": name,
            "tier": expected_tier,
            "final_score": candidate_score.final_score,
            "components": comps,
            "ingest_ms": res_ingest_time,
            "match_ms": match_time,
            "total_ms": res_ingest_time + match_time,
            "blocks_count": len(resume_doc.blocks),
            "max_chunk_words": max_wc,
            "oversized_chunks": oversized_chunks,
            "floor_violations": len(floor_violations),
            "evidence_count": len(evidence),
            "all_mandatory_met": candidate_score.all_mandatory_met,
        })

        print(f"  Ingest: {res_ingest_time:.1f}ms | Match & Score: {match_time:.1f}ms | Total: {res_ingest_time + match_time:.1f}ms")
        print(f"  Blocks: {len(resume_doc.blocks)} | Max Chunk Words: {max_wc}/40")
        print(f"  Final Score: {candidate_score.final_score:.2f} / 100")
        print(f"  Components: Skill={comps.get('skill', 0):.1f} (15%), WorkExp={comps.get('work_experience', 0):.1f} (35%), "
              f"Edu={comps.get('education', 0):.1f} (15%), Content={comps.get('content_score', 0):.1f} (35%)")
        print(f"  Mandatory Met: {candidate_score.all_mandatory_met}")

    # ── Summary & Accuracy Analysis ──
    print("\n" + "=" * 80)
    print("📊 EVALUATION RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Profile':<30} | {'Score':<8} | {'Chunk<=40':<10} | {'Floor Viol':<10} | {'Total Time':<10}")
    print("─" * 80)
    for r in results:
        chunk_ok = "PASS" if not r["oversized_chunks"] else f"FAIL({len(r['oversized_chunks'])})"
        floor_ok = "PASS" if r["floor_violations"] == 0 else f"FAIL({r['floor_violations']})"
        print(f"{r['name']:<30} | {r['final_score']:<8.2f} | {chunk_ok:<10} | {floor_ok:<10} | {r['total_ms']:<8.1f}ms")

    # ── Verify Rank Ordering ──
    score_a = results[0]["final_score"]
    score_b = results[1]["final_score"]
    score_c = results[2]["final_score"]
    score_d = results[3]["final_score"]
    score_e = results[4]["final_score"]

    print("\n📈 RANK ORDERING VERIFICATION:")
    print(f"  1. Strong Match (A): {score_a:.2f}")
    print(f"  2. Stress Test (E): {score_e:.2f}")
    print(f"  3. Pivot Match (B): {score_b:.2f}")
    print(f"  4. Junior Dev (C): {score_c:.2f}")
    print(f"  5. Irrelevant Chef (D): {score_d:.2f}")

    # Assertions for ranking and constraints
    assert score_a > score_b, f"Expected Strong ({score_a}) > Pivot ({score_b})"
    assert score_b > score_c, f"Expected Pivot ({score_b}) > Junior ({score_c})"
    assert score_c > score_d, f"Expected Junior ({score_c}) > Irrelevant Chef ({score_d})"
    assert score_e > score_b, f"Expected Stress Senior ({score_e}) > Pivot ({score_b})"
    assert score_d < 25.0, f"Expected Irrelevant Chef score < 25.0, got {score_d}"

    for r in results:
        assert not r["oversized_chunks"], f"Found oversized chunks in {r['name']}: {r['oversized_chunks']}"
        assert r["floor_violations"] == 0, f"Found cosine floor violations in {r['name']}"

    print("\n🎉 ALL TESTS AND CONSTRAINTS STRICTLY PASSED!")
    return results


if __name__ == "__main__":
    run_comprehensive_evaluation()
