# Recruiter Report Schema & Specification

Every execution of the Resume Matcher produces an auditable, machine-readable JSON report adhering to schema version `1.0.0`.

---

## Top-Level Report Structure

```json
{
  "_schema_version": "1.0.0",
  "_report_id": "30d466db9e93",
  "processing_details": {
    "processing_version": "0.1.0",
    "schema_version": "1.0.0",
    "processed_at": "2026-09-19T17:50:35.815443+00:00",
    "config_hash": "8e078202",
    "jd_filename": "sample_jd.pdf",
    "resume_filename": "sample_resume.pdf",
    "jd_block_count": 7,
    "resume_block_count": 5
  },
  "candidate_score": { ... },
  "skill_evidence": { ... },
  "experience_evidence": { ... },
  "education_evidence": { ... },
  "semantic_evidence": { ... },
  "strongest_matches": [ ... ],
  "weakest_matches": [ ... ],
  "mandatory_requirements": [ ... ],
  "missing_requirements": [ ... ],
  "review_flags": [ ... ],
  "data_quality_warnings": [ ... ],
  "recruiter_summary": "Overall Assessment: Strong Candidate (Score: 82.9/100)..."
}
```

---

## Field Specifications

### 1. `candidate_score`
Contains the overall score and the weighted component breakdown:

```json
{
  "candidate_id": "77ef5d3c1c64",
  "jd_id": "74a6aa76944e",
  "final_score": 82.96,
  "components": [
    {
      "name": "skill",
      "raw_score": 81.96,
      "weight": 0.3,
      "weighted_contribution": 24.588,
      "evidence_count": 5,
      "explanation": "Quality: 0.699, Coverage: 100.0%, Matched 1/1 skill requirements",
      "missing_evidence": []
    },
    {
      "name": "work_experience",
      "raw_score": 83.96,
      "weight": 0.15,
      "weighted_contribution": 12.594,
      "evidence_count": 10,
      "explanation": "Quality: 0.733, Coverage: 100.0%, Matched 2/2 experience requirements",
      "missing_evidence": []
    },
    {
      "name": "education",
      "raw_score": 77.93,
      "weight": 0.15,
      "weighted_contribution": 11.689,
      "evidence_count": 10,
      "explanation": "Quality: 0.632, Coverage: 100.0%, Matched 2/2 education requirements",
      "missing_evidence": []
    },
    {
      "name": "semantic_match",
      "raw_score": 85.23,
      "weight": 0.4,
      "weighted_contribution": 34.092,
      "evidence_count": 25,
      "explanation": "Overall quality: 0.710, Overall coverage: 100.0%, Matched 7/7 total JD blocks",
      "missing_evidence": []
    }
  ],
  "all_mandatory_met": true
}
```

### 2. Evidence Item Schema (`strongest_matches`, `weakest_matches`, component evidence)

```json
{
  "evidence_id": "93d6a2df97",
  "jd_text": "Experience with cloud platforms (AWS/GCP/Azure) is essential.",
  "resume_text": "Experienced software engineer with 5+ years in Python, AWS, and cloud technologies.",
  "similarity_score": 0.7647,
  "jd_section": "requirements",
  "resume_section": "summary",
  "jd_line_numbers": [5, 6, 7],
  "resume_line_numbers": [4, 5, 6],
  "match_strength": "strong",
  "explanation": "Match strength: strong (similarity: 0.765) | JD section: requirements | Resume section: summary",
  "uncertainty": false
}
```

### 3. `mandatory_requirements`

```json
{
  "requirement_text": "Must have 5+ years of experience in software development.",
  "is_met": true,
  "best_match_score": 0.6858,
  "best_match_text": "Experienced software engineer with 5+ years of experience in Python...",
  "explanation": "Met: best match score 0.686 (threshold: 0.5)"
}
```
