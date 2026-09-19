# Scoring Logic

## Formula

```
Final Score = Skill × 0.30 + Work Experience × 0.15 + Education × 0.15 + Semantic Match × 0.40
```

All scores are on a **0–100 scale**. Weights **must sum to 1.0** (validated at startup).

## Components

### 1. Skill Score (30%)

- **Input**: JD blocks in SKILLS, PROJECTS, CERTIFICATIONS, REQUIREMENTS sections
- **Metric**: `quality × 0.6 + coverage × 0.4`
  - Quality = average similarity of best matches per JD block
  - Coverage = ratio of JD skill blocks that have any match
- **Supporting sections**: PROJECTS and CERTIFICATIONS boost this score

### 2. Work Experience Score (15%)

- **Input**: JD blocks in EXPERIENCE, PROJECTS, ACHIEVEMENTS, RESPONSIBILITIES sections
- **Metric**: Same quality × coverage formula
- **Note**: Projects and achievements contribute here too

### 3. Education Score (15%)

- **Input**: JD blocks in EDUCATION, CERTIFICATIONS sections
- **Metric**: Same quality × coverage formula

### 4. Overall Semantic Match Score (40%)

- **Input**: ALL JD blocks across ALL sections
- **Metric**: `quality × 0.5 + coverage × 0.5`
- **Purpose**: Captures holistic fit beyond specific section matching

## Important Rules

1. Projects, certifications, achievements **support existing components** but are NOT separate score components
2. Missing evidence produces **0 score with explanation**, not NaN
3. All component weights are **validated at startup** (must sum to 1.0)
4. Component contributions are **preserved for auditability**
5. No hidden scoring rules — everything is in the config and code

## Match Strength Classification

| Strength | Similarity Threshold | Meaning |
|----------|---------------------|---------|
| Strong   | ≥ 0.75              | High-confidence match |
| Moderate | ≥ 0.55              | Reasonable match |
| Weak     | ≥ 0.35              | Low-confidence match |
| None     | < 0.35              | No meaningful match |

Thresholds are configurable in `config/default.yaml`.
