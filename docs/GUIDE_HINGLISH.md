# Resume Matcher — Simple Guide (English/Hinglish)

Yeh guide aapko batayegi ki Resume Matcher kaise kaam karta hai, simple language mein.

## System Kya Karta Hai?

Yeh system ek **Job Description (JD)** aur ek **Resume** ko compare karta hai aur batata hai ki resume kitna match karta hai JD ke saath.

- Koi LLM ya ChatGPT nahi use hota
- Sab kuch **deterministic** hai — same input pe same result aayega
- Har score ka **reason** diya jaata hai

## Kaise Kaam Karta Hai? (Pipeline)

```
PDF/DOCX File
  ↓
1. FILE CHECK     → Kya file sahi hai? Size, type check karo
  ↓
2. TEXT NIKALO    → PDF/DOCX se text extract karo, line by line
  ↓
3. SAAF KARO      → Extra spaces, junk characters hatao
  ↓
4. SAMJHO         → "3.5 years", "C++" jaise terms ko protect karo
  ↓
5. SENTENCES      → Text ko sentences mein todo (rules se, AI se nahi)
  ↓
6. BLOCKS         → Sentences ko meaningful groups mein daalo
  ↓
7. SECTIONS       → "SKILLS", "EXPERIENCE" jaise sections detect karo
  ↓
8. VECTORS        → BGE model se har block ka 768-number wala vector banao
  ↓
9. MATCH          → FAISS se JD blocks aur Resume blocks compare karo
  ↓
10. SCORE         → 4 scores calculate karo, final score nikalo
  ↓
11. REPORT        → Recruiter ke liye full report banao (JSON mein)
```

## Scoring Kaise Hota Hai?

```
Final Score = Skill × 30% + Experience × 15% + Education × 15% + Semantic × 40%
```

| Component | Weight | Kya Dekha Jaata Hai |
|-----------|--------|-------------------|
| Skill Score | 30% | Resume mein JD ke skills kitne match hote hain |
| Work Experience | 15% | Kaam ka experience kitna match karta hai |
| Education | 15% | Padhai/degree kitni match karti hai |
| Semantic Match | 40% | Overall, poora resume JD se kitna milta hai |

## Important Folders

| Folder | Kya Karta Hai |
|--------|-------------|
| `domain/` | Data structures define karta hai (jaise Block, Sentence) |
| `ingestion/` | PDF/DOCX files padhta hai |
| `preprocessing/` | Text saaf karta hai |
| `segmentation/` | Text ko sentences aur blocks mein todta hai |
| `metadata/` | "SKILLS", "EDUCATION" sections dhundhta hai |
| `embeddings/` | BGE model se vectors banata hai |
| `retrieval/` | FAISS se similar blocks dhundhta hai |
| `matching/` | JD aur Resume ke blocks match karta hai |
| `scoring/` | Score calculate karta hai |
| `reporting/` | Report banata hai |
| `services/` | Sabko milake pipeline chalata hai |
| `api/` | FastAPI web server |

## Kaise Chalayein?

### Terminal se:
```bash
python -m scripts.run_pipeline --jd jd.pdf --resume resume.pdf
```

### API se:
```bash
# Server start karo
uvicorn resume_matcher.api.app:app --reload --port 8000

# Browser mein jao: http://localhost:8000/docs
# Wahan se files upload karke match karo
```

### Test karo:
```bash
pytest tests/ -v
```

## Important Concepts

### Block Kya Hai?
Block = ek meaningful piece of information. Jaise "5+ years Python experience" ya "B.Tech from IIT Delhi". Yeh sentences ka group hota hai jo ek hi topic ke baare mein baat karta hai.

### Section Inference Kya Hai?
Agar document mein heading nahi likha hai (jaise "SKILLS" ya "EDUCATION"), toh system khud samajhne ki koshish karta hai ki yeh text kis section ka hai. Woh dekhta hai:
- Aas paas ke blocks kis section mein hain
- Text mein kya likha hai (dates = experience, degree names = education)
- Document mein kahan hai (start mein = summary, end mein = education)

### Mandatory Requirements Kya Hain?
JD mein "must have", "required", "essential" likha ho toh woh mandatory hai. System check karta hai ki resume mein yeh sab milte hain ya nahi.

### FAISS Kya Hai?
Facebook ka banaya hua tool hai jo bahut tezi se vectors compare karta hai. Hum cosine similarity use karte hain — yani do texts kitne similar hain, 0 se 1 ke beech.

### BGE Model Kya Hai?
BAAI/bge-base-en-v1.5 — yeh ek model hai jo kisi bhi English text ko 768 numbers ke vector mein convert karta hai. Similar meaning wale texts ke vectors bhi similar hote hain.
