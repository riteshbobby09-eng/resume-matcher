# Troubleshooting & Diagnostics Guide

This guide covers common issues, root causes, and verified resolutions for Resume Matcher.

---

## 1. File Ingestion & Extraction Issues

### `FileValidationError: File not found`
- **Cause:** Provided file path is nonexistent or misspelled.
- **Fix:** Verify relative or absolute path. Ensure files are stored in `data/` or provide full path.

### `FileValidationError: Unsupported file type`
- **Cause:** Uploaded file extension is not `.pdf` or `.docx`.
- **Fix:** Convert file to PDF or DOCX. Plain text files (.txt) or images (.png/.jpg) are not supported.

### `CorruptedDocumentError`
- **Cause:** File is truncated, has invalid magic bytes, or is password-protected.
- **Fix:** Ensure document is not encrypted and can be opened in Adobe Acrobat or Microsoft Word.

---

## 2. Embedding Model Issues

### Slow first run (Cold start ~12-14 seconds)
- **Cause:** The BGE model weights (`~438 MB`) are loaded lazily into memory on the first call.
- **Fix:** This is normal behavior. Subsequent calls execute in `~300 ms`. In production services, the model is pre-warmed on server startup.

### HuggingFace Rate Limit Warnings
- **Cause:** Unauthenticated requests to Hugging Face Hub.
- **Fix:** Set the `HF_TOKEN` environment variable in your `.env` file:
  ```bash
  export HF_TOKEN="your_hf_token_here"
  ```
  Or pre-download the model using:
  ```bash
  python -m scripts.download_model
  ```

---

## 3. FAISS & Vector Issues

### `ImportError: DLL load failed` or FAISS issues on Windows/Linux
- **Cause:** Incompatible FAISS wheel.
- **Fix:** Ensure `faiss-cpu>=1.8.0` is installed. On Apple Silicon, use `pip install faiss-cpu`.

---

## 4. Port Conflicts (FastAPI Server)

### `OSError: [Errno 48] Address already in use: 8000`
- **Cause:** Another process is occupying port 8000.
- **Fix:** Kill the existing process:
  ```bash
  lsof -ti :8000 | xargs kill -9
  ```
  Or run the server on a custom port:
  ```bash
  uvicorn resume_matcher.api.app:app --port 8080 --reload
  ```
