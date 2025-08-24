# resume_rag_builder.py
# ----------------------
# Install first: pip install google-generativeai PyPDF2 numpy

import os, re, json
import numpy as np
import google.generativeai as genai
import PyPDF2

# === CONFIG ===
PDF_PATH = "resume_chunks.pdf"        # your input PDF
OUTPUT_JSON = "resume_embeddings.json"
EMBED_MODEL = "models/text-embedding-004"
API_KEY = "AIzaSyDLlxsxmiK0KACUEq_neAMtE102uecGHQM"

# =============

def extract_text_from_pdf(path):
    """Extract text from PDF into a single string."""
    text = ""
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            extracted = page.extract_text() or ""
            text += extracted + "\n"
    return text

def split_into_chunks(text):
    """Split text by ### markers into sections."""
    sections = []
    current_title, current_lines = None, []

    for line in text.splitlines():
        if re.match(r'^\s*###\s+', line):
            if current_title is not None:
                sections.append({
                    "title": current_title.strip(),
                    "text": "\n".join(current_lines).strip()
                })
            current_title = re.sub(r'^\s*###\s+', '', line).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_title is not None:
        sections.append({
            "title": current_title.strip(),
            "text": "\n".join(current_lines).strip()
        })
    return sections

def build_embeddings(sections):
    """Attach Gemini embeddings to each chunk."""
    genai.configure(api_key=API_KEY)
    out = []
    for i, sec in enumerate(sections):
        if not sec["text"].strip():
            emb = None
        else:
            resp = genai.embed_content(model=EMBED_MODEL, content=sec["text"])
            emb = resp["embedding"]
        out.append({
            "id": i,
            "title": sec["title"],
            "text": sec["text"],
            "embedding": emb
        })
    return out

def main():
    text = extract_text_from_pdf(PDF_PATH)
    sections = split_into_chunks(text)
    data = build_embeddings(sections)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ Created {OUTPUT_JSON} with {len(data)} chunks")

if __name__ == "__main__":
    main()
