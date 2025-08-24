import os
import json
import numpy as np
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai

# === CONFIG ===
EMBED_MODEL = "models/text-embedding-004"
CHAT_MODEL = "gemini-1.5-flash"
app = Flask(__name__, static_folder='static', template_folder='templates')

# Load resume embeddings
with open("resume_embeddings.json", "r", encoding="utf-8") as f:
    CHUNKS = json.load(f)

# Configure Gemini API
API_KEY = "AIzaSyDLlxsxmiK0KACUEq_neAMtE102uecGHQM"
if not API_KEY:
    raise ValueError("Please set API_KEY environment variable.")
genai.configure(api_key=API_KEY)

# === Helper functions ===
def cosine_similarity(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def retrieve(query, top_k=3):
    q_emb = genai.embed_content(model=EMBED_MODEL, content=query)["embedding"]
    scored = [(cosine_similarity(q_emb, c["embedding"]), c) for c in CHUNKS if c["embedding"]]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]

def generate_answer(query, context_chunks):
    if not context_chunks:
        return "I don’t know the answer to that."
    context_text = "\n\n".join(f"{c['title']}: {c['text']}" for c in context_chunks)
    prompt = f"""
You are Shashank Jaiswal's personal assistant.
Answer in first-person as Shashank.
ONLY based on the following context. If unsure, say you don’t know.
Context:
{context_text}
Question: {query}
"""
    try:
        response = genai.GenerativeModel(CHAT_MODEL).generate_content(prompt)
        return response.text if response.text else "I don’t know."
    except Exception as e:
        return f"Error: {str(e)}"

# === Flask app ===
app = Flask(__name__, template_folder="templates")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Missing query"}), 400

    context = retrieve(query)
    answer = generate_answer(query, context)
    return jsonify({"answer": answer})

# === Run server ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))  # Railway will provide PORT
    app.run(host="0.0.0.0", port=port)
