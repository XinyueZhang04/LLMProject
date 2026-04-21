# ask_table.py
import os
import pickle
import faiss
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer

# ========================
# 解决路径问题（核心）
# ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

table_path = os.path.join(BASE_DIR, "../output/table_rows.pkl")
index_path = os.path.join(BASE_DIR, "../output/table_index.faiss")

print("[DEBUG] table_path:", table_path)
print("[DEBUG] index_path:", index_path)

# ========================
# 加载数据
# ========================
with open(table_path, "rb") as f:
    table_rows = pickle.load(f)

print(f"[INFO] Loaded {len(table_rows)} table rows.")

index = faiss.read_index(index_path)
print("[INFO] FAISS index loaded.")

# ========================
# embedding 模型（不变，保证和 index 一致）
# ========================
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ========================
# 检索函数
# ========================
def retrieve_context(query, top_k=5):
    query_vec = model.encode([query])
    query_vec = np.array(query_vec, dtype="float32")

    distances, indices = index.search(query_vec, top_k)

    context_rows = [table_rows[i] for i in indices[0]]
    return context_rows

# ========================
# 问答函数（Ollama - FIX ONLY HERE）
# ========================
def ask_table(query):
    context_rows = retrieve_context(query)

    context_text = "\n".join([
        r.get("text", str(r)) for r in context_rows
    ])

    prompt = f"""
You are a precise assistant.

Rules:
- Use ONLY the provided table data
- If not found, say "I don't know"
- Avoid guessing

Table Data:
{context_text}

Question: {query}

Answer:
"""

    response = ollama.chat(
        model='qwen2.5:7b',   # ⭐ FIX: 原来 qwen1.5b → 升级
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response['message']['content']

# ========================
# 主循环
# ========================
if __name__ == "__main__":
    print("[INFO] Table QA system started. Type 'exit' to quit.")

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break

        answer = ask_table(query)
        print("AI:", answer)