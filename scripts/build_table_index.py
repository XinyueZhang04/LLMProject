import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ============================================================
# 读取 extractor 输出（关键改动）
# ============================================================
rows_path = "../output/rows.pkl"

with open(rows_path, "rb") as f:
    table_rows = pickle.load(f)

print(f"Loaded {len(table_rows)} rows.")

# ============================================================
# FIX 1：只用 extractor 的 text（不是 raw row）
# ============================================================
texts = []

for r in table_rows:
    text = r.get("text", "")

    # ===== FIX：过滤空/垃圾行 =====
    if text and len(text.strip()) > 0:
        texts.append(text)

print(f"Valid texts for embedding: {len(texts)}")

# ============================================================
# embedding model（可替换）
# ============================================================
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ===== FIX 2：normalize_embeddings 提升检索稳定性 =====
embeddings = model.encode(
    texts,
    show_progress_bar=True,
    normalize_embeddings=True
)

embeddings = np.array(embeddings, dtype=np.float32)

# ============================================================
# FAISS index
# ============================================================
embedding_dim = embeddings.shape[1]

index = faiss.IndexFlatIP(embedding_dim)  # FIX 3：改 cosine similarity（比 L2 更适合文本）
index.add(embeddings)

print(f"FAISS index built with {index.ntotal} vectors.")

# ============================================================
# 保存 index + 对齐文本
# ============================================================
faiss.write_index(index, "../output/table_index.faiss")

with open("../output/table_texts.pkl", "wb") as f:
    pickle.dump(texts, f)

# ===== FIX 4：保存原 row 映射（用于 debug / rerank）=====
with open("../output/table_rows_for_index.pkl", "wb") as f:
    pickle.dump(table_rows, f)

print("FAISS index + texts saved.")