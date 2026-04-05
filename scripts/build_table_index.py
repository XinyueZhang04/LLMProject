# build_table_index.py
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 读取表格行
table_path = "../output/table_rows.pkl"  # 相对路径，从 scripts 访问 output
with open(table_path, "rb") as f:
    table_rows = pickle.load(f)

print(f"Loaded {len(table_rows)} table rows.")

# 生成向量嵌入
# 使用小型模型，速度快
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(table_rows, show_progress_bar=True)

# 建 FAISS 索引
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(np.array(embeddings, dtype=np.float32))

print(f"FAISS index built with {index.ntotal} vectors.")

# 保存索引和表格行，方便查询
faiss.write_index(index, "../output/table_index.faiss")
with open("../output/table_rows_for_index.pkl", "wb") as f:
    pickle.dump(table_rows, f)

print("FAISS index and table rows saved in output/ folder.")