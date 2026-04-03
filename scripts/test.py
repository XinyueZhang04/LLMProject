import os
import pickle

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 拼接 output 路径
table_path = os.path.join(script_dir, "../output/table_rows.pkl")

with open(table_path, "rb") as f:
    table_rows = pickle.load(f)

print(f"Loaded {len(table_rows)} table rows")