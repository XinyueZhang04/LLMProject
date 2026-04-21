import camelot
import pdfplumber
import os
import pickle
import json
import warnings
import pandas as pd
import re
import time
from collections import Counter

warnings.filterwarnings("ignore")
time.sleep(1)

# ============================================================
# 配置路径
# ============================================================
pdf_dir = "../data/aluminium"  # PDF 所在目录
output_dir = "../output"  # 输出目录
rows_path = os.path.join(output_dir, "rows.pkl")  # 行级记录
tables_path = os.path.join(output_dir, "tables.pkl")  # 表级记录
documents_path = os.path.join(output_dir, "documents.jsonl")  # RAG 入库文件

os.makedirs(output_dir, exist_ok=True)

# 最终收集容器
all_rows = []  # 行级
all_tables = []  # 表级


# ============================================================
# 工具函数：文本统计
# ============================================================
def count_letters(text: str) -> int:
    return len(re.findall(r'[A-Za-z\u4e00-\u9fff]', str(text)))


def count_digits(text: str) -> int:
    return len(re.findall(r'[0-9]', str(text)))


# ============================================================
# Step 1：基础清洗
# ============================================================
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.replace(r'[\r\n\t]+', ' ', regex=True)
    df = df.replace(r'^\s*$', None, regex=True)

    # ===== FIX 1：避免空字符串污染 embedding =====
    def smart_clean(x):
        if isinstance(x, str):
            x = x.strip()
            return x if x != "" else None
        return x

    df = df.map(smart_clean)

    df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')
    df = df.reset_index(drop=True)
    return df


# ============================================================
# Step 2：表格类型识别
# ============================================================
def is_key_value_table(df: pd.DataFrame) -> bool:
    if df.shape[1] != 2 or df.shape[0] < 2:
        return False
    left = df.iloc[:, 0].dropna().astype(str)
    right = df.iloc[:, 1].dropna().astype(str)
    if len(left) == 0 or len(right) == 0:
        return False
    left_avg_letters = left.apply(count_letters).mean()
    right_avg_letters = right.apply(count_letters).mean()
    left_unique_ratio = left.nunique() / max(len(left), 1)
    return left_avg_letters >= right_avg_letters * 0.8 and left_unique_ratio > 0.6


def detect_header_rows(df: pd.DataFrame, max_header_rows: int = 3) -> int:
    header_rows = 0
    for i in range(min(max_header_rows, len(df))):
        row_str = " ".join([str(x) for x in df.iloc[i].tolist() if pd.notna(x)])
        if count_letters(row_str) >= count_digits(row_str):
            header_rows += 1
        else:
            break
    return max(header_rows, 1)


def has_merged_cells_pattern(df: pd.DataFrame) -> bool:
    if df.shape[0] < 3 or df.shape[1] < 2:
        return False
    col_null_ratios = df.isna().mean()
    suspicious = (col_null_ratios > 0.2) & (col_null_ratios < 0.9)
    return suspicious.any()


def classify_table(df: pd.DataFrame) -> str:
    if is_key_value_table(df):
        return "key_value"
    if detect_header_rows(df) >= 2:
        return "multi_level_header"
    if has_merged_cells_pattern(df):
        return "merged_cells"
    return "simple_grid"


# ============================================================
# Step 3：合并单元格补全
# ============================================================
def fill_merged_cells(df: pd.DataFrame, header_rows: int = 1) -> pd.DataFrame:
    df = df.copy()
    if len(df) <= header_rows:
        return df

    body = df.iloc[header_rows:].copy()

    # ===== FIX 2：避免 NaN 传播造成 |||| =====
    body = body.fillna("")

    body = body.ffill(axis=0)
    body = body.ffill(axis=1)
    df.iloc[header_rows:] = body
    return df


# ============================================================
# Step 4：列名构造
# ============================================================
def deduplicate_columns(columns: list) -> list:
    counter = Counter()
    result = []
    for col in columns:
        counter[col] += 1
        result.append(col if counter[col] == 1 else f"{col}_{counter[col]}")
    return result


def build_headers(df: pd.DataFrame, header_rows: int = 1) -> list:
    header_df = df.iloc[:header_rows].copy()
    header_df = header_df.ffill(axis=0).ffill(axis=1)
    headers = []
    for col in range(df.shape[1]):
        parts = []
        seen = set()
        for r in range(header_rows):
            val = str(header_df.iloc[r, col]).strip()
            if val and val.lower() not in ("none", "nan") and val not in seen:
                seen.add(val)
                parts.append(val)
        headers.append("_".join(parts) if parts else f"Column_{col}")
    return deduplicate_columns(headers)


# ============================================================
# Step 5：行文本构造（embedding）
# ============================================================
def row_to_text(source: str, page, table_idx: int,
                row_idx: int, row_dict: dict, table_type: str) -> str:

    parts = [
        f"File: {source}",
        f"Page: {page}",
        f"Table_{table_idx}",
        f"Type: {table_type}",
    ]

    for k, v in row_dict.items():
        if pd.notna(v):
            v_str = str(v).strip()
            if v_str and v_str.lower() not in ("none", "nan"):
                parts.append(f"{k}: {v_str}")

    # ===== FIX 3：增强数字语义（解决 99.7 vs 99.8）=====
    num_boost = []
    for v in row_dict.values():
        if any(c.isdigit() for c in str(v)):
            num_boost.append(str(v))

    return " | ".join(map(str, parts + num_boost))


# ============================================================
# Step 6：表格级文本
# ============================================================
def table_to_summary_text(source: str, page, table_idx: int,
                          table_type: str, row_dicts: list) -> str:

    header = f"File: {source} | Page: {page} | Table_{table_idx} | Type: {table_type}"

    row_texts = []
    for rd in row_dicts:

        # ===== FIX 4：过滤低信息行 =====
        if len(rd) <= 1:
            continue

        parts = [f"{k}: {v}" for k, v in rd.items()
                 if pd.notna(v) and str(v).strip().lower() not in ("none", "nan", "")]

        if parts:
            row_texts.append(" | ".join(parts))

    return header + " | " + " ;; ".join(row_texts)


# ============================================================
# Step 7：grid table
# ============================================================
def process_grid_table(df, source, page, table_idx, table_type, mode, score):
    records = []
    header_rows = detect_header_rows(df)

    if table_type in ("merged_cells", "multi_level_header"):
        df = fill_merged_cells(df, header_rows)

    headers = build_headers(df, header_rows)
    body = df.iloc[header_rows:].copy()

    body = body.fillna("")
    body.columns = headers
    body = body.reset_index(drop=True)

    for r_idx, row in body.iterrows():

        row_dict = row.to_dict()

        # ===== FIX 2：去除空字段污染 embedding =====
        row_dict = {
            k: v for k, v in row_dict.items()
            if pd.notna(v) and str(v).strip() not in ("", "None", "nan")
        }

        # ===== FIX 2-2：过滤垃圾行 =====
        if len(row_dict) <= 1:
            continue

        valid_vals = [v for v in row_dict.values()
                      if pd.notna(v) and str(v).strip().lower() not in ("none", "nan", "")]

        if not valid_vals:
            continue

        records.append({
            "source": source,
            "page": page,
            "table": table_idx,
            "row": r_idx,
            "table_type": table_type,
            "extract_mode": mode,
            "parse_score": score,
            "structured_data": row_dict,
            "text": row_to_text(source, page, table_idx, r_idx, row_dict, table_type)
        })

    return records

# ============================================================
# Step 8：提取质量评分（用于 lattice vs stream 比较）
# ============================================================
def score_extraction(table) -> float:
    """
    利用 Camelot 内置 parsing_report 评估提取质量。
    accuracy 越高越好，whitespace 越低越好。
    """
    try:
        r = table.parsing_report
        return r.get("accuracy", 0) - r.get("whitespace", 100) * 0.2
    except Exception:
        return 0.0


# ============================================================
# Step 9：PDF 提取（Camelot 双模式 + pdfplumber fallback）
# ============================================================
def extract_with_camelot(pdf_path: str) -> list:
    """
    同时尝试 lattice（有线框表）和 stream（无线框表），
    返回 [(mode, camelot_table), ...] 列表。
    """
    results = []
    for flavor in ("lattice", "stream"):
        try:
            tables = camelot.read_pdf(pdf_path, pages='all', flavor=flavor)
            for tb in tables:
                results.append((flavor, tb))
        except Exception as e:
            print(f"    [WARN] Camelot {flavor} failed: {e}")
    return results


def extract_with_pdfplumber(pdf_path: str) -> list:
    """
    pdfplumber fallback：当 Camelot 提取质量差时使用。
    返回 [("pdfplumber", page_num, pd.DataFrame), ...] 列表。
    """
    results = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()
                for tb in tables:
                    if tb:
                        df = pd.DataFrame(tb)
                        results.append(("pdfplumber", page_num, df))
    except Exception as e:
        print(f"    [WARN] pdfplumber failed: {e}")
    return results


def deduplicate_tables(camelot_results, plumber_results):
    """
    简单去重策略：
    - Camelot 结果优先
    - pdfplumber 结果中，如果与 Camelot 同页已有高质量结果，则跳过

    返回统一格式：[(mode, page, df, score), ...]
    """
    unified = []
    camelot_pages_with_good_score = set()

    for mode, tb in camelot_results:
        page = getattr(tb, "page", None)
        sc = score_extraction(tb)
        df = tb.df
        unified.append((mode, page, df, sc))
        if sc > 50:  # 质量阈值：高于 50 认为 Camelot 已经够用
            camelot_pages_with_good_score.add(page)

    for mode, page, df in plumber_results:
        if page in camelot_pages_with_good_score:
            continue  # 该页 Camelot 已有好结果，跳过 pdfplumber
        unified.append((mode, page, df, 0.0))

    return unified


# ============================================================
# Step 10：主流程
# ============================================================
for pdf_file in os.listdir(pdf_dir):
    if not pdf_file.endswith(".pdf"):
        continue

    pdf_path = os.path.join(pdf_dir, pdf_file)
    print(f"[INFO] Processing: {pdf_file}")

    try:
        # --- 提取 ---
        camelot_results = extract_with_camelot(pdf_path)
        plumber_results = extract_with_pdfplumber(pdf_path)
        all_extracted = deduplicate_tables(camelot_results, plumber_results)

        for t_idx, (mode, page, df, sc) in enumerate(all_extracted):
            if df is None or df.empty:
                continue

            df = clean_dataframe(df)
            if df.empty or df.shape[0] < 1 or df.shape[1] < 1:
                continue

            # --- 分类 ---
            table_type = classify_table(df)
            print(f"    Table {t_idx}: page={page}, type={table_type}, mode={mode}, score={sc:.1f}")

            # --- 行级处理 ---
            if table_type == "key_value":
                rows = process_key_value_table(df, pdf_file, page, t_idx, mode, sc)
            else:
                rows = process_grid_table(df, pdf_file, page, t_idx, table_type, mode, sc)

            all_rows.extend(rows)

            # --- 表级处理（每张表生成一条 summary 记录）---
            if rows:
                row_dicts = [r["structured_data"] for r in rows]
                headers_used = list(row_dicts[0].keys()) if row_dicts else []
                table_record = {
                    "source": pdf_file,
                    "page": page,
                    "table": t_idx,
                    "table_type": table_type,
                    "extract_mode": mode,
                    "parse_score": sc,
                    "n_rows": len(rows),
                    "n_cols": df.shape[1],
                    "headers": headers_used,
                    "table_summary": table_to_summary_text(
                        pdf_file, page, t_idx, table_type, row_dicts
                    )
                }
                all_tables.append(table_record)

    except Exception as e:
        print(f"[ERROR] {pdf_file}: {e}")

# ============================================================
# Step 11：保存三种输出格式
# ============================================================

# 1. rows.pkl —— 行级记录，适合精确检索和 DataFrame 分析
with open(rows_path, "wb") as f:
    pickle.dump(all_rows, f)
print(f"[SAVED] rows.pkl → {len(all_rows)} row records")

# 2. tables.pkl —— 表级记录，适合表级语义理解
with open(tables_path, "wb") as f:
    pickle.dump(all_tables, f)
print(f"[SAVED] tables.pkl → {len(all_tables)} table records")

# 3. documents.jsonl —— 所有 chunk（行级 + 表级），适合直接 embedding 入库
# 格式：每行一个 JSON，包含 id、text、metadata
# 可直接用于 LangChain Document / LlamaIndex TextNode
with open(documents_path, "w", encoding="utf-8") as f:
    # 行级 chunk
    for rec in all_rows:
        doc = {
            "id": f"{rec['source']}_p{rec['page']}_t{rec['table']}_r{rec['row']}",
            "text": rec["text"],
            "metadata": {
                "source": rec["source"],
                "page": rec["page"],
                "table": rec["table"],
                "row": rec["row"],
                "table_type": rec["table_type"],
                "extract_mode": rec["extract_mode"],
                "parse_score": rec["parse_score"],
                "chunk_level": "row"
            }
        }
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # 表级 chunk（补充更大粒度的上下文）
    for rec in all_tables:
        doc = {
            "id": f"{rec['source']}_p{rec['page']}_t{rec['table']}_TABLE",
            "text": rec["table_summary"],
            "metadata": {
                "source": rec["source"],
                "page": rec["page"],
                "table": rec["table"],
                "table_type": rec["table_type"],
                "extract_mode": rec["extract_mode"],
                "parse_score": rec["parse_score"],
                "n_rows": rec["n_rows"],
                "headers": rec["headers"],
                "chunk_level": "table"
            }
        }
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")

print(f"[SAVED] documents.jsonl → {len(all_rows) + len(all_tables)} total chunks")
print("[DONE]")