import camelot
import os
import pickle
import warnings
warnings.filterwarnings("ignore")
import time
time.sleep(1)

pdf_dir = "../data/aluminium"
all_table_rows = []

for pdf_file in os.listdir(pdf_dir):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_dir, pdf_file)

        print(f"[INFO] Processing {pdf_file}...")

        try:
            tables = camelot.read_pdf(pdf_path, pages='all')

            for t_idx, table in enumerate(tables):
                df = table.df

                for r_idx, row in df.iterrows():
                    # 把一行转成文本
                    row_text = " | ".join(row.astype(str))

                    all_table_rows.append({
                        "source": pdf_file,
                        "table": t_idx,
                        "row": r_idx,
                        "text": row_text
                    })

        except Exception as e:
            print(f"[ERROR] {pdf_file} failed:", e)

# 保存
with open("../output/table_rows.pkl", "wb") as f:
    pickle.dump(all_table_rows, f)

print(f"[DONE] Extracted {len(all_table_rows)} table rows")
