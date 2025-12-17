import os
import json
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()
    created_docs_dir = os.getenv("HDCT_DOCS_DIR")
    
    input_files = [
        f"{created_docs_dir}/docs_0_2000000.jsonl.jsonl",
        f"{created_docs_dir}/docs_2000000_3213835.jsonl",
    ]

    output_file = f"{created_docs_dir}/docs_combined.jsonl"

    with open(output_file, "w", encoding="utf-8") as fout:
        for fname in input_files:
            print(f"Merging {fname} ...")
            with open(fname, "r", encoding="utf-8") as fin:
                for line in fin:
                    if not line.strip():
                        continue
                    try:
                        json.loads(line)   # validate JSON
                    except Exception as e:
                        print(f"ERROR in file {fname}: {e}")
                        continue
                    fout.write(line)

    print("Merging complete:", output_file)