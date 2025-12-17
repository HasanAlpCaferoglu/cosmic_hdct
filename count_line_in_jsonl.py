import os
import json
from dotenv import load_dotenv


def count_jsonl_lines(file_path: str) -> int:
    """
    Count the number of lines in a JSONL file.

    Each line in a JSONL file corresponds to a single JSON object.

    Args:
        file_path (str): Path to the JSONL file.

    Returns:
        int: Number of lines in the file.
    """
    line_count: int = 0
    with open(file_path, "r", encoding="utf-8") as file:
        for _ in file:
            line_count += 1
    return line_count


if __name__ == "__main__":

    load_dotenv()
    created_docs_dir = os.getenv("HDCT_DOCS_DIR")

    
    # JSONL_PATH = f"{created_docs_dir}/docs_2000000_4000000.jsonl"  # change this path if needed
    JSONL_PATH = f"{created_docs_dir}_v1/docs_1000000_4000000.jsonl"  # change this path if needed
    total_lines: int = count_jsonl_lines(JSONL_PATH)
    print(f"Number of lines: {total_lines}")