import os
import json
from dotenv import load_dotenv



def write_first_n_lines(
    input_path: str,
    output_path: str,
    n_lines: int
) -> None:
    """
    Write the first N lines from a JSONL file to a new JSONL file.

    Args:
        input_path (str): Path to the input JSONL file.
        output_path (str): Path to the output JSONL file.
        n_lines (int): Number of lines to copy.
    """
    written: int = 0

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:

        for line in infile:
            if written >= n_lines:
                break
            outfile.write(line)
            written += 1


if __name__ == "__main__":

    load_dotenv()
    created_docs_dir = os.getenv("HDCT_DOCS_DIR")



    INPUT_JSONL = f"{created_docs_dir}/docs_2000000_4000000.jsonl"  # change this path if needed
    OUTPUT_JSONL = f"{created_docs_dir}/docs_2000000_3213835.jsonl"  # change this path if needed

    N_LINES: int = 1213835               # number of lines to keep

    write_first_n_lines(INPUT_JSONL, OUTPUT_JSONL, N_LINES)
    print(f"Wrote first {N_LINES} lines to {OUTPUT_JSONL}")
