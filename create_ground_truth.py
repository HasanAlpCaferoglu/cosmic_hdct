import os
import json
import lucene

from dotenv import load_dotenv
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.analysis.tokenattributes import CharTermAttribute, OffsetAttribute
from java.io import StringReader
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datetime import datetime
import pandas as pd

import csv
csv.field_size_limit(sys.maxsize)

def load_documents(docs_path, sample_size=150000, seed=42):
    """
    Safely load MS MARCO document rows.
    Randomly select 60K documents that HAVE a title.
    Returns list of dicts: [{"doc_id":..., "title":..., "body":...}, ...]
    """

    # Load using python engine for safety
    df = pd.read_csv(
        docs_path,
        sep="\t",
        header=None,
        names=["doc_id", "url", "title", "body"],
        dtype=str,
        engine="python"
    )

    # Clean doc_id
    df["doc_id"] = df["doc_id"].astype(str).str.strip()

    # Replace NaNs
    df = df.fillna("")

    # -------- FILTER: keep only docs WITH a title --------
    df = df[df["title"].str.strip().str.len() > 3]

    # -------- RANDOM SAMPLE 60K --------
    # If fewer than 60K available, sample all
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed)
    else:
        print(f"Warning: Only {len(df)} docs have titles; returning all.")

    # -------- Return as list of dicts --------
    return df.to_dict(orient="records")



def analyze_with_original(analyzer, text):
    """
    Returns:
       analyzed_tokens: list of normalized tokens
       analyzed_to_original: dict mapping analyzed_token → set(original_strings)
    """
    stream = analyzer.tokenStream("field", StringReader(text))
    term_attr = stream.addAttribute(CharTermAttribute.class_)
    offset_attr = stream.addAttribute(OffsetAttribute.class_)
    stream.reset()

    analyzed_tokens = []
    mapping = {}  # stem → set(original words)

    while stream.incrementToken():
        analyzed = term_attr.toString()
        start = offset_attr.startOffset()
        end = offset_attr.endOffset()
        original_word = text[start:end].lower()

        analyzed_tokens.append(analyzed)

        if analyzed not in mapping:
            mapping[analyzed] = set()
        mapping[analyzed].add(original_word)

    stream.end()
    stream.close()

    return analyzed_tokens, mapping


def title_terms_in_document(document_text: str, title: str) -> dict:
    """
    Uses EnglishAnalyzer to match title terms semantically
    but returns ORIGINAL document words, not analyzed tokens.
    """
    analyzer = EnglishAnalyzer()

    # Analyze title
    title_tokens, _ = analyze_with_original(analyzer, title)

    # Analyze document, but keep mapping analyzed → original words
    doc_tokens, doc_map = analyze_with_original(analyzer, document_text)

    doc_token_set = set(doc_tokens)

    result = {}

    # Match title stems to document stems
    for stem in title_tokens:
        if stem in doc_token_set:
            # add *all original document words* that match this stem
            for orig in doc_map.get(stem, []):
                result[orig] = 1

    return result


def main():  
    docs_path = os.getenv("MS_MARCO_DOCS_PATH")
    ground_truth_dir = os.getenv("HDCT_GROUND_TRUTH_FOLDER")
       
    start = datetime.now()
    
    train_jsonl_path = f"{ground_truth_dir}/train_150000.json"
    
    documents = load_documents(docs_path=docs_path)
    print(f"Total {len(documents)} documents are loaded.", flush=True)

    line_index = 0
    train_doc_dicts = []
    for document_obj in documents:
        try:
            doc_id = document_obj["doc_id"]
            document_text = document_obj["body"]
            doc_title = document_obj["title"]

            doc_terms_and_weights = title_terms_in_document(document_text=document_text, title=doc_title)

            doc_obj = {
                "doc_id": doc_id,
                "title": doc_title,
                "body": document_text,
                "term_weights": doc_terms_and_weights
            }
            
            
            train_doc_dicts.append(doc_obj)
                
            line_index += 1
            
            if 20 < line_index and line_index < 40:
                print(doc_title, flush=True)
                filtered_values = {k: v for k, v in doc_terms_and_weights.items() if v != 0}
                print(dict(sorted(filtered_values.items(), key=lambda item: item[1], reverse=True)), flush=True)
                print("\n\n", flush=True)
            
            if line_index % 10000 == 0:
                print(f"10000 documents are created.", flush=True)
                end = datetime.now()
                print(f"Duration: {(end - start)}", flush=True)
                            
        except Exception as e:
            print(f"Failed in indexing document {e}", flush=True)
            continue
    
    # Save the lists to JSON files
    with open(train_jsonl_path, 'w') as train_file:
        json.dump(train_doc_dicts, train_file, indent=4)
    print(f"The train split is saved to {train_jsonl_path}", flush=True)

    end = datetime.now()
    print(f"Duration: {(end - start)}", flush=True)
    
    return


if __name__ == "__main__":
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    print('lucene', lucene.VERSION)
    
    load_dotenv()
    main()