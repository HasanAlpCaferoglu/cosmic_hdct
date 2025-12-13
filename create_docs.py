import os
import torch
import argparse
import string
import numpy as np
from dotenv import load_dotenv
from transformers import AutoTokenizer
from typing import List, Dict
from model import CustomModel
from datetime import datetime
import pandas as pd
import csv
import string
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json

from transformers import AutoTokenizer


csv.field_size_limit(sys.maxsize)

def load_documents(docs_path, start_line, end_line):
    """
    Safely load MS MARCO document rows without column shifting.
    Always returns list of dicts: [{"doc_id":..., "title":..., "body":...}, ...]
    """

    # Always load raw with python engine to avoid column misalignment
    df = pd.read_csv(
        docs_path,
        sep="\t",
        header=None,
        names=["doc_id", "url", "title", "body"],
        dtype=str,
        engine="python",
        skiprows=start_line,
        nrows=end_line - start_line
    )

    # Clean doc_ids immediately
    df["doc_id"] = df["doc_id"].astype(str).str.strip()

    # Replace NaNs
    df = df.fillna("")

    return df.to_dict(orient="records")


def clean_state_dict(state_dict):
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", ""): v for k, v in state_dict.items()}
    return state_dict



def load_model_and_tokenizer(model_name, local_model_dir_name, model_class=CustomModel, device="cuda"):

    checkpoint_dirs = [d for d in os.listdir(local_model_dir_name) if d.startswith("checkpoint-epoch-")]
    if not checkpoint_dirs:
        raise FileNotFoundError("No checkpoints found")

    epoch_numbers = [int(d.split("-")[-1]) for d in checkpoint_dirs if d.split("-")[-1].isdigit()]
    latest_epoch = max(epoch_numbers)
    latest_checkpoint_dir = os.path.join(local_model_dir_name, f"checkpoint-epoch-{latest_epoch}")

    print("Loading model and tokenizer", flush=True)
    model = model_class(model_name)

    checkpoint = torch.load(os.path.join(latest_checkpoint_dir, "checkpoint.pth"), map_location="cpu")

    # ---- FIX: remove 'module.' prefix from DDP checkpoints ----
    state_dict = checkpoint["model_state_dict"]
    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(clean_state_dict)
    model.to(device)

    print(f"Loaded checkpoint from {latest_checkpoint_dir}, Epoch: {latest_epoch}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    passage_word_limit = int(0.6 * tokenizer.model_max_length)

    return model, tokenizer, passage_word_limit



def hdct_logits_to_scores(logits):
    """
    HDCT keeps only positive logits; negative ones become zero.
    No normalization across the document.
    """
    scores = np.array([float(l) for l in logits], dtype=np.float64)
    scores = np.maximum(scores, 0.0)  # clamp negatives to zero
    return scores


def merge_wordpieces_hdct(tokens, predictions):
    """
    HDCT-style merging:
    - Use only FIRST subword piece of each word.
    - Discard all '##' continuation pieces.
    - Clamp negative logits to zero.
    - If the same term appears multiple times, take MAX score.
    - Quantize using sqrt scaling.
    """
    SPECIAL = {'[CLS]', '[SEP]', '[PAD]'}
    PUNCT = set(string.punctuation)

    # 1) Convert predictions to floats + clamp negatives to zero
    def to_float(x):
        try:
            return float(x.item())
        except Exception:
            return float(x)

    raw_logits = [to_float(p) for p in predictions]
    raw_scores = hdct_logits_to_scores(raw_logits)

    # 2) Iterate tokens and take only the FIRST WORDPIECE
    word_list = []
    word_scores = []

    current_word = None
    current_score = None

    for tok, score in zip(tokens, raw_scores):

        tok = tok.strip()

        # Skip special tokens & punctuation
        if (not tok) or (tok in SPECIAL) or (tok in PUNCT):
            current_word = None
            continue

        # If token begins a new word
        if not tok.startswith("##"):
            # Close previous
            if current_word is not None:
                word_list.append(current_word)
                word_scores.append(current_score)

            # Start new word (HDCT takes ONLY this first piece)
            current_word = tok
            current_score = score

        else:
            # tok.startswith("##") → continuation piece
            # HDCT IGNORES continuation pieces entirely.
            continue

    # append final word
    if current_word is not None:
        word_list.append(current_word)
        word_scores.append(current_score)

    if not word_list:
        return {}

    # 3) Deduplicate terms by taking MAX score across occurrences (HDCT rule)
    term_to_scores = {}
    for w, s in zip(word_list, word_scores):
        if w not in term_to_scores:
            term_to_scores[w] = s
        else:
            term_to_scores[w] = max(term_to_scores[w], s)

    terms = list(term_to_scores.keys())
    scores = np.array([term_to_scores[t] for t in terms], dtype=np.float64)

    # 4) Quantize using sqrt (HDCT-like)
    term_weights = quantize_doc_terms(terms, scores, L=100)
    return term_weights


def quantize_doc_terms(words, scores, L=100):
    # HDCT uses sqrt scaling
    quant = np.rint(L * np.sqrt(scores)).astype(int)
    return {words[i]: int(quant[i]) for i in range(len(words))}


# HDCT - decay
def combine_passages_terms_and_weights(passages_terms_and_weights_list: List):
    doc_terms_and_weights = {}
    for idx, term_weight_dict in enumerate(passages_terms_and_weights_list):
        p_idx = idx + 1
        for term, weight in term_weight_dict.items():
            doc_terms_and_weights[term] = int(round(doc_terms_and_weights.get(term, 0) + weight / p_idx))
    return doc_terms_and_weights



def construct_doc_content(doc_terms_and_weights: Dict):
    new_doc_content_word_list = []
    for term, weight in doc_terms_and_weights.items():
        new_term_list = [term] * int(round(weight))
        new_doc_content_word_list.extend(new_term_list)
    
    new_doc_content = " ".join(new_doc_content_word_list)
    return new_doc_content


def get_model_predictions_batch(model, tokenizer, device, batch_word_lists):
    """
    batch_word_lists: list of lists of words
        Example: [
            ["this","is","passage","one"],
            ["another","passage","here"]
        ]

    Returns:
        tokens_list: list of token lists (B × seq_len_i)
        preds_list: list of predictions per passage (B × seq_len_i)
    """

    # Tokenize whole batch at once
    batch_enc = tokenizer(
        batch_word_lists,
        return_tensors="pt",
        is_split_into_words=True,
        padding=True,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        # model output shape: (batch_size, seq_len)
        logits = model(**batch_enc).cpu()

    input_ids = batch_enc["input_ids"].cpu().numpy()

    tokens_list = []
    preds_list = []

    # Split back per passage
    for i in range(len(batch_word_lists)):
        seq_len = (batch_enc["attention_mask"][i] == 1).sum().item()

        # Convert ids -> tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids[i][:seq_len])
        preds = logits[i][:seq_len]

        tokens_list.append(tokens)
        preds_list.append(preds)

    return tokens_list, preds_list


def document_processing(
    documents, output_jsonl_file, model, tokenizer,
    passage_word_limit, device, batch_size=16
):
    start = datetime.now()
    total = len(documents)
    processed = 0
    errors = 0

    with open(output_jsonl_file, "a", encoding="utf-8") as fout:
        for doc_obj in documents:
            try:
                doc_id = doc_obj["doc_id"]
                title = doc_obj.get("title", "") or ""
                body = doc_obj.get("body", "") or ""

                # Split into passages
                words = body.split()
                passages = [
                    words[i : i + passage_word_limit]
                    for i in range(0, len(words), passage_word_limit)
                ]

                # Run model batch inference
                all_passage_results = []
                for i in range(0, len(passages), batch_size):
                    batch_passages = passages[i : i + batch_size]

                    tokens_list, preds_list = get_model_predictions_batch(
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        batch_word_lists=batch_passages
                    )

                    for tks, preds in zip(tokens_list, preds_list):
                        merged = merge_wordpieces_hdct(tks, preds)
                        all_passage_results.append(merged)

                # Combine weighting outputs
                doc_terms_and_weights = combine_passages_terms_and_weights(
                    all_passage_results
                )

                # Convert to compact representation
                doc_content = construct_doc_content(doc_terms_and_weights)

                # ---- SAFE JSON WRITE ----
                json_record = {
                    "doc_id": doc_id,
                    "title": title,
                    "body": doc_content
                }
                fout.write(json.dumps(json_record, ensure_ascii=False) + "\n")

                processed += 1
                
                if processed > 20 and processed < 40:
                    print(title, flush=True)
                    filtered = {k: v for k, v in doc_terms_and_weights.items() if v != 0}
                    result = dict(sorted(filtered.items(), key=lambda item: item[1], reverse=True))
                    print(result, flush=True)
                    print("\n\n", flush=True)

                if processed % 10000 == 0:
                    print(f"{processed}/{total} documents processed. Duration: {datetime.now() - start}", flush=True)
                    fout.flush()
                    os.fsync(fout.fileno())

            except Exception as e:
                # Always log error
                print(f"[ERROR] Failed on doc: {doc_obj.get('doc_id')} | {e}", flush=True)
                errors += 1
                continue

    print(f"\nCompleted {processed} documents with {errors} errors.")
    print(f"Total duration: {datetime.now() - start}")    


def main(args):
    model_name = args.model_name
    
    start_line = args.line_start_row_index
    end_line = args.line_end_row_index
    
    print(f"Start Line: {start_line} End Line: {end_line}", flush=True)

    docs_path = os.getenv("MS_MARCO_DOCS_PATH")
    custom_model_docs = os.getenv("HDCT_DOCS_DIR")
    
    output_jsonl_file = f"{custom_model_docs}/docs_{start_line}_{end_line}.jsonl"
    
    start_line = int(start_line)
    end_line = int(end_line)

    # Determine device
    print(f"Cuda Available: {torch.cuda.is_available()}")
    # device = f"cuda:1" if torch.cuda.is_available() else "cpu"
    device = f"cuda" if torch.cuda.is_available() else "cpu"
    print("device: ", device, flush=True)

    model_dir_name = model_name.split("/")[1] + "_hdct_150k_optimized"
            
    # Load model and tokenizer
    model, tokenizer, passage_word_limit = load_model_and_tokenizer(model_name=model_name, local_model_dir_name=model_dir_name, device=device)
    
    """
    WE MAY NEED TO CHECK IF THEY USE DIFFERENT PASSAGE LENGTH
    """
    model_max_length = tokenizer.model_max_length
    passage_word_limit = int(0.6 * model_max_length)
        
    documents = load_documents(docs_path=docs_path, start_line=start_line, end_line=end_line)
    print(f"Total {len(documents)} documents are loaded.", flush=True)
    document_processing(documents, output_jsonl_file, model, tokenizer, passage_word_limit, device)


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser()
    # Arguments
    parser.add_argument("--model_name", default="google-bert/bert-base-uncased", help="Enter your model name")
    parser.add_argument("--line_start_row_index", default='0', type=str, help='Enter msmarco dataset start row for creating documents (inclusive)' )
    parser.add_argument("--line_end_row_index", default='500000', type=str, help='Enter msmarco dataset start row for creating documents (exclusive)' )
    args = parser.parse_args()
    main(args)
