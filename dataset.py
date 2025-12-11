import json
import torch
from torch.utils.data import Dataset

import ijson

class MSMarcoDataset(Dataset):
    def __init__(self, dataset_json_path, max_word_cnt, tokenizer, doc_start_index, doc_end_index):
        print("--> Constructing MSMarcoDataset...", flush=True)
        self.dataset_json_path = dataset_json_path
        self.max_word_cnt = max_word_cnt
        self.tokenizer = tokenizer
        self.doc_start_index = doc_start_index
        self.doc_end_index = doc_end_index
        self.gt_passage_dicts = self.get_gt_passage_dict_list()
        print("--> End of MSMarcoDataset construction.", flush=True)

    def load_dataset(self):
        subset = []
        with open(self.dataset_json_path, "r") as f:
            raw_data = f.read()

        # Replace unquoted NaN with null
        import re
        import io
        raw_data = re.sub(r'\bNaN\b', 'null', raw_data)

        file_like = io.StringIO(raw_data)
        objects = ijson.items(file_like, 'item')
        for idx, obj in enumerate(objects):
            if idx >= self.doc_end_index:
                break
            if idx >= self.doc_start_index:
                subset.append(obj)
        print(f"------ Docs [{self.doc_start_index}:{self.doc_end_index}] are loaded. Doc Count: {len(subset)}", flush=True)
        return subset
    
    def get_gt_passage_dict_list(self):
        """
        After computing term weights in document level, we splite the docs into passages because of context window limit

        """
        gt_doc_dicts = self.load_dataset()
        gt_passage_dicts = []
        doc_no = 0
        for doc_dict in gt_doc_dicts:
            doc_content = doc_dict["body"]
            doc_id = doc_dict["doc_id"]
            doc_term_weights = doc_dict["cosine_values"]
            doc_content_word_list = doc_content.split(" ")
            if len(doc_content_word_list) > self.max_word_cnt:
                passage_internal_id = 0
                start_word_index = 0
                end_word_index = start_word_index + self.max_word_cnt
                while start_word_index < len(doc_content_word_list):
                    passage_id = doc_id + f"_{passage_internal_id}"
                    passage_content_word_list = doc_content_word_list[start_word_index : end_word_index]
                    passage_content = " ".join(passage_content_word_list)
                    passage_dict = {
                        'passage_id': passage_id,
                        'body': passage_content,
                        'term_weights': doc_term_weights
                    }
                    gt_passage_dicts.append(passage_dict)
                    passage_internal_id += 1
                    start_word_index = end_word_index
                    end_word_index = end_word_index + self.max_word_cnt
            else:
                passage_dict = {
                        'passage_id': doc_id,
                        'body': doc_content,
                        'term_weights': doc_term_weights
                }
                gt_passage_dicts.append(passage_dict)

            doc_no +=1 

        return gt_passage_dicts

    
    def tokenize_and_align_weights(self, passage, term_weights):
        """ 
        Aligns ground truth term weights with tokenized passage tokens.
        Handles subword tokenization by assigning weights to the first subword to avoid double-counting
        """
        # print(f"content: {passage}")
        aligned_weights = []
        tokenized_passage = self.tokenizer(
            passage,
            return_tensors="pt",
            max_length=self.tokenizer.model_max_length,  # Ensure truncation
            truncation=True,  # Truncate sequences longer than max_length
            padding=False  # Do not pad here (handled in collate_fn)
        )
        passage_tokens = self.tokenizer.convert_ids_to_tokens(tokenized_passage["input_ids"].squeeze(0))

        # Determine subword marker if applicable
        subword_prefix = "##"  # Default for BERT-like tokenizers
        
        # To reconstruct terms for matching with term_weights
        reconstructed_term = ""
        # is_continuing_word = False
        continuing_word_cnt = 0

        for token_idx, token in enumerate(passage_tokens):              
            if token.startswith(subword_prefix):
                reconstructed_term += token[len(subword_prefix):]  # Append subword without subword_prefix
                continuing_word_cnt += 1
                aligned_weights.append(0)  # Add 0 for subsequent subwords
            elif continuing_word_cnt > 0:
                # If a complete word has ended, finalize its weight
                weight = term_weights.get(reconstructed_term, 0)
                aligned_weights[-(continuing_word_cnt+1)] = weight  # Update the first subword weight
                reconstructed_term = token  # Start a new term
                continuing_word_cnt = 0
                weight = term_weights.get(reconstructed_term, 0)
                aligned_weights.append(weight)  # Add 0 for the new token (until it gets a weight). Token could be a complete word or initial subword of a word
            else:
                # New word (not a continuation)
                reconstructed_term = token
                weight = term_weights.get(reconstructed_term, 0)
                aligned_weights.append(weight)
            
        
        # Handle the last reconstructed term
        if continuing_word_cnt>0:
            weight = term_weights.get(reconstructed_term, 0)
            aligned_weights[-(continuing_word_cnt+1)] = weight  # Update the last weight

        return tokenized_passage, aligned_weights, passage_tokens


    def __len__(self):
        return len(self.gt_passage_dicts)
    
    def __getitem__(self, idx):
        """
        Returns a single data item containing the tokenized input, attention mask,
        and aligned term weights for a document at the given index.
        """

        # Get the passage object
        passage_obj = self.gt_passage_dicts[idx]
        content = passage_obj["body"]
        term_weights = passage_obj["term_weights"]

        # Tokenize the document content
        tokenized_passage, aligned_weights, passage_tokens = self.tokenize_and_align_weights(content, term_weights)

        # Convert input_ids and attention_mask to tensors
        input_ids = tokenized_passage["input_ids"].squeeze(0)
        attention_mask = tokenized_passage["attention_mask"].squeeze(0)

        # Convert aligned_weights to a tensor
        aligned_weights_tensor = torch.tensor(aligned_weights, dtype=torch.float)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "term_weights": aligned_weights_tensor
        }
    