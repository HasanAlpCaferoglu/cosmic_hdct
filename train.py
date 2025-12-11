import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence

import os
import argparse
import time
from dotenv import load_dotenv
from transformers import AutoTokenizer
from model import CustomModel
from dataset import MSMarcoDataset
from transformers import get_linear_schedule_with_warmup



def train(model, tokenizer, train_loader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0.0
    start_time = time.time()
    criterion = torch.nn.MSELoss()

    print(f"\n[Epoch {epoch}] Starting training... "
          f"Total batches: {len(train_loader)} | ", flush=True)
    
    special_ids = {tokenizer.cls_token_id, tokenizer.sep_token_id}

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        term_weights = batch['term_weights'].to(device)
        
        valid_mask = attention_mask.clone().bool()
        for sid in special_ids:
            valid_mask &= (input_ids != sid)

        optimizer.zero_grad()

        logits = model(input_ids, attention_mask)  # raw logits per token

        loss = criterion(logits, term_weights)

        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        # 7. Logging
        if (batch_idx + 1) % 1000 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            elapsed = time.time() - start_time
            print(
                f"  Batch {batch_idx+1}/{len(train_loader)} "
                f"- Loss: {loss.item():.4f} (Avg: {avg_loss:.4f}) "
                f"- {elapsed:.1f}s",
                flush=True,
            )

    epoch_time = time.time() - start_time
    avg_loss = total_loss / max(1, len(train_loader))
    print(
        f"[Epoch {epoch}] Training complete. "
        f"Avg Loss: {avg_loss:.4f} - Time Taken: {epoch_time:.2f}s\n",
        flush=True,
    )
    return avg_loss



def evaluate(model, tokenizer, val_loader, device, epoch="Final(test)"):
    model.eval()
    total_loss = 0.0
    criterion = torch.nn.MSELoss()
    print(
        f"[Epoch {epoch}] Starting evaluation... "
        f"Total batches: {len(val_loader)} | ",
        flush=True,
    )
    
    special_ids = {tokenizer.cls_token_id, tokenizer.sep_token_id}

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            term_weights = batch["term_weights"].to(device)
            
            valid_mask = attention_mask.clone().bool()
            for sid in special_ids:
                valid_mask &= (input_ids != sid)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)

            loss = criterion(logits, term_weights)
            
            total_loss += float(loss.item())

            if (batch_idx + 1) % 1000 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(
                    f"  Batch {batch_idx + 1}/{len(val_loader)} "
                    f"- Loss: {loss.item():.4f} (Avg: {avg_loss:.4f}) ",
                    flush=True,
                )

    n = max(1, len(val_loader))
    return total_loss / n




def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences by padding.
    """
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    term_weights = [item["term_weights"] for item in batch]

    # Pad sequences to the same length
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    padded_attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    padded_term_weights = pad_sequence(term_weights, batch_first=True, padding_value=0)

    return {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask,
        "term_weights": padded_term_weights
    }

def create_dataloader(file_path, max_word_cnt, tokenizer, batch_size, doc_start_index, doc_end_index):
    """Create the dataset and return a DataLoader."""
    print(f"-> Dataset creation for {file_path}", flush=True)
    dataset = MSMarcoDataset(file_path, max_word_cnt, tokenizer, doc_start_index=doc_start_index, doc_end_index=doc_end_index)

    dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                collate_fn=collate_fn)
    return dataloader


def save_checkpoint(model, tokenizer, optimizer, scheduler, epoch, local_model_dir_name):
    """
    Save the model checkpoint locally.

    Args:
        model: The model to save.
        tokenizer: The tokenizer to save.
        optimizer: The optimizer whose state will be saved.
        epoch: The current epoch number.
        local_model_dir_name: Directory to save the checkpoint locally
    """
    # Save locally
    model_dir = os.path.join(local_model_dir_name, f"checkpoint-epoch-{epoch}")
    os.makedirs(model_dir, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,  
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
    }
    
    torch.save(checkpoint, os.path.join(model_dir, "checkpoint.pth"))
    tokenizer.save_pretrained(model_dir)

    print(f"Checkpoint saved locally at {model_dir}", flush=True)



def get_param_groups(model, weight_decay=0.01):
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]

    param_groups = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return param_groups



def build_optimizer(model, train_loader, epochs, lr=2e-5, weight_decay=0.01):

    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)  # 10% warmup
    
    print(f"Total Steps: {total_steps}\tWarmup Steps: {warmup_steps}")

    optimizer = AdamW(
        get_param_groups(model, weight_decay),
        lr=lr
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    return optimizer, scheduler



def main(args):
    # Each process gets its own GPU device
    device = f"cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Device: {device}")

    ground_truth_dir = os.getenv("HDCT_GROUND_TRUTH_FOLDER")
    model_name = args.model_name
    batch_size = args.batch_size

    train_path = f"{ground_truth_dir}/train_150000.json"
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_max_length = tokenizer.model_max_length
    passage_word_limit = int(0.6 * model_max_length)

    # Create model and move it to this GPU
    model = CustomModel(model_name)
    model = model.to(device)

    # Create data loaders
    train_loader = create_dataloader(train_path, passage_word_limit, tokenizer,
                                     batch_size=batch_size, doc_start_index=0, doc_end_index=130000)
    val_loader = create_dataloader(train_path, passage_word_limit, tokenizer,
                                   batch_size=batch_size, doc_start_index=130000, doc_end_index=150000)

    # Directory for saving checkpoints
    model_dir_name = model_name.split("/")[1] + "_hdct_150k_optimized"
    
    if not os.path.exists(model_dir_name):
        os.makedirs(model_dir_name)
        print(f"Local directory {model_dir_name} is created.", flush=True)
        
    epochs = 3
    
    optimizer, scheduler = build_optimizer(
        model, train_loader, epochs
    )

    # Simple training loop
    for epoch in range(epochs):
        print(f"\n=== Starting Epoch {epoch+1}/{epochs} ===", flush=True)
        train_loss = train(model, tokenizer, train_loader, optimizer, scheduler, device, epoch)
        val_loss   = evaluate(model, tokenizer, val_loader, device, epoch)

        print(f"=== Epoch {epoch + 1} Summary ===", flush=True)
        print(f"  Train Loss: {train_loss:.4f}", flush=True)
        print(f"  Validation Loss: {val_loss:.4f}", flush=True)

        save_checkpoint(model, tokenizer, optimizer, scheduler, epoch + 1, model_dir_name)
        

if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser()
    # Arguments
    parser.add_argument("--model_name", default="google-bert/bert-base-uncased", help="Enter your model name")
    parser.add_argument("--batch_size", default=16, type=int, help='Batch Size' )
    parser.add_argument("--learning_rate", default=0.00002, type=float, help='Learning Rate')
    args = parser.parse_args()
    main(args)