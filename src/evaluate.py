""" 
src/evaluate.py 
COS760 Group 16 — Shared Evaluation Module 
Resego Morei (u20570326) 
Reusable evaluation function used across all Phase 3 and Phase 4 experiments. 
Usage: 
from src.evaluate import evaluate_model, log_result 
""" 
import os 
import csv 
import numpy as np 
import pandas as pd 
from scipy.stats import spearmanr 
from sklearn.metrics import mean_squared_error 
import torch 
from torch.utils.data import Dataset, DataLoader 
 
# ── Constants ────────────────────────────── 
RESULTS_LOG = os.path.join(os.path.dirname(__file__), '..', 'results', 'results_log.csv') 
DATA_ROOT   = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw') 
MAX_LEN     = 128 
BATCH_SIZE  = 32 
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
 
LOG_HEADERS = [ 
    'experiment_id', 'model', 'model_variant', 'language', 
    'split', 'spearman_rho', 'mse', 'augmented', 'notes' 
] 
 
# ── Dataset class ────────────────────────────── 
class SemRelDataset(Dataset): 
    """ 
    PyTorch Dataset for SemRel sentence pairs. 
    Tokenises sentence1 + sentence2 as a pair and returns label. 
    """ 
    def __init__(self, df, tokenizer, max_len=MAX_LEN): 
        self.df        = df.reset_index(drop=True) 
        self.tokenizer = tokenizer 
        self.max_len   = max_len 
 
    def __len__(self): 
        return len(self.df) 
 
    def __getitem__(self, idx): 
        row = self.df.iloc[idx] 
        enc = self.tokenizer( 
            str(row['sentence1']), 
            str(row['sentence2']), 
            truncation=True, 
            max_length=self.max_len, 
            padding='max_length', 
            return_tensors='pt' 
        ) 
        return { 
            'input_ids':      enc['input_ids'].squeeze(), 
            'attention_mask': enc['attention_mask'].squeeze(), 
            'label':          torch.tensor(float(row['label']), dtype=torch.float) 
        } 
 
# ── Core evaluation function ────────────────────────────── 
def evaluate_model(model, tokenizer, df, batch_size=BATCH_SIZE, device=DEVICE): 
    """ 
    Run inference on a DataFrame of sentence pairs and return 
    Spearman ρ, MSE, and the raw predictions. 
 
    Args: 
        model:      A PyTorch model with forward(input_ids, attention_mask) 
        tokenizer:  HuggingFace tokenizer matching the model 
        df:         pandas DataFrame with columns sentence1, sentence2, label 
        batch_size: inference batch size 
        device:     torch.device 
 
    Returns: 
        rho (float):   Spearman correlation between predictions and gold labels 
        mse (float):   Mean squared error 
        preds (list):  Raw model predictions 
        labels (list): Gold labels 
    """ 
    dataset = SemRelDataset(df, tokenizer) 
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False) 
 
    model.eval() 
    preds, labels = [], [] 
 
    with torch.no_grad(): 
        for batch in loader: 
            ids  = batch['input_ids'].to(device) 
            mask = batch['attention_mask'].to(device) 
            out  = model(ids, mask).cpu().numpy() 
            preds.extend(out.tolist()) 
            labels.extend(batch['label'].numpy().tolist()) 
 
    rho, _ = spearmanr(preds, labels) 
    mse    = mean_squared_error(labels, preds) 
    return float(rho), float(mse), preds, labels 
 
# ── Load split helper ────────────────────────────── 
def load_split(lang_code, split, data_root=DATA_ROOT): 
    """ 
    Load a SemRel CSV split. 
 
    Args: 
        lang_code: 'eng', 'afr', 'hau', or 'kin' 
        split:     'train', 'dev', or 'test' 
        data_root: path to data/raw/ 
 
    Returns: 
        pandas DataFrame or None if file not found 
    """ 
    path = os.path.join(data_root, lang_code, f'{split}.csv') 
    if not os.path.exists(path): 
        print(f"  [evaluate.py] File not found: {path}") 
        return None 
    return pd.read_csv(path) 
 
# ── Results logging ────────────────────────────── 
def log_result(experiment_id, model_name, model_variant, 
               language, split, spearman_rho, mse, 
               augmented=False, notes='', 
               results_log=RESULTS_LOG): 
    """ 
    Append one row to results/results_log.csv. 
 
    Args: 
        experiment_id:  e.g. 'TL-3-hau' 
        model_name:     e.g. 'afriberta_finetuned' 
        model_variant:  e.g. 'castorini/afriberta_large' 
        language:       e.g. 'Hausa' 
        split:          e.g. 'test' 
        spearman_rho:   float 
        mse:            float 
        augmented:      bool 
        notes:          optional string 
        results_log:    path to CSV log file 
    """ 
    os.makedirs(os.path.dirname(results_log), exist_ok=True) 
 
    row = { 
        'experiment_id': experiment_id, 
        'model':         model_name, 
        'model_variant': model_variant, 
        'language':      language, 
        'split':         split, 
        'spearman_rho':  round(spearman_rho, 4), 
        'mse':           round(mse, 4), 
        'augmented':     augmented, 
        'notes':         notes 
    } 
 
    write_header = not os.path.exists(results_log) 
    with open(results_log, 'a', newline='', encoding='utf-8') as f: 
        writer = csv.DictWriter(f, fieldnames=LOG_HEADERS) 
        if write_header: 
            writer.writeheader() 
        writer.writerow(row) 
 
    print(f"  ✓ Logged [{experiment_id}] {language} | " 
          f"ρ={spearman_rho:.4f} | MSE={mse:.4f} | aug={augmented}") 
 
 
# ── Combined convenience function ────────────────────────────── 
def evaluate_and_log(model, tokenizer, lang_code, lang_name, 
                     split, experiment_id, model_name, model_variant, 
                     augmented=False, notes='', 
                     data_root=DATA_ROOT, device=DEVICE): 
    """ 
    Load a split, run evaluation, print results, and log to CSV. 
    This is the main function called from all Phase 3 and Phase 4 experiments. 
 
    Args: 
        model:          PyTorch model 
        tokenizer:      HuggingFace tokenizer 
        lang_code:      'eng', 'afr', 'hau', or 'kin' 
        lang_name:      display name e.g. 'Hausa' 
        split:          'train', 'dev', or 'test' 
        experiment_id:  e.g. 'TL-3-hau' 
        model_name:     e.g. 'afriberta_zeroshot' 
        model_variant:  e.g. 'castorini/afriberta_large' 
        augmented:      bool 
        notes:          optional string 
 
    Returns: 
        rho (float), mse (float) 
    """ 
    df = load_split(lang_code, split, data_root) 
    if df is None: 
        print(f"  Skipping {lang_name} {split} — no data found.") 
        return None, None 
 
    print(f"\nEvaluating {model_name} on {lang_name} ({split}, n={len(df)})...") 
    rho, mse, preds, labels = evaluate_model(model, tokenizer, df, device=device) 
    print(f"  Spearman ρ = {rho:.4f} | MSE = {mse:.4f}") 
 
    log_result( 
        experiment_id=experiment_id, 
        model_name=model_name, 
        model_variant=model_variant, 
        language=lang_name, 
        split=split, 
        spearman_rho=rho, 
        mse=mse, 
        augmented=augmented, 
        notes=notes 
    ) 
    return rho, mse 
 
 
# ── Sanity check ────────────────────────────── 
def sanity_check(model, tokenizer, expected_min_rho=0.70, 
                 data_root=DATA_ROOT, device=DEVICE): 
    """ 
    Phase 2.3 sanity check: evaluate English-fine-tuned model on English dev. 
    Spearman ρ should be >= expected_min_rho (default 0.70). 
 
    Returns True if check passes, False otherwise. 
    """ 
    print("\nRunning sanity check on English dev set...") 
    df = load_split('eng', 'dev', data_root) 
    if df is None: 
        print("  ✗ English dev set not found.") 
        return False 
 
    rho, mse, _, _ = evaluate_model(model, tokenizer, df, device=device) 
    print(f"  English dev: ρ={rho:.4f}, MSE={mse:.4f}") 
 
    if rho >= expected_min_rho: 
        print(f"  ✓ Sanity check passed (ρ={rho:.4f} >= {expected_min_rho})") 
        return True 
    else: 
        print(f"  ✗ Sanity check FAILED (ρ={rho:.4f} < {expected_min_rho})") 
        print("    Check the training loop before proceeding to Phase 3.") 
        return False 
 
# ── Print results summary ────────────────────────────── 
def print_results_summary(results_log=RESULTS_LOG): 
    """ 
    Print a formatted summary of all logged results so far. 
    """ 
    if not os.path.exists(results_log): 
        print("No results logged yet.") 
        return 
 
    df = pd.read_csv(results_log) 
    print("\n" + "="*75) 
    print(f"  RESULTS SUMMARY  ({len(df)} experiments logged)") 
    print("="*75) 
    print(f"  {'Experiment':<14} {'Model':<22} {'Language':<13} " 
          f"{'ρ':>7} {'MSE':>7} {'Aug':>5}") 
    print("-"*75) 
    for _, row in df.iterrows(): 
        print(f"  {str(row['experiment_id']):<14} " 
              f"{str(row['model']):<22} " 
              f"{str(row['language']):<13} " 
              f"{float(row['spearman_rho']):>7.4f} " 
              f"{float(row['mse']):>7.4f} " 
              f"{'Yes' if row['augmented'] else 'No':>5}") 
    print("="*75) 
 
# ── Run as script for sanity check ────────────────────────────── 
if __name__ == '__main__': 
    """ 
    Quick test: print current results summary. 
    Run from project root: py src/evaluate.py 
    """ 
    print_results_summary()