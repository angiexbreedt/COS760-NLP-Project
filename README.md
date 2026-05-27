# COS760 Group 16 - Semantic Relatedness Using SemRel Datasets
**Evaluating Cross-Lingual Transfer and Data Augmentation for Semantic Relatedness in Afrikaans, Hausa and Kinyarwanda**

### 👥 Team Members
- Angelique Breedt (u23542838)
- Christopher Yoko (u22857941)
- Resego Morei (u20570326)

---

## 📌 Project Overview
Semantic relatedness measures how meaningfully connected two pieces of text are. This project investigates how well NLP models capture semantic relatedness in low-resource African languages.

We focus on:
- Afrikaans
- Hausa
- Kinyarwanda

Using the **SemRel dataset**, we evaluate:
- Cross-lingual transfer (train on English → test on African languages)
- Multilingual vs African-specific models
- Impact of data augmentation
---


## 🎯 Research Questions
1. How well do models trained on English transfer to African languages?
2. How do multilingual models compare to African-focused models?
3. Does data augmentation improve performance in low-resource settings?

---

## 🧠 Methods
- Sentence embeddings (Sentence-BERT, XLM-R, AfriBERTa)
- Cosine similarity for semantic relatedness
- Fine-tuning on English and multilingual data
- Data augmentation (back-translation, paraphrasing)

---

## 📊 Evaluation Metrics
- Spearman Correlation (primary)
- Mean Squared Error (MSE)

---

## Project Structure

```
COS760-NLP-Project/
├── data/
│   ├── raw/                        # SemRel dataset splits (CSV format)
│   │   ├── afr/  dev.csv, test.csv
│   │   ├── eng/  train.csv, dev.csv, test.csv
│   │   ├── hau/  train.csv, dev.csv, test.csv
│   │   ├── kin/  train.csv, dev.csv, test.csv
│   │   └── dataset_summary.json
│   └── augmented/                  # Back-translated data (see README inside)
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory data analysis
│   ├── 02_baseline.ipynb           # Cosine similarity baseline
│   ├── 03-xlmr-finetuning.ipynb   # XLM-R fine-tuning (run on Kaggle GPU)
│   ├── 04-afriberta.ipynb          # AfriBERTa fine-tuning (run on Kaggle GPU)
│   └── 05-augmentation.ipynb       # Back-translation augmentation (run on Kaggle GPU)
├── src/
│   ├── evaluate.py                 # Shared evaluation module (Spearman ρ, MSE, logging)
│   ├── train.py                    # Training reference (see notebooks for full implementation)
│   └── augment.py                  # Augmentation reference (see notebooks for full implementation)
├── results/
│   ├── results_log.csv             # All experiment results (19 experiments)
│   ├── dataset_statistics.csv      # EDA statistics per language/split
│   ├── fig1_score_distributions.png
│   ├── fig2_sentence_lengths.png
│   └── fig3_baseline_results.png
├── requirements.txt
└── README.md
```
--- 
## Dataset

All data comes from the [SemRel2024 dataset](https://huggingface.co/datasets/SemRel/SemRel2024)
(Abdulmumin et al., 2024), downloaded from Hugging Face.

| Language    | Code | Train | Dev | Test |
|-------------|------|-------|-----|------|
| English     | eng  | 5500  | 250 | 2600 |
| Afrikaans   | afr  | -     | 375 | 375  |
| Hausa       | hau  | 1736  | 212 | 603  |
| Kinyarwanda | kin  | 778   | 102 | 222  |

> **Note on Afrikaans:** No training split exists. Zero-shot cross-lingual transfer only.

---
## Installation
 
```bash
git clone https://github.com/angiexbreedt/COS760-NLP-Project.git
cd COS760-NLP-Project
pip install -r requirements.txt
```
 
On Windows:
```powershell
py -m pip install -r requirements.txt
```
 
---

## Running the Experiments
 
### 1. Exploratory Data Analysis (local, no GPU needed)
```bash
py -m notebook notebooks/01_eda.ipynb
```
Produces score distribution plots, sentence length histograms, and dataset statistics saved to `results/`.
 
### 2. Cosine Similarity Baseline (local, no GPU needed)
```bash
py -m notebook notebooks/02_baseline.ipynb
```
Runs the unsupervised baseline on all four languages and logs results to `results/results_log.csv`.
 
### 3. XLM-R Fine-Tuning (requires GPU - run on Kaggle)
Open `notebooks/03-xlmr-finetuning.ipynb` on Kaggle with GPU T4 x2 enabled.  
Trains XLM-R on English, Hausa, and Kinyarwanda. Evaluates zero-shot on all African languages.
 
### 4. AfriBERTa Fine-Tuning (requires GPU - run on Kaggle)
Open `notebooks/04-afriberta.ipynb` on Kaggle with GPU T4 x2 enabled.  
Trains AfriBERTa on English, Hausa, and Kinyarwanda. Evaluates zero-shot on all African languages.
 
### 5. Data Augmentation (requires GPU - run on Kaggle)
Open `notebooks/05-augmentation.ipynb` on Kaggle with GPU T4 x2 enabled.  
Performs back-translation augmentation for Hausa and Kinyarwanda, retrains XLM-R and AfriBERTa.
 
**Note:** Notebooks 3–5 require a Kaggle account with phone verification for GPU access. The SemRel dataset must be uploaded as a Kaggle dataset named `semrel2024-raw`.
 
---

## Results

All results logged in `results/results_log.csv`.
Key results:
 
| Experiment | Model | Language | Spearman ρ |
|---|---|---|---|
| BL-1 | Cosine baseline | Afrikaans | 0.7646 |
| TL-1 | XLM-R zero-shot | Afrikaans | **0.8177** |
| TL-2 | XLM-R fine-tuned | Hausa | 0.6862 |
| TL-2 | XLM-R fine-tuned | Kinyarwanda | 0.6339 |
| AU-1 | XLM-R + augmentation | Hausa | 0.6880 |
 
See the full report (`report/Group16_u23542838.pdf`) for complete results and analysis.
 
---

## Evaluation Module
 
`src/evaluate.py` provides reusable functions for running inference and logging results:
 
```python
from src.evaluate import evaluate_and_log, print_results_summary
 
# Print all logged results
print_results_summary()
 
# Evaluate a model and log results
evaluate_and_log(model, tokenizer, 
                 lang_code='hau', lang_name='Hausa',
                 split='test', experiment_id='TL-2-hau',
                 model_name='xlmr_finetuned',
                 model_variant='xlm-roberta-base')
```
 
---
 
## Checkpoints
 
Model checkpoints are saved as Kaggle notebook output versions and are not tracked in this repository due to file size constraints (~1GB per checkpoint). To reproduce:
- Run the relevant Kaggle notebook
- Checkpoints are saved to `/kaggle/working/checkpoints/` during the session
---

## 🧪 Experiments
- Baseline (no training)
- Fine-tuned models
- Cross-lingual evaluation
- Data augmentation experiments

---

## ⚠️ Challenges
- Limited data for some languages
- Translation quality for augmentation
- Computational constraints

---

## References

Abdulmumin, I., et al. (2024). SemRel: A collection of semantic textual relatedness datasets for 13 languages. *ACL 2024*.
 
Conneau, A., et al. (2020). Unsupervised cross-lingual representation learning at scale. *ACL 2020*.
 
Alabi, J., et al. (2022). Adapting pretrained language models to African languages via multilingual adaptive fine-tuning. *COLING 2022*.
 
Reimers, N. and Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *EMNLP 2019*.



