# COS760 Group 16 — Semantic Relatedness Project
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
cos760_group16/
├── data/
│   ├── raw/
│   │   ├── afr/        # Afrikaans (dev.csv, test.csv)
│   │   ├── eng/        # English (train.csv, dev.csv, test.csv)
│   │   ├── hau/        # Hausa (train.csv, dev.csv, test.csv)
│   │   ├── kin/        # Kinyarwanda (train.csv, dev.csv, test.csv)
│   │   └── dataset_summary.json
│   └── augmented/      # Back-translated data (generated in Phase 4)
├── notebooks/
│   ├── 01_eda.ipynb            # Phase 1.4: Exploratory data analysis
│   ├── 02_baseline.ipynb       # Phase 2.1: Cosine similarity baseline
│   └── 03_error_analysis.ipynb # Phase 5.2: Error analysis
├── src/
│   ├── train.py        # Fine-tuning loop (XLM-R / AfriBERTa)
│   ├── evaluate.py     # Shared evaluation function
│   └── augment.py      # Back-translation augmentation pipeline
├── results/
│   └── results_log.csv # All experiment outputs (auto-populated)
├── checkpoints/        # Saved model checkpoints
├── report/             # Final report .docx
├── requirements.txt
└── README.md
```

## Dataset

All data comes from the [SemRel2024 dataset](https://huggingface.co/datasets/SemRel/SemRel2024)
(Abdulmumin et al., 2024), downloaded from Hugging Face.

| Language    | Code | Train | Dev | Test |
|-------------|------|-------|-----|------|
| English     | eng  | 5500  | 250 | 2600 |
| Afrikaans   | afr  | —     | 375 | 375  |
| Hausa       | hau  | 1736  | 212 | 603  |
| Kinyarwanda | kin  | 778   | 102 | 222  |

> **Note on Afrikaans:** No training split exists. Zero-shot cross-lingual transfer only.

## Installation

```bash
pip install -r requirements.txt
```

## Running the Code

### 1. EDA (Phase 1)
```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 2. Baseline (Phase 2)
```bash
python src/evaluate.py --model cosine --lang eng
```

### 3. Fine-tuning (Phase 3)
```bash
# Fine-tune XLM-R on English
python src/train.py --model xlmr --lang eng --epochs 10

# Evaluate zero-shot on Afrikaans, Hausa, Kinyarwanda
python src/evaluate.py --model xlmr --checkpoint checkpoints/xlmr_eng_best --lang afr
python src/evaluate.py --model xlmr --checkpoint checkpoints/xlmr_eng_best --lang hau
python src/evaluate.py --model xlmr --checkpoint checkpoints/xlmr_eng_best --lang kin
```

### 4. Augmentation (Phase 4)
```bash
python src/augment.py --lang hau --quality_threshold 0.75
python src/augment.py --lang kin --quality_threshold 0.75
```

## Results

All results are logged to `results/results_log.csv` automatically. 
See the final report (`report/`) for the full results table and analysis.

---

## ⚙️ Setup Instructions

### 1. Clone the repo

git clone https://github.com/YOUR-USERNAME/COS760-NLP-Project.git

cd COS760-NLP-Project

### 2. Create virtual environment

python -m venv venv
source venv/bin/activate # Linux/Mac
venv\Scripts\activate # Windows

### 3. Install dependencies

pip install -r requirements.txt

---

## 🚀 Quick Start (Baseline)
Run baseline semantic similarity using pretrained embeddings:

python src/baseline.py

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

Abdulmumin, I., et al. (2024). SemRel: A collection of semantic textual 
relatedness datasets for 13 languages. In *Proceedings of ACL 2024*.

Conneau, A., et al. (2020). Unsupervised cross-lingual representation 
learning at scale. In *Proceedings of ACL 2020*.

Alabi, J., et al. (2022). Adapting pretrained language models to African 
languages via multilingual adaptive fine-tuning. In *Proceedings of COLING 2022*.



