# COS760 NLP Project – Group 16

## Semantic Relatedness in African Languages
Evaluating Cross-Lingual Transfer and Data Augmentation for Afrikaans, Hausa, and Kinyarwanda

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

## 📁 Project Structure

COS760Project/
│
├── data/ # Dataset files
├── src/ # Core code
│ ├── data_loader.py
│ ├── models.py
│ ├── train.py
│ ├── evaluate.py
│ └── augment.py
│
├── notebooks/ # Experiments & exploration
├── results/ # Outputs, metrics, plots
└── README.md

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

## 📚 References
- SemRel Dataset (SemEval 2024)
- Sentence-BERT
- XLM-R
- AfriBERTa





