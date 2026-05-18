# deepcsat-nlp-customer-satisfaction
E-commerce customer satisfaction classification on 85K support records. TF-IDF + Logistic Regression, 71.75% accuracy. Token-level SHAP identifies complaint phrases. Honest macro-F1 0.25 reported due to class imbalance. Python · NLP · SHAP
# 💬 DeepCSAT — E-Commerce Customer Satisfaction Prediction

> NLP pipeline classifying 85K customer support records by sentiment. TF-IDF + Logistic Regression. Token-level SHAP pinpoints complaint phrases. Honest macro-F1 0.25 reported due to class imbalance.

---

## 📌 Problem Statement
E-commerce platforms handle thousands of customer support tickets daily. Automatically classifying whether a customer is satisfied, neutral, or dissatisfied allows support teams to prioritise urgent cases and identify recurring complaint patterns.

## 📊 Key Results

| Metric | Value |
|---|---|
| Dataset size | 85,907 support records |
| Model | TF-IDF + Logistic Regression |
| Accuracy | **71.75%** |
| Macro F1 | **0.25** (honest — class imbalance) |
| Primary metric chosen | Macro F1 (not accuracy) |
| Explainability | Token-level SHAP |

> **Why Macro F1 and not accuracy?** The dataset is heavily imbalanced — most records fall into one class. Accuracy would be misleadingly high. Macro F1 treats all classes equally and is the honest metric here.

## 🛠️ Tech Stack
`Python` · `scikit-learn` · `TF-IDF` · `Logistic Regression` · `SHAP` · `NLTK` · `Pandas` · `Matplotlib` · `Seaborn`

## 📁 Project Structure
```
DeepCSAT/
│
├── DeepCSAT_Ecommerce_CSAT_Prediction.ipynb   ← Main notebook
├── eCommerce_Customer_support_data.csv        ← Dataset (place here)
├── requirements.txt
└── README.md
```

## ▶️ How to Run
```bash
pip install -r requirements.txt
jupyter notebook DeepCSAT_Ecommerce_CSAT_Prediction.ipynb
```

## 🔍 Key Findings
- TF-IDF captures complaint vocabulary effectively at low computational cost
- Token-level SHAP identifies exact words/phrases that drive negative classification
- Class imbalance is the main challenge — future work: try SMOTE on text features or BERT fine-tuning
- Support tickets containing words like "delay", "wrong item", "refund" consistently score negative

---
*Project completed as part of AI/ML Internship at Labmentix Pvt. Ltd. (2025–2026)*
