# Sentiment and Emotion Analysis using RoBERTa and EDA Techniques on Twitter Data


This repository presents a complete pipeline for analyzing public sentiment and emotions from Twitter data. We apply both traditional machine learning models (e.g., Logistic Regression, SVM) and transformer-based NLP models (e.g., RoBERTa) to classify tweets based on sentiment (Negative, Neutral, Positive) and emotion (Joy, Sadness, Anger, Optimism).
## 📂 Dataset Source

This project uses the publicly available Twitter dataset related to the Ukraine–Russia crisis:

🔗 [Ukraine-Russian Crisis Twitter Dataset (1.2M Tweets) — Kaggle](https://www.kaggle.com/datasets/bwandowando/ukraine-russian-crisis-twitter-dataset-1-2-m-rows/data)
---

## 📌 Project Objectives

- Perform **exploratory data analysis (EDA)** on a large Twitter dataset.
- Engineer custom features like engagement scores and time-based tweet behavior.
- Implement **sentiment classification** using both:
  - Rule-based models (TextBlob)
  - Machine learning (Logistic Regression, SVM, Random Forest)
  - Transformer model: [`cardiffnlp/twitter-roberta-base-sentiment`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)
- Perform **emotion classification** (anger, joy, optimism, sadness) using RoBERTa and traditional classifiers.
- Evaluate models using precision, recall, F1-score, accuracy, and confusion matrix.

---

## 📂 Dataset Description

- **Source**: Sampled from Twitter API stream
- **Size**: ~700,000 tweets with 28 attributes
- **Features**:
  - **User Info**: `username`, `acctdesc`, `location`, `followers`, `following`, `totaltweets`
  - **Tweet Info**: `text`, `hashtags`, `created_at`, `language`
  - **Engagement**: `retweetcount`, `favorite_count`
  - **NLP Targets**: Sentiment & Emotion labels using RoBERTa
- All tweets were filtered to include only English-language content.

---

## ⚙️ Methodology

### 🔧 Preprocessing

- Lowercasing, URL and mention removal, emoji handling
- Stopword removal using NLTK
- Lemmatization using spaCy
- Custom features: text length, hourly activity, engagement score

### 📊 Exploratory Data Analysis (EDA)

- Word clouds for frequent terms and sentiment/emotion categories
- Time-series plots (hourly, weekly, monthly)
- Text length distributions vs. engagement levels

### 🧠 Models Used

| Task            | Traditional Models                           | Deep Learning |
|-----------------|-----------------------------------------------|---------------|
| Sentiment       | TextBlob, Logistic Regression, SVM, RF, DT   | RoBERTa       |
| Emotion         | Naive Bayes, KNN, Logistic Regression, SVM   | RoBERTa       |

---

## 📈 Model Evaluation

### 🔹 Sentiment Classification (Linear SVC)

- **Accuracy**: 86.7%
- **Macro F1-score**: 0.86
- **Weighted F1-score**: 0.87

### 🔹 Emotion Classification (Linear SVC)

- **Accuracy**: 85.6%
- **Macro F1-score**: 0.82
- **Best per-class F1**:
  - Anger: 0.91
  - Joy: 0.83
  - Optimism: 0.78
  - Sadness: 0.76

---

## 🏗️ File Structure

```bash
.
├── data/                         # Input CSVs and RoBERTa outputs
├── models/                       # Trained ML models (optional)
├── figures/                      # All generated plots and wordclouds
├── notebooks/                    # EDA and modeling notebooks
├── requirements.txt              # Python dependencies
├── sentiment_classification.py   # Main script for sentiment modeling
├── emotion_classification.py     # Emotion detection logic
└── README.md
