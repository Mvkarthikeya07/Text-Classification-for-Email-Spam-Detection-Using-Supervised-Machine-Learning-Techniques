# 📧 Email Spam Detection — NLP Text Classification System

> **Binary text classification pipeline** using TF-IDF vectorization and Multinomial Naïve Bayes, trained on labeled email/SMS corpora to distinguish spam from legitimate messages with high accuracy.

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Dataset](#dataset)
- [Model Design](#model-design)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Technical Deep-Dive](#technical-deep-dive)
- [Results](#results)
- [Future Roadmap](#future-roadmap)
- [Author](#author)

---

## Overview

This project implements a complete, production-ready **supervised machine learning pipeline** for detecting spam in email and SMS messages. It covers every stage of the ML lifecycle — raw text ingestion, class-balanced preprocessing, TF-IDF feature extraction, Naïve Bayes classification, model serialization, and a live Streamlit prediction interface.

The system is deliberately lean: no heavy neural networks, no cloud dependencies. It demonstrates that classical NLP techniques, applied rigorously, can achieve strong real-world performance with minimal computational cost.

**What makes this different from a tutorial project:**

- Class imbalance handled via upsampling before training
- Custom text normalization removing URLs, HTML, punctuation, and numeric tokens
- Full `sklearn.Pipeline` object serialized with `joblib` — preprocessing and model travel together
- Streamlit UI with per-prediction spam probability score (not just a label)
- Enron maildir ingestion script for scaling to large corpora

---

## Pipeline Architecture

```
Raw Email / SMS Text
        │
        ▼
┌──────────────────────┐
│   Text Normalization  │  lowercase · strip URLs · remove HTML tags
│   (clean_text)        │  strip punctuation · remove alphanumeric tokens
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   TF-IDF Vectorizer   │  lowercase=True · stop_words="english"
│                       │  converts text → sparse numerical matrix
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Multinomial Naïve    │  trained on class-balanced corpus
│  Bayes Classifier     │  outputs label + posterior probability
└──────────┬───────────┘
           │
           ▼
    Spam / Not Spam
    + Confidence Score
```

The preprocessing and classifier are wrapped in a single `sklearn.Pipeline` object, ensuring that the exact same transformations applied during training are automatically applied at inference time — no train/serve skew.

---

## Dataset

| File | Description | Format |
|---|---|---|
| `spam.csv` | Curated labeled samples (ham/spam labels, message text) | `label, message` |
| `emails.csv` | Extended email corpus with `Category` and `Message` columns | `Category, Message` |

**Class balancing:** The training script detects the ham/spam ratio and upsamples the minority (spam) class using `sklearn.utils.resample` with replacement, producing a balanced 50/50 split before the train-test divide. This prevents the classifier from developing a naïve "always predict ham" bias.

To build a larger dataset from the **Enron maildir** corpus, use the provided ingestion script:

```bash
python emails.csv.py   # set root_folder = "enron_maildir" inside the script
```

---

## Model Design

### Why Multinomial Naïve Bayes?

| Property | Detail |
|---|---|
| **Input** | Non-negative integer / float feature counts (TF-IDF scores) |
| **Assumption** | Feature independence given class (well-suited to bag-of-words) |
| **Training cost** | O(n · d) — extremely fast even on large vocabularies |
| **Inference cost** | O(d) per sample — real-time capable |
| **Spam domain fit** | Strong empirical track record; interpretable word-level probabilities |

### Why TF-IDF over raw counts?

Raw term frequencies give disproportionate weight to common words even after stopword removal. TF-IDF downweights terms that appear in many documents (low discriminative power) and upweights rare, message-specific tokens (high spam signal). For spam detection — where signal words like "FREE", "WINNER", "CLAIM" appear in few documents but are highly class-indicative — this is a meaningful improvement over raw counts or binary bag-of-words.

---

## Project Structure

```
email_spam_detection/
│
├── spam_classifier/
│   ├── spam.csv                 # Primary labeled dataset (ham / spam)
│   ├── emails.csv               # Extended corpus (Category / Message)
│   ├── emails.csv.py            # Enron maildir → emails.csv ingestion script
│   ├── train_model.py           # Full training pipeline (balance → TF-IDF → NB → save)
│   ├── spam_pipeline.joblib     # Serialized sklearn Pipeline (vectorizer + classifier)
│   ├── spam_app.py              # Streamlit prediction UI
│   └── requirements.txt        # Python dependencies
│
├── LICENSE                      # MIT License
└── README.md
```

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/Mvkarthikeya07/Text-Classification-for-Email-Spam-Detection-Using-Supervised-Machine-Learning-Techniques.git
cd email_spam_detection/spam_classifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`**
```
streamlit
scikit-learn
pandas
numpy
joblib
```

### 3. (Optional) Retrain the model

The repo ships with a pre-trained `spam_pipeline.joblib`. To retrain from scratch:

```bash
python train_model.py
```

This will print the test-set accuracy and overwrite `spam_model.pkl`.

### 4. Launch the prediction UI

```bash
streamlit run spam_app.py
```

Navigate to `http://localhost:8501`, paste any email or SMS body, and click **Check**.

---

## Usage

### Streamlit Web UI

Paste any message into the text area and hit **Check**:

- ✅ **NOT SPAM** — displayed with the model's ham probability
- 🚨 **SPAM** — displayed with the spam confidence score (0.00 – 1.00)

The confidence score is derived from `predict_proba`, giving you a calibrated sense of certainty rather than a hard binary label.

### Programmatic inference

```python
import joblib, re, string

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

pipeline = joblib.load("spam_pipeline.joblib")

message = "Congratulations! You have won a £1000 cash prize. Call now to claim."
cleaned  = clean_text(message)
label    = pipeline.predict([cleaned])[0]          # 'spam' or 'ham'
proba    = pipeline.predict_proba([cleaned])[0][1] # spam probability

print(f"Label: {label}  |  Spam probability: {proba:.4f}")
```

---

## Technical Deep-Dive

### Text Normalization

The `clean_text` function applies six sequential transformations before vectorization:

| Step | Regex / Operation | Rationale |
|---|---|---|
| Lowercasing | `str.lower()` | Merge case variants of the same token |
| Strip bracket content | `\[.*?\]` | Remove encoding artifacts and metadata |
| Remove URLs | `https?://\S+\|www\.\S+` | URLs add noise; domain-level features require separate extraction |
| Strip HTML tags | `<.*?>+` | Clean forwarded / rendered email content |
| Remove punctuation | `string.punctuation` | Reduce vocabulary; punctuation rarely carries class signal |
| Remove alphanumeric tokens | `\w*\d\w*` | Phone numbers, codes, prices — too variable to generalize |

### Training Procedure

```python
# Abbreviated from train_model.py

# Class balance via upsampling
spam_upsampled = resample(spam_df, replace=True,
                          n_samples=len(ham_df), random_state=42)
df_balanced = pd.concat([ham_df, spam_upsampled])

# 80/20 stratified split
X_train, X_test, y_train, y_test = train_test_split(
    df_balanced["text"], df_balanced["label"],
    test_size=0.2, random_state=42
)

# Pipeline: TF-IDF → Multinomial Naïve Bayes
pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer(lowercase=True, stop_words="english")),
    ("classifier", MultinomialNB())
])

pipeline.fit(X_train, y_train)
print(f"Accuracy: {pipeline.score(X_test, y_test) * 100:.2f}%")
joblib.dump(pipeline, "spam_model.pkl")
```

---

## Results

| Metric | Value |
|---|---|
| **Algorithm** | Multinomial Naïve Bayes |
| **Feature extraction** | TF-IDF (English stopwords removed) |
| **Train / Test split** | 80% / 20% |
| **Class balancing** | Upsample minority class to 1:1 ratio |
| **Output** | Binary label + calibrated probability score |

> Exact accuracy figures vary by dataset version. Run `train_model.py` against your corpus to generate current metrics.

---

## Future Roadmap

| Enhancement | Notes |
|---|---|
| Deep learning classifier | Fine-tuned `distilbert-base-uncased` or LSTM for contextual embeddings |
| Multi-class categorization | Phishing · Promotional · Social · Updates · Primary |
| Confidence threshold tuning | Adjustable decision boundary for precision/recall trade-off |
| REST API | FastAPI wrapper around the joblib pipeline for integration |
| Evaluation dashboard | Precision, recall, F1, confusion matrix, ROC-AUC via Streamlit |
| Active learning loop | Flag low-confidence predictions for human review and retraining |

---

## Author

**M V Karthikeya**
Computer Science Engineer · Machine Learning & NLP

[![GitHub](https://img.shields.io/badge/GitHub-Mvkarthikeya07-181717?style=flat-square&logo=github)](https://github.com/Mvkarthikeya07)

---

## License

Distributed under the [MIT License](LICENSE).

---

<div align="center">
  <sub>Built with scikit-learn · Streamlit · Python</sub>
</div>
