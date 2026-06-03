<div align="center">

<h1>📧 Email Spam Detection System</h1>
<h3>Text Classification Using Supervised Machine Learning & NLP</h3>

<p>
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-ML%20Pipeline-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/NLP-TF--IDF%20%7C%20Naive%20Bayes-8A2BE2?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

<p>
  <img src="https://img.shields.io/badge/Model-Multinomial%20Naive%20Bayes-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Vectorizer-TF--IDF-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/App-Streamlit%20UI-red?style=flat-square"/>
  <img src="https://img.shields.io/badge/Persistence-Joblib%20Pipeline-purple?style=flat-square"/>
</p>

> A complete **end-to-end NLP classification pipeline** — from raw labeled email data through TF-IDF vectorization and Multinomial Naive Bayes training to a live Streamlit web app that classifies any message as **Spam** or **Not Spam** with a confidence probability.

</div>

---

## 📑 Table of Contents

- [Problem Statement](#-problem-statement)
- [Solution](#-solution)
- [ML Pipeline Architecture](#-ml-pipeline-architecture)
- [How It Works](#-how-it-works)
- [Dataset Details](#-dataset-details)
- [Model Details](#-model-details)
- [Text Preprocessing](#-text-preprocessing)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [App Usage](#-app-usage)
- [Engineering Highlights](#-engineering-highlights)
- [Future Roadmap](#-future-roadmap)
- [Author](#-author)
- [License](#-license)

---

## 🎯 Problem Statement

Email spam remains one of the most pervasive cybersecurity and productivity threats globally. Traditional rule-based filters are brittle — easily bypassed by minor rewording. What's needed is a system that **learns the statistical language of spam** and generalizes to unseen messages.

---

## 💡 Solution

This project implements a **supervised binary text classifier** trained on labeled real-world email datasets. The model learns the vocabulary patterns that distinguish spam from legitimate (ham) messages, then deploys via a real-time Streamlit interface — no API keys, no cloud services, entirely self-contained.

---

## 🏗️ ML Pipeline Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Raw Email Text                       │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                  Text Preprocessing                      │
│  • Lowercasing          • URL removal                   │
│  • HTML tag stripping   • Punctuation removal           │
│  • Alphanumeric filter  • Newline normalization         │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│            TF-IDF Vectorization                          │
│  TfidfVectorizer(lowercase=True, stop_words="english")  │
│  Converts text → sparse numerical feature matrix        │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│         Multinomial Naive Bayes Classifier               │
│  Learns P(spam | word frequencies) via Bayes' theorem   │
│  Outputs: label prediction + probability score          │
└──────────────────────────┬──────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
         🚨 SPAM                  ✅ NOT SPAM
      (+ confidence %)          (+ confidence %)
```

The entire pipeline — vectorizer + classifier — is serialized into a single `.joblib` file and loaded once by the Streamlit app via `@st.cache_resource`, enabling fast, stateless inference.

---

## 🔄 How It Works

### Training Phase (`train_model.py`)

```
1. Load emails.csv  (Category, Message columns)
2. Normalize labels → lowercase "spam" / "ham"
3. Balance dataset  → upsample minority class (spam) to match ham count
4. Train/test split → 80% train, 20% test (random_state=42)
5. Build sklearn Pipeline:
      TfidfVectorizer → MultinomialNB
6. Fit pipeline on training data
7. Evaluate on test set → print accuracy
8. Serialize pipeline → spam_model.pkl (joblib)
```

### Inference Phase (`spam_app.py`)

```
1. Load serialized pipeline (cached by Streamlit)
2. User pastes message into text area
3. clean_text() preprocessing applied
4. pipeline.predict()     → "spam" or "ham"
5. pipeline.predict_proba() → confidence score
6. Display result with visual label and probability
```

---

## 📊 Dataset Details

Two labeled datasets power this classifier:

| File | Format | Columns | Description |
|------|--------|---------|-------------|
| `emails.csv` | CSV | `Category`, `Message` | Primary SMS/email dataset with ham/spam labels |
| `spam.csv` | CSV | `label`, `message` | Supplementary labeled spam corpus |

**Class Balancing:** The training script applies **upsampling with replacement** (`sklearn.utils.resample`) to equalize spam and ham class sizes before training — preventing the model from defaulting to the majority class.

**Enron Dataset Support:** The included `emails.csv.py` script walks a local `enron_maildir/` folder structure and generates a `emails.csv` with binary labels — making it straightforward to train on the real-world Enron spam corpus.

---

## 🧠 Model Details

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Vectorizer** | `TfidfVectorizer` | Weights terms by document frequency — suppresses common noise words, elevates discriminative tokens |
| **Stop Words** | `english` (built-in) | Removes non-informative function words before vectorization |
| **Classifier** | `MultinomialNB` | Proven fast and accurate for discrete text feature counts; ideal for TF-IDF bag-of-words |
| **Pipeline** | `sklearn.Pipeline` | Chains vectorizer + classifier into one atomic object — ensures consistent preprocessing at inference time |
| **Persistence** | `joblib` | Efficient binary serialization of large NumPy arrays inside the fitted pipeline |

---

## 🧹 Text Preprocessing

The `clean_text()` function (in `spam_app.py`) applies six sequential transformations before inference:

```python
def clean_text(text):
    text = str(text).lower()                          # 1. Lowercase
    text = re.sub(r'\[.*?\]', '', text)               # 2. Remove bracket content
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # 3. Remove URLs
    text = re.sub(r'<.*?>+', '', text)                 # 4. Strip HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # 5. Remove punctuation
    text = re.sub(r'\n', ' ', text)                   # 6. Normalize newlines
    text = re.sub(r'\w*\d\w*', '', text)              # 7. Remove alphanumeric tokens
    return text
```

This ensures the text reaching the model at inference time matches the statistical distribution seen during training — a critical requirement for consistent prediction quality.

---

## 🛠️ Tech Stack

| Library | Version | Role |
|---------|---------|------|
| **Python** | 3.8+ | Core language |
| **Scikit-learn** | Latest | TF-IDF vectorizer, Naive Bayes, Pipeline, train_test_split, resample |
| **Streamlit** | Latest | Real-time interactive web UI |
| **Pandas** | Latest | CSV ingestion and label processing |
| **NumPy** | Latest | Numerical array operations |
| **Joblib** | Latest | Model serialization and deserialization |
| **re / string** | stdlib | Regex-based text preprocessing |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Mvkarthikeya07/email-spam-detection.git
cd email-spam-detection/spam_classifier

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install all dependencies
pip install -r requirements.txt
```

### Step 1 — Train the Model

```bash
python train_model.py
```

**Expected output:**
```
Model Accuracy: 97.XX%
Model saved as spam_model.pkl
```

> Rename `spam_model.pkl` to `spam_pipeline.joblib` OR update the filename in `spam_app.py` to match.

### Step 2 — Launch the Web App

```bash
streamlit run spam_app.py
```

The app opens automatically at `http://localhost:8501`.

---

## 📁 Project Structure

```
spam_classifier/
│
├── emails.csv               # Primary labeled dataset (Category, Message)
├── spam.csv                 # Supplementary spam corpus (label, message)
├── emails.csv.py            # Enron maildir → emails.csv converter script
│
├── train_model.py           # Full training pipeline: load → preprocess →
│                            #   balance → split → vectorize → train →
│                            #   evaluate → serialize
│
├── spam_app.py              # Streamlit inference app: load pipeline →
│                            #   clean input → predict → display result
│
├── spam_pipeline.joblib     # Serialized TF-IDF + NaiveBayes pipeline
│                            # (generated after running train_model.py)
│
├── requirements.txt         # Python dependencies
├── LICENSE
└── README.md
```

---

## 🖥️ App Usage

1. Navigate to `http://localhost:8501`
2. Paste any email or SMS body into the text area
3. Click **Check**
4. The system returns one of two outcomes:

```
🚨 SPAM  (probability 0.97)
```
```
✅ NOT SPAM  (probability 0.04)
```

The **probability score** reflects the model's confidence in the spam class — providing interpretable, calibrated output beyond a binary label.

---

## 🔬 Engineering Highlights

| Highlight | Detail |
|-----------|--------|
| **Unified Pipeline Object** | Vectorizer and classifier chained into one `sklearn.Pipeline` — prevents data leakage and guarantees consistent preprocessing between train and inference |
| **Class Imbalance Handling** | Minority class upsampled before training — prevents accuracy-paradox where model predicts all-ham |
| **Streamlit Caching** | `@st.cache_resource` loads the model once per session — zero redundant deserialization |
| **Reproducibility** | `random_state=42` on both `resample` and `train_test_split` — fully deterministic training runs |
| **Inference Preprocessing** | `clean_text()` mirrors the training-time normalization — ensures distributional consistency |
| **Modular Design** | Training (`train_model.py`) and serving (`spam_app.py`) are fully decoupled |

---

## 🔮 Future Roadmap

- [ ] **Deep Learning Models** — LSTM and Transformer-based classifiers (BERT, DistilBERT)
- [ ] **Confidence Threshold Tuning** — User-configurable spam sensitivity slider
- [ ] **Multi-class Categorization** — Phishing, promotional, social, updates beyond binary
- [ ] **REST API** — FastAPI/Flask endpoint for integration with email clients
- [ ] **Batch Classification** — Upload a CSV of emails and download annotated results
- [ ] **Explainability** — Highlight which words contributed most to the spam prediction (SHAP/LIME)
- [ ] **Live Feedback Loop** — User corrections fed back into incremental model retraining
- [ ] **Docker Deployment** — Containerized one-command deployment

---

## 👤 Author

**M V Karthikeya**
Computer Science Engineer — Machine Learning & NLP

[![GitHub](https://img.shields.io/badge/GitHub-Mvkarthikeya07-181717?style=flat-square&logo=github)](https://github.com/Mvkarthikeya07)

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for full details.

---

<div align="center">

**Building intelligent systems that understand human language.**

*Text Classification · NLP · Supervised Learning · Real-world AI*

</div>
