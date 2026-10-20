# 🧠 NLP Emotion Recognition & Interactive Streamlit Dashboard

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-1.9+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![spaCy](https://img.shields.io/badge/spaCy-3.8+-09A3D5?style=for-the-badge&logo=spacy&logoColor=white)](https://spacy.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

An end-to-end Natural Language Processing (NLP) framework and interactive web dashboard for fine-grained multi-class emotion classification. The platform cleans, tokenizes, and classifies English text into six primary human emotions: **Joy, Sadness, Anger, Fear, Love, and Surprise**.

---

## 🌟 Key Features

* **🔮 Real-Time Interactive Emotion Prediction**: Predict emotion categories instantly from custom text input with confidence scores and probability breakdowns.
* **📁 Batch Processing**: Upload CSV or text files to perform emotion classification across thousands of records with downloadable results and summary charts.
* **📊 Comprehensive Exploratory Data Analysis (EDA)**: Interactive distribution plots, text length histograms, word count box plots, and split proportion analytics.
* **🧪 Model Performance Benchmarks**: Interactive confusion matrix, macro/weighted metrics (Accuracy, Precision, Recall, F1-Score), and per-class performance tables.
* **🛠️ Step-by-Step NLP Pipeline Simulator**: Test custom sentences to observe live text cleaning, tokenization, stopword removal, and negation handling.
* **🔄 On-the-Fly Model Switching & Retraining**: Train and compare multiple classifiers (`Logistic Regression + TF-IDF`, `Multinomial Naive Bayes + BoW`, `Multinomial Naive Bayes + TF-IDF`) directly from the UI.

---

## 📊 Dataset & Emotion Taxonomy

The project utilizes an extensive text emotion dataset formatted as semi-colon delimited text and labels (`text;emotion`).

### Dataset Statistics
| Dataset Split | File Path | Total Samples | Description |
| :--- | :--- | :---: | :--- |
| **Training Set** | [`data/train.txt`](file:///c:/Users/HP/OneDrive/Documents/NLP_Project/data/train.txt) | **16,000** | Primary dataset used for vectorizer fitting and classifier training |
| **Validation Set** | [`data/val.txt`](file:///c:/Users/HP/OneDrive/Documents/NLP_Project/data/val.txt) | **2,000** | Model validation and hyperparameter evaluation |
| **Testing Set** | [`data/test.txt`](file:///c:/Users/HP/OneDrive/Documents/NLP_Project/data/test.txt) | **2,000** | Independent test set for final benchmark metrics |

### Target Emotions
| Label ID | Emotion | Emoji | Color Code | Description |
| :---: | :--- | :---: | :--- | :--- |
| `0` | **Sadness** | 😢 | `#70A1FF` | Feeling of sorrow, hopelessness, or unhappiness |
| `1` | **Anger** | 😠 | `#FF4757` | Feeling of strong displeasure, annoyance, or hostility |
| `2` | **Love** | ❤️ | `#FF6B81` | Feeling of deep affection, fondness, or tenderness |
| `3` | **Surprise** | 😲 | `#1E90FF` | Feeling of astonishment, wonder, or unexpected amazement |
| `4` | **Fear** | 😨 | `#FFA502` | Feeling of anxiety, threat, or apprehension |
| `5` | **Joy** | 😊 | `#2ED573` | Feeling of happiness, pleasure, or contentment |

---

## 🏗️ NLP & Machine Learning Architecture

```
[Raw User Text / File Input]
         │
         ▼
[Preprocessing Pipeline] ──► 1. Lowercasing
                         ──► 2. Punctuation Removal
                         ──► 3. Numeric Digits Removal
                         ──► 4. Emoji / Non-ASCII Removal
                         ──► 5. SpaCy Tokenization & Stopwords Filtering
                                (Preserving Negations: 'not', 'no', 'never')
         │
         ▼
[Vectorization (TF-IDF)]  ──► Unigrams & Bigrams (15,000 vocabulary features)
         │
         ▼
[Classifier (Logistic Regression)] ──► Multi-class log-odds calculation
         │
         ▼
[Output] ──► Top Emotion Label + Confidence Score + Probability Distribution
```

### Preprocessing Highlights
* **Negation Awareness**: Standard stopword filters remove words like "not" or "no", which severely damages emotion accuracy (e.g. *"i am not happy"* vs *"i am happy"*). The pipeline explicitly retains negation tokens.
* **SpaCy Blank Pipeline**: Fast, lightweight tokenization leveraging `spacy.blank("en")`.

---

## 📈 Model Performance Benchmarks

Evaluated on the independent test dataset (`data/test.txt`):

| Model Architecture | Accuracy | Weighted F1-Score | Macro F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression + TF-IDF** | **86.69%** | **86.00%** | **81.00%** | ⭐ **Best / Default** |
| **Multinomial Naive Bayes + BoW** | 82.40% | 81.80% | 75.20% | Tested |
| **Multinomial Naive Bayes + TF-IDF** | 78.50% | 76.90% | 68.40% | Tested |

---

## 📁 Repository Structure

```
NLP_Project/
├── app.py                      # Interactive Streamlit Web UI application
├── main.py                     # CLI entrypoint for training and demo execution
├── main.ipynb                  # Exploratory Data Analysis & Model Training Notebook
├── pyproject.toml              # Project metadata & dependency declarations
├── README.md                   # Comprehensive project documentation
├── data/                       # Text dataset splits
│   ├── train.txt               # 16,000 training records
│   ├── val.txt                 # 2,000 validation records
│   └── test.txt                # 2,000 testing records
├── models/                     # Saved trained model artifacts
│   └── emotion_model.joblib    # Serialized ML pipeline & metrics
└── src/                        # Core Python NLP modules
    ├── __init__.py             # Module initializer
    ├── preprocessing.py        # Text cleaning, spaCy tokenization & mappings
    └── model.py                # Model training, evaluation & prediction APIs
```

---

## 🚀 Quick Start & Usage

### 1. Requirements & Prerequisites
Ensure Python 3.11+ is installed.

### 2. Environment Setup
Clone or navigate to the repository directory and install dependencies:

Using `uv` (recommended):
```bash
uv sync
```

Or using standard `pip`:
```bash
pip install -r pyproject.toml
```

Or installing dependencies directly:
```bash
pip install streamlit scikit-learn spacy pandas numpy matplotlib seaborn plotly joblib
```

### 3. Launching the Interactive Streamlit UI
Run the following command to start the web app:

```bash
python -m streamlit run app.py
```

*(Note: If your virtual environment is not automatically active, you can also use `uv run streamlit run app.py`)*

The Streamlit app will automatically open in your default browser at `http://localhost:8501`.

### 4. Running CLI Demo or Model Retraining
To run a quick command-line prediction demo:
```bash
python main.py
```

To retrain the machine learning model via command line:
```bash
python main.py --train
```

---

## 🖥️ Streamlit Dashboard Walkthrough

1. **🔮 Emotion Classifier Tab**: Input custom text or select preset samples to view live predictions, confidence gauges, and horizontal bar charts for all 6 emotion probabilities. Supports batch CSV uploading.
2. **📊 Dataset & EDA Analytics Tab**: Explore class balance bar charts, split ratios, text word count distributions per emotion, and search raw dataset samples.
3. **🧪 Model Benchmarks & Metrics Tab**: View confusion matrices, macro precision/recall, weighted F1 scores, and per-class breakdown charts.
4. **🛠️ NLP Pipeline & Preprocessor Tab**: Inspect step-by-step textual transformations (raw input $\rightarrow$ lowercased $\rightarrow$ cleaned tokens).

---

## 📜 License

This project is open-source and released under the [MIT License](LICENSE).
