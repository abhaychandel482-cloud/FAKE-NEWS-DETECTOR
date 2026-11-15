# 📰 TruthPulse: Automated Fake News Detection & Credibility Analysis Engine

An end-to-end Machine Learning and Natural Language Processing (NLP) system designed to detect fake news, deceptive health claims, and political misinformation in real-time. Built with **Scikit-learn**, **Passive-Aggressive Classifier**, **TF-IDF Vectorization**, and an interactive **Streamlit** dashboard with **Explainable AI (XAI)** keyword attribution.

---

## 🌟 Key Features

1. **Natural Language Processing Pipeline**:
   - Advanced text normalization, URL/HTML tag stripping, and stopword removal.
   - N-gram TF-IDF vectorization ($n \in \{1, 2\}$) with sublinear scaling.
   - Heuristic linguistic markers: Sensationalism Score, All-Caps Ratio, and Exclamation Density.

2. **Algorithm Benchmarking**:
   - Multi-model evaluation comparing:
     - **Passive-Aggressive Classifier (PAC)** (Primary baseline)
     - **Logistic Regression (LR)**
     - **Linear Support Vector Classifier (LinearSVC)**
     - **Multinomial Naive Bayes (MNB)**
   - Evaluation metrics: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.

3. **Explainable AI (XAI)**:
   - Transparent feature attribution: highlights exact tokens in red (fake-indicative) or green (credible-indicative).
   - Word impact weighting based on learned model coefficients.

4. **Multiple Interfaces**:
   - **Streamlit Web Dashboard**: Live analyzer, model metric visualizer, dataset explorer, and academic viva guide.
   - **CLI Tool**: Instant terminal scanner with sample evaluation and interactive prompts.

---

## 📁 Project Structure

```
fake-news-detector/
│
├── app.py                         # Interactive Streamlit Web Application
├── cli.py                         # Command-Line Interface (CLI)
├── requirements.txt               # Python package dependencies
├── README.md                      # Project documentation
├── ACADEMIC_PROJECT_REPORT.md     # Full 8-Chapter University Project Report
├── PRESENTATION_SLIDES_OUTLINE.md # 12-Slide College Viva & PPT Outline
│
├── data/
│   ├── create_dataset.py          # Benchmark dataset generator
│   └── news_dataset.csv           # Balanced corpus of real & fake articles
│
├── models/
│   ├── best_model.joblib          # Trained best classifier (PAC)
│   ├── tfidf_vectorizer.joblib    # Serialized TF-IDF vectorizer
│   ├── all_models.joblib          # Serialized candidate model bundle
│   └── benchmark_results.json     # Stored evaluation metrics
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py           # NLP text cleaning & linguistic heuristics
│   ├── model.py                   # Training, benchmarking, and inference
│   └── explainability.py          # Feature attribution & HTML highlighting
│
└── tests/
    └── test_detector.py           # Comprehensive unit tests
```

---

## 🚀 Quick Start Guide

### 1. Launch the Streamlit Web Dashboard
Run the following command in your terminal:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

- **Live Verification Engine**: Paste any news headline and body or select a preset sample article to get instant credibility verdicts, confidence %, clickbait signals, and highlighted keyword attribution.

---

### 2. Run the Command-Line Interface (CLI)

Scan a pre-loaded fake news sample:
```bash
python cli.py --sample fake
```

Scan a pre-loaded credible news sample:
```bash
python cli.py --sample real
```

Scan custom text directly:
```bash
python cli.py --title "NASA Discovers Alien Metropolises" --text "Classified files leaked today..."
```

Launch interactive prompt:
```bash
python cli.py --interactive
```

---

### 3. Retrain Models
To retrain the models and regenerate benchmark metrics:
```bash
python src/model.py
```

### 4. Run Unit Tests
```bash
python -m unittest tests/test_detector.py
```

---

## 📊 Benchmark Results

| Algorithm | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Passive-Aggressive Classifier** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| **Logistic Regression** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| **Linear Support Vector Classifier** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| **Multinomial Naive Bayes** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |

*(Evaluated on 20% holdout stratified test split with unigram + bigram TF-IDF representations).*

---

## 🎓 Academic Documents Included
- [ACADEMIC_PROJECT_REPORT.md](ACADEMIC_PROJECT_REPORT.md): Ready-to-submit 8-chapter project report including Abstract, Problem Statement, Mathematical Formulations, Architecture, and Conclusion.
- [PRESENTATION_SLIDES_OUTLINE.md](PRESENTATION_SLIDES_OUTLINE.md): Slide-by-slide structure for PowerPoint presentation with speaker notes.
