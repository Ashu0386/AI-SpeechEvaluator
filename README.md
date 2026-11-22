# AI Speech Evaluator

> Automated, end-to-end intelligent assessment system for spoken technical interview responses using Whisper, KeyBERT, and Ensembled Machine Learning architectures.

---

## 📌 Overview

In technical interviews (software engineering, system architecture, engineering leadership), candidate evaluation hinges on two dimensions: **strong technical depth** and **fluent, confident articulation**.

The **AI Speech Evaluator** implements a hybrid multimodal assessment pipeline:
1. **Speech-to-Text Transcription**: Converts spoken audio into text using OpenAI's Whisper model.
2. **Linguistic Fluency Analysis**: Quantifies filler words (`um`, `uh`, `like`, `basically`), hedging phrases (`maybe`, `I think`, `I guess`), and computes density percentages.
3. **KeyBERT Dynamic Semantic Keyphrase Extraction**: Leverages BERT embeddings (`all-MiniLM-L6-v2`) and cosine similarity ranking to dynamically score technical relevance and semantic depth without hardcoded dictionaries.
4. **Machine Learning Classification**: Benchmarks and serves multiple ML architectures trained across **100,000 balanced interview samples**.
5. **Structured Terminal Diagnostic Output**: Delivers clean, actionable feedback with transcription, key metrics, confidence scores, and confusion matrix evaluations.

---

## 📊 Model Benchmark (100,000 Samples)

Evaluated across an 80/20 train-test split (80,000 train / 20,000 test):

| Model | Architecture | Test Accuracy | Precision | Recall |
| :--- | :--- | :---: | :---: | :---: |
| **Meta Model (Stacking)** | Meta-Learner Stacking | **88.24%** | **0.8782** | **0.8879** |
| **Ensemble (Voting)** | Soft Voting Ensemble | **87.45%** | **0.8698** | **0.8808** |
| **XGBoost** | Gradient Boosted Trees | **86.78%** | **0.8631** | **0.8742** |
| **Random Forest** | Bagging Trees (40 estimators) | **85.92%** | **0.8548** | **0.8654** |
| **Linear SVM** | Max-Margin Hyperplane | **85.31%** | **0.8491** | **0.8589** |
| **Logistic Regression** | Linear Probabilistic Baseline | **84.65%** | **0.8423** | **0.8526** |

---

## 📁 Repository Structure

```
AI_Speech_Evaluator/
├── audio/
│   └── sample.mp3                     # Sample audio recording for transcription
├── build_study_guide_pdf.py           # Generator for PDF study guide
├── AI_Speech_Evaluator_Study_Guide.pdf # Exported system study guide
├── compare_models.py                  # Benchmark script for all 6 ML classifiers
├── create_dataset.py                  # Synthetic data generator for 100k samples
├── create_embeddings.py               # KeyBERT & BERT embeddings generation
├── demo.py                            # Interactive live demo evaluation tool
├── extract_features.py                # Feature extraction & KeyBERT NLP pipeline
├── interview_classifier.pkl           # Serialized trained model checkpoint
├── model_comparison_results.csv       # Benchmark results across evaluated models
├── models.py                          # Model definitions & training abstractions
├── predict.py                         # Single-file inference & formatted output
├── PROJECT_DOCUMENTATION.md           # In-depth system design & theoretical background
├── test_system.py                     # System integration & component tests
├── train_model.py                     # Training script with model selection CLI
├── transcribe.py                      # Whisper audio transcription engine
├── transcript.txt                     # Cached transcription text
└── .gitignore                         # Git exclusion rules
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites & Virtual Environment

```bash
# Clone the repository
git clone <YOUR_REPOSITORY_URL>
cd AI_Speech_Evaluator

# Create and activate virtual environment
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# Linux / macOS
source venv/bin/activate
```

### 2. Run Pipeline Steps

```powershell
# 1. Transcribe speech audio with Whisper
python transcribe.py

# 2. Extract features using KeyBERT
python extract_features.py

# 3. Generate dataset (100,000 samples)
python create_dataset.py

# 4. Compare all benchmarked models
python compare_models.py

# 5. Train a model (Options: meta_model, ensemble, xgboost, random_forest, svm, logistic_regression)
python train_model.py --model meta_model

# 6. Run prediction with structured terminal output
python predict.py

# 7. Launch interactive demo
python demo.py --interactive
```

---

## 📄 License
MIT License.
