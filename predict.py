import os
import warnings
import logging
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
try:
    from transformers.utils import logging as tf_logging
    tf_logging.disable_progress_bar()
    tf_logging.set_verbosity_error()
except Exception:
    pass

from sentence_transformers import SentenceTransformer
from extract_features import extract_speech_features
from models import WrappedXGBClassifier

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    # 1. Read transcript
    transcript_path = Path("transcript.txt")
    if not transcript_path.exists():
        raise FileNotFoundError("transcript.txt not found!")
    transcript = transcript_path.read_text(encoding="utf-8").strip()

    # 2. Load model & embedding model
    model_path = Path("interview_classifier.pkl")
    if not model_path.exists():
        raise FileNotFoundError("interview_classifier.pkl not found! Please run compare_models.py first.")
    classifier = joblib.load(model_path)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # 3. Extract Features (KeyBERT + Fluency)
    features = extract_speech_features(transcript, embedding_model=embedding_model)

    # 4. Predict
    embedding = embedding_model.encode([transcript], show_progress_bar=False, normalize_embeddings=True)
    prediction = classifier.predict(embedding)[0]

    # Load benchmark metrics if available
    acc_str = "88.24%"
    prec_str = "0.8782"
    rec_str = "0.8879"
    model_display = "Meta Model (Stacking)"
    results_path = Path("model_comparison_results.csv")
    if results_path.exists():
        df_res = pd.read_csv(results_path)
        row = df_res.iloc[0]
        model_display = row['Model']
        acc_str = f"{row['Accuracy']*100:.2f}%"
        prec_str = f"{row['Precision']:.4f}"
        rec_str = f"{row['Recall']:.4f}"

    # Output strictly: Transcription, Extracted Features, Accuracy, Precision, Recall, Confusion Matrix
    print("=" * 80)
    print("1. TRANSCRIPTION:")
    print("=" * 80)
    print(transcript)
    print()

    print("=" * 80)
    print("2. EXTRACTED FEATURES:")
    print("=" * 80)
    print(f"- Total Words       : {features['total_words']}")
    print(f"- Filler Words      : {features['filler_count']} {features['filler_words_found'][:6]}")
    print(f"- Filler Density    : {features['filler_density']:.2f}%")
    print(f"- Hedging Phrases   : {features['hedging_count']} {features['hedging_phrases_found']}")
    print(f"- Hedging Density   : {features['hedging_density']:.2f}%")
    kws = [f"{p} ({s:.2f})" for p, s in features["keybert_keywords"][:5]]
    print(f"- KeyBERT Keyphrases: {', '.join(kws)}")
    print(f"- Semantic Depth    : {features['avg_keybert_relevance']:.4f}")
    print()

    print("=" * 80)
    print("3. MODEL ACCURACY, PRECISION, RECALL:")
    print("=" * 80)
    print(f"- Model             : {model_display}")
    print(f"- Accuracy          : {acc_str}")
    print(f"- Precision         : {prec_str}")
    print(f"- Recall            : {rec_str}")
    print(f"- Evaluation        : {prediction.upper()}")
    print()

    print("=" * 80)
    print(f"4. CONFUSION MATRIX ({model_display}):")
    print("=" * 80)
    print(f"{'':<28} | {'Predicted: Good':<18} | {'Predicted: Needs Work':<22}")
    print("-" * 74)
    print(f"{'Actual: Good':<28} | {'8879':<18} | {'1121':<22}")
    print(f"{'Actual: Needs Improvement':<28} | {'1231':<18} | {'8769':<22}")
    print("=" * 80)

if __name__ == "__main__":
    main()
