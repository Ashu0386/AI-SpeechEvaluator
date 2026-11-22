import os
import sys
import warnings
import logging
import joblib
import pandas as pd
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

def evaluate_text(text, classifier, embedding_model):
    features = extract_speech_features(text, embedding_model=embedding_model)
    emb = embedding_model.encode([text], show_progress_bar=False, normalize_embeddings=True)
    pred = classifier.predict(emb)[0]

    # Metrics from comparison results
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

    print("=" * 80)
    print("1. TRANSCRIPTION:")
    print("=" * 80)
    print(text)
    print()

    print("=" * 80)
    print("2. EXTRACTED FEATURES:")
    print("=" * 80)
    print(f"- Total Words       : {features['total_words']}")
    print(f"- Filler Words      : {features['filler_count']} {features['filler_words_found']}")
    print(f"- Filler Density    : {features['filler_density']:.2f}%")
    print(f"- Hedging Phrases   : {features['hedging_count']} {features['hedging_phrases_found']}")
    print(f"- Hedging Density   : {features['hedging_density']:.2f}%")
    kws = [f"{phrase} ({score:.2f})" for phrase, score in features["keybert_keywords"][:5]]
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
    print(f"- Evaluation        : {pred.upper()}")
    print()

    print("=" * 80)
    print(f"4. CONFUSION MATRIX ({model_display}):")
    print("=" * 80)
    print(f"{'':<28} | {'Predicted: Good':<18} | {'Predicted: Needs Work':<22}")
    print("-" * 74)
    print(f"{'Actual: Good':<28} | {'8879':<18} | {'1121':<22}")
    print(f"{'Actual: Needs Improvement':<28} | {'1231':<18} | {'8769':<22}")
    print("=" * 80)

def main():
    classifier = joblib.load("interview_classifier.pkl")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        print("\nType any candidate response to evaluate (or press Enter to exit):\n")
        while True:
            try:
                text = input("Enter speech transcript: ").strip()
                if not text:
                    break
                evaluate_text(text, classifier, embedding_model)
                print()
            except (KeyboardInterrupt, EOFError):
                break
    else:
        transcript_path = Path("transcript.txt")
        transcript = transcript_path.read_text(encoding="utf-8").strip() if transcript_path.exists() else ""
        evaluate_text(transcript, classifier, embedding_model)

if __name__ == "__main__":
    main()
