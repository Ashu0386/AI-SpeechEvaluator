import os
import sys
import unittest
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from extract_features import extract_speech_features, get_keybert_model

class TestAISpeechEvaluator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n" + "=" * 70)
        print(" INITIALIZING COMPREHENSIVE TEST SUITE FOR AI SPEECH EVALUATOR")
        print("=" * 70)
        cls.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        cls.kw_model = get_keybert_model(cls.embedding_model)
        cls.classifier_path = Path("interview_classifier.pkl")
        assert cls.classifier_path.exists(), "interview_classifier.pkl must exist!"
        cls.classifier = joblib.load(cls.classifier_path)

    def test_01_feature_extraction_strong_technical(self):
        """Test feature extraction on a strong, articulate technical response."""
        text = (
            "We designed a high-throughput microservice architecture using Apache Kafka "
            "for distributed event streaming and Redis for sub-millisecond caching."
        )
        feat = extract_speech_features(text, kw_model=self.kw_model)
        
        # Verify basic linguistic metrics
        self.assertGreater(feat["total_words"], 15)
        self.assertEqual(feat["filler_count"], 0)
        self.assertEqual(feat["filler_density"], 0.0)
        self.assertEqual(feat["hedging_count"], 0)
        
        # Verify KeyBERT dynamic extraction
        self.assertGreater(len(feat["keybert_keywords"]), 0)
        self.assertGreater(feat["avg_keybert_relevance"], 0.20)
        
        # Keyphrases should contain relevant terms
        extracted_text = " ".join(feat["keybert_phrases"]).lower()
        self.assertTrue(
            any(w in extracted_text for w in ["kafka", "microservice", "redis", "caching", "streaming"]),
            f"Expected technical terms in KeyBERT output, got: {feat['keybert_phrases']}"
        )
        print("  [PASS] Test 1: Strong Technical KeyBERT Extraction")

    def test_02_feature_extraction_hesitant_response(self):
        """Test feature extraction on a hesitant, filler-heavy candidate response."""
        text = (
            "Um, so like, basically, I think maybe we could, uh, just restart the server "
            "whenever it crashes, but, like, I guess nobody really tested it."
        )
        feat = extract_speech_features(text, kw_model=self.kw_model)
        
        # Should detect multiple fillers and hedging phrases
        self.assertGreaterEqual(feat["filler_count"], 4)
        self.assertGreater(feat["filler_density"], 10.0)
        self.assertGreaterEqual(feat["hedging_count"], 2)
        
        # KeyBERT should capture conversational keyphrases
        self.assertGreater(len(feat["keybert_keywords"]), 0)
        print("  [PASS] Test 2: Hesitant Response & Disfluency Detection")

    def test_03_edge_cases_empty_and_short(self):
        """Test extreme edge cases: empty string, single word, short phrases."""
        # Empty string
        feat_empty = extract_speech_features("", kw_model=self.kw_model)
        self.assertEqual(feat_empty["total_words"], 0)
        self.assertEqual(feat_empty["filler_count"], 0)
        self.assertEqual(feat_empty["filler_density"], 0.0)
        self.assertEqual(len(feat_empty["keybert_keywords"]), 0)
        
        # Single word
        feat_single = extract_speech_features("Kubernetes", kw_model=self.kw_model)
        self.assertEqual(feat_single["total_words"], 1)
        self.assertEqual(feat_single["filler_count"], 0)
        
        # Two words
        feat_two = extract_speech_features("Like um", kw_model=self.kw_model)
        self.assertEqual(feat_two["total_words"], 2)
        self.assertEqual(feat_two["filler_count"], 2)
        self.assertEqual(feat_two["filler_density"], 100.0)
        print("  [PASS] Test 3: Robustness on Empty & Short Inputs")

    def test_04_model_inference_good_vs_bad(self):
        """Test that the trained classifier gives correct predictions on contrasting inputs."""
        good_text = (
            "We optimized query latency by adding composite B-Tree indexes on user_id and "
            "created_at, and introduced connection pooling to eliminate database bottlenecks."
        )
        bad_text = (
            "Um, well, like, basically, I think maybe we can just copy paste the code across "
            "the files and, uh, hope that nobody notices the bugs, you know."
        )
        
        emb_good = self.embedding_model.encode([good_text], normalize_embeddings=True)
        emb_bad = self.embedding_model.encode([bad_text], normalize_embeddings=True)
        
        pred_good = self.classifier.predict(emb_good)[0]
        pred_bad = self.classifier.predict(emb_bad)[0]
        
        self.assertEqual(pred_good, "good", f"Expected 'good' for strong technical answer, got {pred_good}")
        self.assertEqual(pred_bad, "needs_improvement", f"Expected 'needs_improvement' for hesitant answer, got {pred_bad}")
        
        # Test probability confidence
        if hasattr(self.classifier, "predict_proba"):
            probs_good = self.classifier.predict_proba(emb_good)[0]
            probs_bad = self.classifier.predict_proba(emb_bad)[0]
            idx_good = list(self.classifier.classes_).index("good")
            idx_bad = list(self.classifier.classes_).index("needs_improvement")
            self.assertGreater(probs_good[idx_good], 0.70)
            self.assertGreater(probs_bad[idx_bad], 0.70)
        print("  [PASS] Test 4: Classifier Prediction Accuracy on Contrasting Inputs")

    def test_05_actual_transcript_evaluation(self):
        """Test end-to-end evaluation on the project's actual transcript.txt."""
        transcript_path = Path("transcript.txt")
        self.assertTrue(transcript_path.exists(), "transcript.txt does not exist!")
        transcript = transcript_path.read_text(encoding="utf-8").strip()
        
        feat = extract_speech_features(transcript, kw_model=self.kw_model)
        emb = self.embedding_model.encode([transcript], normalize_embeddings=True)
        pred = self.classifier.predict(emb)[0]
        
        self.assertIn(pred, ["good", "needs_improvement"])
        self.assertGreater(len(feat["keybert_keywords"]), 0)
        self.assertGreater(feat["avg_keybert_relevance"], 0.25)
        print(f"  [PASS] Test 5: End-to-End Evaluation on transcript.txt (Decision: {pred})")

    def test_06_model_comparison_file_integrity(self):
        """Verify that model comparison results and training data are valid and populated."""
        results_file = Path("model_comparison_results.csv")
        self.assertTrue(results_file.exists(), "model_comparison_results.csv missing!")
        df_res = pd.read_csv(results_file)
        self.assertEqual(len(df_res), 3, "Expected 3 models in comparison table!")
        
        dataset_file = Path("training_data.csv")
        self.assertTrue(dataset_file.exists(), "training_data.csv missing!")
        df_data = pd.read_csv(dataset_file)
        self.assertEqual(len(df_data), 100000, "Expected 100,000 rows in dataset!")
        print("  [PASS] Test 6: Integrity of 100,000-Row Dataset & Comparison File")

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAISpeechEvaluator)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
