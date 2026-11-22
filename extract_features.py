import os
import re
import warnings
import logging
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

from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

# Speech fluency markers (filler words & hedging disfluencies)
FILLER_WORDS = [
    "um", "umm", "uh", "hmm", "ah", "like", "basically", "actually", "you know"
]

HEDGING_PHRASES = [
    "maybe", "probably", "perhaps", "i think", "i guess", "sort of", "kind of"
]

# Legacy technical keyword list retained for backward compatibility & fallback
TECHNICAL_KEYWORDS = [
    "api", "database", "sql", "nosql", "postgres", "postgresql", "mongodb", "redis",
    "cache", "caching", "latency", "throughput", "scalability", "scalable", "sharding",
    "microservice", "microservices", "kafka", "rabbitmq", "queue", "docker", "kubernetes",
    "container", "index", "indexes", "indexing", "concurrency", "thread", "threads",
    "asynchronous", "async", "load balancer", "load balancing", "failover", "replication",
    "acid", "idempotent", "idempotency", "rest", "graphql", "grpc", "ci/cd", "pipeline",
    "metric", "metrics", "prometheus", "grafana", "monitoring", "heap", "stack", "tree",
    "binary search", "dynamic programming", "hash", "algorithm", "complexity", "o(n)",
    "o(1)", "o(log n)", "o(n log n)", "transaction", "rollback", "deploy", "deployment",
    "opentelemetry", "tracing", "resilience", "circuit breaker", "rate limit", "token bucket"
]

# Singleton holder for KeyBERT model to avoid re-allocating memory
_KW_MODEL = None

def get_keybert_model(embedding_model=None):
    """Returns a singleton KeyBERT instance, reusing an existing SentenceTransformer model if provided."""
    global _KW_MODEL
    if embedding_model is not None:
        return KeyBERT(model=embedding_model)
    if _KW_MODEL is None:
        _KW_MODEL = KeyBERT(model="all-MiniLM-L6-v2")
    return _KW_MODEL

def extract_speech_features(
    text: str,
    kw_model: KeyBERT = None,
    embedding_model: SentenceTransformer = None,
    top_n: int = 8,
    use_keybert: bool = True
) -> dict:
    """
    Extract linguistic delivery features and dynamic BERT-based semantic keyphrases.
    
    Replaces static hardcoded technical keyword matching with KeyBERT:
    - KeyBERT uses dense BERT embeddings (all-MiniLM-L6-v2) and cosine similarity
      to dynamically discover the most topically salient 1-gram & 2-gram keyphrases.
    - Computes semantic relevance scores reflecting technical depth and focus.
    """
    text_lower = text.lower()
    words = re.findall(r"\b[\w']+\b", text_lower)
    total_words = len(words)
    
    # 1. Vocal Filler Words
    filler_words_found = [w for w in words if w in FILLER_WORDS]
    filler_count = len(filler_words_found)
    
    # 2. Hedging & Hesitation Markers
    hedging_phrases_found = [p for p in HEDGING_PHRASES if p in text_lower]
    hedging_count = sum(text_lower.count(p) for p in HEDGING_PHRASES)
    
    # 3. Disfluency Density (%)
    filler_density = round((filler_count / total_words * 100), 2) if total_words > 0 else 0.0
    hedging_density = round((hedging_count / total_words * 100), 2) if total_words > 0 else 0.0
    
    # 4. Dynamic KeyBERT Semantic Keyphrase Extraction (No hardcoded keyword list!)
    keybert_results = []
    if use_keybert and total_words >= 3:
        try:
            if kw_model is None:
                kw_model = get_keybert_model(embedding_model)
            keybert_results = kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),
                stop_words="english",
                top_n=top_n
            )
        except Exception:
            keybert_results = []
            
    keybert_phrases = [kw for kw, _ in keybert_results]
    keybert_scores = {kw: round(score, 4) for kw, score in keybert_results}
    avg_keybert_relevance = (
        round(sum(score for _, score in keybert_results) / len(keybert_results), 4)
        if keybert_results else 0.0
    )
    
    # 5. Technical Density based on KeyBERT
    keybert_keyword_count = len(keybert_phrases)
    keybert_density = round((keybert_keyword_count / total_words * 100), 2) if total_words > 0 else 0.0
    
    # 6. Lexical Diversity (Type-Token Ratio)
    lexical_diversity = round((len(set(words)) / total_words), 3) if total_words > 0 else 0.0

    return {
        "total_words": total_words,
        "filler_count": filler_count,
        "hedging_count": hedging_count,
        "filler_density": filler_density,
        "hedging_density": hedging_density,
        # KeyBERT dynamic extractions:
        "keybert_keywords": keybert_results,       # List of (phrase, score)
        "keybert_phrases": keybert_phrases,         # List of strings
        "keybert_scores": keybert_scores,           # Dict of phrase -> score
        "avg_keybert_relevance": avg_keybert_relevance, # Mean semantic relevance
        "keybert_keyword_count": keybert_keyword_count,
        "keybert_density": keybert_density,
        # Backward compatibility aliases:
        "technical_keyword_count": keybert_keyword_count,
        "technical_keywords_found": keybert_phrases,
        "technical_density": keybert_density,
        "lexical_diversity": lexical_diversity,
        "filler_words_found": filler_words_found,
        "hedging_phrases_found": hedging_phrases_found,
    }

if __name__ == "__main__":
    transcript_file = Path("transcript.txt")
    if not transcript_file.exists():
        raise FileNotFoundError("transcript.txt not found!")
    
    transcript = transcript_file.read_text(encoding="utf-8")
    print("Extracting features using KeyBERT (all-MiniLM-L6-v2)...")
    features = extract_speech_features(transcript)
    
    print("=" * 70)
    print("     KEYBERT-POWERED LINGUISTIC & SEMANTIC FEATURE EXTRACTION")
    print("=" * 70)
    print(f"Total Words                : {features['total_words']}")
    print(f"Filler Words               : {features['filler_count']} {features['filler_words_found'][:6]}")
    print(f"Hedging Phrases            : {features['hedging_count']} {features['hedging_phrases_found']}")
    print(f"Filler Density (%)         : {features['filler_density']}%")
    print(f"Hedging Density (%)        : {features['hedging_density']}%")
    print("-" * 70)
    print("KeyBERT Extracted Keyphrases (Semantic Relevance):")
    for phrase, score in features["keybert_keywords"]:
        bar = "#" * int(score * 20)
        print(f"  * {phrase:<24} : {score:.4f}  {bar}")
    print(f"Avg KeyBERT Semantic Depth : {features['avg_keybert_relevance']}")
    print(f"Lexical Diversity (TTR)    : {features['lexical_diversity']}")
    print("=" * 70)
