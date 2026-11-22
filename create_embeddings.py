from sentence_transformers import SentenceTransformer
from pathlib import Path

# Read transcript
transcript = Path("transcript.txt").read_text(encoding="utf-8")

# Load pretrained NLP model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert text into numerical embedding
embedding = model.encode(transcript)

print("Embedding created successfully!")
print("Embedding dimensions:", len(embedding))
print("First 5 values:", embedding[:5])