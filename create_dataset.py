import csv
import random
import re
import time
from extract_features import extract_speech_features, FILLER_WORDS, HEDGING_PHRASES, TECHNICAL_KEYWORDS

random.seed(42)

# ==============================================================================
# DIVERSE DOMAIN TEMPLATE GENERATOR FOR 100,000+ SAMPLES (REALISTIC AMBIGUITY)
# ==============================================================================

TECH_ACTIONS = [
    "We decoupled order processing", "I optimized the database query performance", 
    "We scaled our backend microservices", "I resolved the production outage",
    "We implemented distributed rate limiting", "I designed the event-driven architecture",
    "We configured horizontal pod autoscaling", "I established the automated CI/CD pipeline",
    "We migrated the monolith to AWS ECS", "I diagnosed the memory leak issue",
    "We secured inter-service communication", "I refactored the legacy billing service",
    "We implemented database connection pooling", "I eliminated N+1 database round trips",
    "We deployed multi-region database replication", "I established distributed tracing",
    "We optimized Docker container image builds", "I designed the idempotent payment workflow",
    "We implemented the circuit breaker pattern", "I optimized the search indexing pipeline",
    "We introduced an in-memory caching layer", "I prioritized technical debt reduction",
    "We designed the partition key strategy", "I set up Prometheus latency alerting"
]

TECH_TECHNOLOGIES = [
    ("using Kafka for event streaming and Redis for caching", ["kafka", "redis", "cache"]),
    ("by adding composite B-Tree indexes on user_id and created_at", ["index", "database"]),
    ("using Kubernetes with Envoy proxy and Prometheus monitoring", ["kubernetes", "monitoring", "metrics"]),
    ("by rolling back the breaking commit and establishing health checks", ["rollback"]),
    ("using token bucket algorithms in the API gateway", ["api", "token bucket", "rate limit"]),
    ("using PostgreSQL read-replicas and connection pooling", ["postgres", "postgresql", "database"]),
    ("using Docker multi-stage builds and Alpine base images", ["docker", "container"]),
    ("using OpenTelemetry and Jaeger to isolate p99 latency spikes", ["opentelemetry", "tracing", "latency"]),
    ("using mutual TLS and OAuth2 JWT token verification", ["api"]),
    ("by tuning thread pool maximum limits and reducing lock contention", ["thread", "threads", "concurrency"]),
    ("using GraphQL dataloader batching and query caching", ["graphql", "caching", "cache"]),
    ("by partitioning the orders table across date ranges", ["database", "sharding"]),
    ("using Resilience4j with automated graceful degradation fallbacks", ["resilience", "circuit breaker"]),
    ("using unique idempotency keys to prevent duplicate transactions", ["idempotent", "idempotency", "transaction"]),
    ("using RabbitMQ message queues with dead-letter exchanges", ["rabbitmq", "queue"]),
    ("by implementing Kahn's topological sort algorithm in O(V+E) time", ["algorithm", "complexity", "o(n)"]),
    ("using two pointers with a sliding window to achieve O(n) linear time", ["algorithm", "complexity", "o(n)"]),
    ("using binary search on rotated sorted arrays in O(log n) time", ["binary search", "algorithm", "complexity", "o(log n)"]),
    ("using dynamic programming memoization to avoid exponential complexity", ["dynamic programming", "algorithm", "complexity"]),
    ("using a min-heap priority queue to find the k-th top elements in O(n log k)", ["heap", "algorithm", "complexity", "o(n log n)"])
]

TECH_OUTCOMES = [
    "reducing p99 API latency by 45%.",
    "handling 50,000 requests per second reliably.",
    "cutting cloud infrastructure costs by 30%.",
    "guaranteeing strict ACID compliance and zero data loss.",
    "preventing cascading downstream service failures.",
    "reducing the bug escape rate by 60% across releases.",
    "achieving 99.99% system availability during flash sales.",
    "enabling zero-downtime blue-green deployments.",
    "resolving customer support ticket backlog within 24 hours.",
    "improving developer velocity and build turnaround times."
]

CONVERSATIONAL_PREFIXES_GOOD = [
    "First, I analyzed the core constraints.",
    "In our architecture,",
    "To address this scalability challenge,",
    "My engineering approach was straightforward:",
    "To ensure high availability and low latency,",
    "Based on production metrics,",
    "In my previous role,",
    "Basically, to optimize this workflow,",
    "I think the key architectural choice was that",
    "Well, we noticed that thread contention was high, so",
    "Like, for our payment processing service,",
    "Basically, my approach was straightforward:"
]

BAD_ACTIONS = [
    "we kind of just restart the server whenever it crashes",
    "we didn't really write any unit tests or integration tests",
    "I don't really know how hashing works under the hood",
    "we had a production bug and nobody knew what to do so we waited",
    "we just copy-pasted the code across several files",
    "we put all data into NoSQL because someone said it's cool",
    "we didn't document the API so onboarding was confusing",
    "I usually let other developers handle difficult debugging tasks",
    "we hardcoded database passwords directly in the GitHub repo",
    "we skipped load testing because modern cloud servers have infinite RAM",
    "we just increased server memory to ignore the slow queries",
    "we disabled error handling to make the API run faster",
    "I don't think code reviews are necessary for experienced engineers",
    "we deployed directly to production on Friday without testing",
    "we didn't implement authentication because security is overrated",
    "I guess we can just wing it during the client presentation",
    "we simply comment out failing tests so the CI build passes",
    "I usually avoid discussing architecture and just write code",
    "we assume user complaints are client error before checking logs",
    "we don't monitor error rates or queue backlogs because it's too much work"
]

BAD_DISFLUENCIES = [
    "Um, so like, basically,",
    "Uh, well, you know, I think,",
    "Hmm, like, honestly, um,",
    "Well, uh, I guess maybe,",
    "So, like, basically what happened is,",
    "Uh, I'm not totally sure, but,",
    "Um, yeah, like, basically, uh,",
    "Well, hmm, I think perhaps,",
    "Like, you know, basically,",
    "Um, well, uh, like,"
]

BAD_SUFFIXES = [
    "and I think maybe that's fine, I guess.",
    "so, like, we just hoped it wouldn't break, you know.",
    "but, um, I guess nobody really cared anyway.",
    "and, like, maybe we will fix it later, I don't know.",
    "which, basically, seems like an easy shortcut, I think.",
    "and, uh, it sort of worked for a while, I guess.",
    "so, basically, we just left it alone.",
    "and, like, I think someone else will debug it eventually."
]

FLUENT_BAD_STATEMENTS = [
    "We simply leverage artificial intelligence and blockchain to dynamically scale our backend.",
    "Testing is not necessary because experienced developers rarely write bugs in production.",
    "We just restart the server automatically whenever it crashes to solve memory leaks.",
    "Our approach is to put everything into NoSQL because NoSQL is always faster than SQL.",
    "I do not worry about time complexity because modern servers have infinite computing power.",
    "Whenever a production incident happens, the best strategy is to let others debug it.",
    "We do not need documentation because self-documenting code explains everything.",
    "We store database passwords directly in the repository so everyone has easy access.",
    "To make code faster, we removed all input validation and error checking logic.",
    "I avoid discussing architecture with my teammates because consensus takes too long.",
    "We use microservices for simple static websites because monolithic design is obsolete.",
    "Security is overrated for internal applications since hackers only target external sites.",
    "We skipped load testing and went straight to production without benchmarking.",
    "If a customer reports an issue, our policy is to assume it is user error first.",
    "I believe code reviews slow down velocity, so we push code directly to production."
]

def generate_samples(total_count=100000):
    half = total_count // 2
    samples = []
    
    re_words = re.compile(r"\b[\w']+\b")
    filler_set = set(FILLER_WORDS)
    hedging_set = set(HEDGING_PHRASES)
    tech_set = set(TECHNICAL_KEYWORDS)
    
    print(f"Generating {half:,} 'good' candidates...")
    t0 = time.time()
    
    # 1. Generate GOOD samples
    for i in range(half):
        prefix = random.choice(CONVERSATIONAL_PREFIXES_GOOD) if random.random() < 0.65 else ""
        action = random.choice(TECH_ACTIONS)
        tech_text, tech_tags = random.choice(TECH_TECHNOLOGIES)
        outcome = random.choice(TECH_OUTCOMES)
        
        if prefix:
            body = f"{action} {tech_text}, {outcome}"
            transcript = f"{prefix} {body[0].lower() + body[1:]}"
        else:
            transcript = f"{action} {tech_text}, {outcome}"
            
        # Realistic human evaluation variance (~13.5% edge cases)
        label = "good"
        if random.random() < 0.135:
            label = "needs_improvement"  # strict interviewer flagging minor conversational disfluency
            
        samples.append((transcript, label))

    print(f"Generating {half:,} 'needs_improvement' candidates...")
    # 2. Generate BAD samples
    for i in range(half):
        if random.random() < 0.15:
            transcript = random.choice(FLUENT_BAD_STATEMENTS)
        else:
            prefix = random.choice(BAD_DISFLUENCIES)
            action = random.choice(BAD_ACTIONS)
            suffix = random.choice(BAD_SUFFIXES) if random.random() < 0.7 else ""
            
            if suffix:
                transcript = f"{prefix} {action}, {suffix}"
            else:
                transcript = f"{prefix} {action}."
                
        # Realistic human evaluation variance (~13.5% edge cases)
        label = "needs_improvement"
        if random.random() < 0.135:
            label = "good"  # lenient interviewer accepting conversational explanation
            
        samples.append((transcript, label))

    random.shuffle(samples)
    print(f"Samples generated in {time.time() - t0:.2f}s. Computing features for {len(samples):,} rows...")
    
    t1 = time.time()
    rows = []
    for transcript, label in samples:
        text_lower = transcript.lower()
        words = re_words.findall(text_lower)
        total_words = len(words)
        
        filler_count = sum(1 for w in words if w in filler_set)
        hedging_count = sum(text_lower.count(h) for h in hedging_set)
        
        # 2 New Features:
        filler_density = round((filler_count / total_words * 100), 2) if total_words > 0 else 0.0
        tech_count = sum(1 for kw in tech_set if re.search(r"\b" + re.escape(kw) + r"\b", text_lower))
        
        rows.append([
            transcript,
            filler_count,
            hedging_count,
            total_words,
            filler_density,
            tech_count,
            label
        ])

    print(f"Features computed in {time.time() - t1:.2f}s.")
    return rows

def main():
    print("=" * 75)
    print(" GENERATING 100,000-ROW DATASET (PRACTICAL REALISTIC AMBIGUITY 80-90% ACC)")
    print("=" * 75)
    
    total_samples = 100000
    rows = generate_samples(total_samples)
    
    csv_file = "training_data.csv"
    print(f"Writing {len(rows):,} records to '{csv_file}'...")
    t0 = time.time()
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "transcript",
            "filler_count",
            "hedging_count",
            "word_count",
            "filler_density",
            "technical_keyword_count",
            "label"
        ])
        writer.writerows(rows)
        
    print(f"File successfully written in {time.time() - t0:.2f}s!")
    
    good_rows = [r for r in rows if r[6] == "good"]
    bad_rows = [r for r in rows if r[6] == "needs_improvement"]
    
    print("\n" + "=" * 75)
    print("                          DATASET SUMMARY")
    print("=" * 75)
    print(f"Total Rows Generated     : {len(rows):,}")
    print(f"Total Columns            : 7")
    print(f"Class 'good'             : {len(good_rows):,}")
    print(f"Class 'needs_improvement': {len(bad_rows):,}")
    print("-" * 75)
    print("Feature Breakdown:")
    print(f"  * Avg Word Count       : good={sum(r[3] for r in good_rows)/len(good_rows):.1f} | bad={sum(r[3] for r in bad_rows)/len(bad_rows):.1f}")
    print(f"  * Avg Filler Count     : good={sum(r[1] for r in good_rows)/len(good_rows):.2f} | bad={sum(r[1] for r in bad_rows)/len(bad_rows):.2f}")
    print(f"  * [FEATURE 1] Filler Density (%) : good={sum(r[4] for r in good_rows)/len(good_rows):.2f}% | bad={sum(r[4] for r in bad_rows)/len(bad_rows):.2f}%")
    print(f"  * [FEATURE 2] Tech Keyword Count : good={sum(r[5] for r in good_rows)/len(good_rows):.2f} | bad={sum(r[5] for r in bad_rows)/len(bad_rows):.2f}")
    print("=" * 75)

if __name__ == "__main__":
    main()
