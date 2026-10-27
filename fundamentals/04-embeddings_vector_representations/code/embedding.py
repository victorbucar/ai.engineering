import math
import sys
import numpy as np
from chunking import chunk_text, chunk_by_sentences
from similarity import cosine_similarity, dot_product, euclidean_distance
from simple_embedder import SimpleEmbedder
from embedders import choose_embedder
from semantic_search_engine import SemanticSearchEngine
from metrics_comparer import compare_metrics


SAMPLE_DOCUMENTS = [
    """Acme Corp Refund Policy.
    All standard plan customers are eligible for a full refund within 30 days of purchase.
    Enterprise plan customers receive an extended 60-day refund window with pro-rated refunds
    calculated from the date of cancellation. Refunds are processed within 5-7 business days
    and returned to the original payment method. No refunds are available after the refund
    window closes. Customers must submit refund requests through the support portal or by
    contacting their account manager directly. Annual subscriptions that are cancelled mid-term
    will receive a pro-rated credit for the remaining months.""",

    """Acme Corp Product Overview.
    Acme Corp offers three product tiers: Starter, Professional, and Enterprise.
    The Starter plan includes basic features for individual users at $29 per month.
    The Professional plan adds team collaboration, advanced analytics, and priority
    support for $99 per month per user. The Enterprise plan includes everything in
    Professional plus custom integrations, dedicated account management, SSO,
    audit logs, and a 99.99% uptime SLA. Enterprise pricing is custom and starts
    at $500 per month for up to 50 users. All plans include a 14-day free trial
    with no credit card required.""",

    """Acme Corp Security Practices.
    Acme Corp maintains SOC 2 Type II compliance and undergoes annual third-party
    security audits. All data is encrypted at rest using AES-256 and in transit
    using TLS 1.3. Customer data is stored in isolated tenants within AWS
    us-east-1 and eu-west-1 regions. Data residency can be configured per
    organization for Enterprise customers. Backups are performed every 6 hours
    with 30-day retention. Acme Corp does not sell or share customer data with
    third parties. Enterprise customers can request data deletion within 24 hours.
    Bug bounty program available through HackerOne.""",

    """Acme Corp API Documentation.
    The Acme API uses REST with JSON request and response bodies. Authentication
    is via Bearer tokens issued through OAuth 2.0. Rate limits are 100 requests
    per minute for Starter, 1000 for Professional, and 10000 for Enterprise.
    Rate limit headers are included in every response: X-RateLimit-Limit,
    X-RateLimit-Remaining, and X-RateLimit-Reset. Exceeding the rate limit
    returns HTTP 429 with a Retry-After header. The API supports pagination
    via cursor-based pagination using the next_cursor field. Webhooks are
    available for real-time event notifications on Professional and Enterprise
    plans. API versioning uses date-based versions in the URL path.""",

    """Acme Corp Uptime and Reliability.
    Acme Corp guarantees 99.9% uptime for Professional plans and 99.99% uptime
    for Enterprise plans. Uptime is calculated monthly excluding scheduled
    maintenance windows which are announced 72 hours in advance. If uptime
    falls below the guaranteed level, customers receive service credits:
    10% credit for each 0.1% below the SLA threshold, up to a maximum of
    30% of the monthly fee. Service credits must be requested within 30 days
    of the incident. Status page updates are posted at status.acme.com
    within 5 minutes of any detected incident. Post-incident reports are
    published within 48 hours for any outage exceeding 15 minutes."""
]


if __name__ == "__main__":
    Embedder = choose_embedder(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"\nUsing embedder: {Embedder.name}\n")

    print("=" * 60)
    print("STEP 1: Document Chunking")
    print("=" * 60)

    sample = SAMPLE_DOCUMENTS[0]
    fixed_chunks = chunk_text(sample, chunk_size=30, overlap=10)
    sentence_chunks = chunk_by_sentences(sample, max_chunk_tokens=30)
    print(f"  Document length: {len(sample.split())} words")
    print(f"\n  Fixed-size chunking (30 words, 10 overlap):")
    print(f"    Chunks: {len(fixed_chunks)}")
    for i, chunk in enumerate(fixed_chunks[:3]):
        print(f"    [{i}] {chunk[:80]}...")
    print(f"\n  Sentence-based chunking (max 30 tokens):")
    print(f"    Chunks: {len(sentence_chunks)}")
    for i, chunk in enumerate(sentence_chunks[:3]):
        print(f"    [{i}] {chunk[:80]}...")

    print("\n" + "=" * 60)
    print("STEP 2: Embedding")
    print("=" * 60)

    mini_docs = [
        "The cat sat on the mat",
        "The dog sat on the rug",
        "Machine learning is a branch of artificial intelligence",
        "Payment transaction was declined by the bank",
        "My credit card charge did not go through"
    ]
    embedder = Embedder()
    embedder.fit(mini_docs)
    embeddings = embedder.embed_batch(mini_docs)

    if hasattr(embedder, "vocab"):
        print(f"  Vocabulary size: {len(embedder.vocab)}")
    print(f"  Embedding dimensions: {len(embeddings[0])}")
    print(f"  Non-zero entries per embedding:")
    for i, emb in enumerate(embeddings):
        nonzero = int(np.count_nonzero(emb))
        print(f"    [{i}] \"{mini_docs[i][:40]}\" -> {nonzero} non-zero dims")

    print("\n" + "=" * 60)
    print("STEP 3: Similarity Metrics Comparison")
    print("=" * 60)

    pairs = [
        (0, 1, "cat/mat vs dog/rug (similar)"),
        (0, 2, "cat/mat vs ML (unrelated)"),
        (3, 4, "payment declined vs charge didn't go through (same meaning)"),
        (2, 3, "ML vs payment declined (unrelated)")
    ]

    for i, j, desc in pairs:
        cos = cosine_similarity(embeddings[i], embeddings[j])
        dot = dot_product(embeddings[i], embeddings[j])
        euc = euclidean_distance(embeddings[i], embeddings[j])
        print(f"\n  {desc}:")
        print(f"    Cosine:    {cos:.4f}")
        print(f"    Dot:       {dot:.4f}")
        print(f"    Euclidean: {euc:.4f}")

    print("\n" + "=" * 60)
    print("STEP 4: Semantic Search Engine")
    print("=" * 60)

    engine = SemanticSearchEngine(chunk_size=50, overlap=10, embedder=Embedder())
    source_names = [
        "refund-policy.md",
        "product-overview.md",
        "security.md",
        "api-docs.md",
        "uptime-sla.md"
    ]
    num_chunks = engine.index_documents(SAMPLE_DOCUMENTS, source_names)
    print(f"  Indexed {len(SAMPLE_DOCUMENTS)} documents into {num_chunks} chunks")
    if hasattr(engine.embedder, "vocab"):
        print(f"  Vocabulary size: {len(engine.embedder.vocab)} terms")
    print(f"  Embedding dimensions: {engine.embedder.dimensions}")

    queries = [
        "What is the refund policy for enterprise customers?",
        "What are the API rate limits?",
        "How is customer data encrypted?",
        "What happens if uptime falls below the SLA?",
        "How much does the Professional plan cost?"
    ]

    for query in queries:
        print(f"\n  Query: \"{query}\"")
        results = engine.search_with_scores(query, top_k=3)
        for r in results:
            print(f"    [{r['source']}] score={r['score']:.4f} | {r['text'][:70]}...")

    print("\n" + "=" * 60)
    print("STEP 5: Metric Comparison on Full Corpus")
    print("=" * 60)

    test_query = "How is data encrypted at rest?"
    print(f"  Query: \"{test_query}\"")
    comparison = compare_metrics(engine, test_query, top_k=3)
    for metric, hits in comparison.items():
        print(f"\n  {metric.upper()}:")
        for h in hits:
            print(f"    score={h['score']:>8.4f} | {h['preview']}...")

    print("\n" + "=" * 60)
    print("STEP 6: Embedding Truncation (Matryoshka Simulation)")
    print("=" * 60)

    full_dim = engine.embedder.dimensions
    query_full = engine.embedder.embed("refund policy enterprise")
    doc_full = engine.embedder.embed(SAMPLE_DOCUMENTS[0][:200])

    for frac in [1.0, 0.5, 0.25, 0.1]:
        dims = max(1, int(full_dim * frac))
        q_trunc = SimpleEmbedder.truncate_embedding(query_full, dims)
        d_trunc = SimpleEmbedder.truncate_embedding(doc_full, dims)
        sim = cosine_similarity(q_trunc, d_trunc)
        print(f"  dims={dims:>4d} ({frac*100:>5.1f}%): cosine={sim:.4f}")

    print("\n" + "=" * 60)
    print("STEP 7: Binary Quantization")
    print("=" * 60)

    query_vec = engine.embedder.embed("API rate limits")
    results_full = engine.index.search(query_vec, top_k=5, metric="cosine")
    results_binary = engine.index.search(query_vec, top_k=5, metric="hamming")

    full_ids = [r["index"] for r in results_full]
    binary_ids = [r["index"] for r in results_binary]
    overlap = len(set(full_ids) & set(binary_ids))

    print(f"  Query: \"API rate limits\"")
    print(f"  Full-precision top-5 indices: {full_ids}")
    print(f"  Binary quant top-5 indices:   {binary_ids}")
    print(f"  Overlap: {overlap}/5 ({overlap/5*100:.0f}%)")

    storage_full = full_dim * 4
    storage_binary = math.ceil(full_dim / 8)
    print(f"\n  Storage per vector:")
    print(f"    Float32: {storage_full:,} bytes")
    print(f"    Binary:  {storage_binary:,} bytes")
    print(f"    Ratio:   {storage_full/storage_binary:.0f}x reduction")

    print("\n" + "=" * 60)
    print("STEP 8: Chunk Size Experiment")
    print("=" * 60)

    test_query = "What is the refund policy for enterprise customers?"
    for chunk_size in [20, 50, 100, 200]:
        eng = SemanticSearchEngine(chunk_size=chunk_size, overlap=max(5, chunk_size // 5), embedder=Embedder())
        n = eng.index_documents(SAMPLE_DOCUMENTS)
        results = eng.search(test_query, top_k=3)
        top_score = results[0]["score"] if results else 0
        print(f"  chunk_size={chunk_size:>3d}: {n:>3d} chunks, "
              f"top_score={top_score:.4f}, "
              f"top_preview=\"{results[0]['text'][:50]}...\"")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Embedder: {engine.embedder.name}")
    print(f"  Documents indexed: {len(SAMPLE_DOCUMENTS)}")
    print(f"  Total chunks: {num_chunks}")
    if hasattr(engine.embedder, "vocab"):
        print(f"  Vocabulary size: {len(engine.embedder.vocab)}")
    print(f"  Embedding dimensions: {engine.embedder.dimensions}")
    print("  Metrics implemented: cosine, dot product, euclidean, hamming")
    print("  Chunking: fixed-size + sentence-based")
    print("  Advanced: Matryoshka truncation, binary quantization")
    print("\n  In production, replace SimpleEmbedder with:")
    print("    OpenAI text-embedding-3-small (1536d, $0.02/1M tokens)")
    print("    BGE-M3 (1024d, free, open source)")
    print("    Voyage-3 (1024d, $0.06/1M tokens)")
