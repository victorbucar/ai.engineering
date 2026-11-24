from bm25 import BM25
from embbeding import chunk_text, build_vocabulary, compute_idf, tfidf_embed, vector_search
from faithfulness_eval import evaluate_faithfulness
from hybrid_search_pipeline import hybrid_search
from hyde import hyde_search
from parent_child_chunking import create_parent_child_chunks
from reciprocal_rank_fusion import reciprocal_rank_fusion
from reranker import rerank, rerank_with_cross_encoder
from sample_documents import SAMPLE_DOCUMENTS, SOURCE_NAMES


if __name__ == "__main__":
    print("=" * 65)
    print("STEP 1: BM25 Keyword Search")
    print("=" * 65)

    all_chunks = []
    chunk_sources = []
    for i, doc in enumerate(SAMPLE_DOCUMENTS):
        doc_chunks = chunk_text(doc, chunk_size=50, overlap=10)
        for c in doc_chunks:
            all_chunks.append(c)
            chunk_sources.append(SOURCE_NAMES[i])

    bm25 = BM25()
    bm25.index(all_chunks)

    test_query = "What was revenue last quarter?"
    bm25_results = bm25.search(test_query, top_k=5)
    print(f"  Query: {test_query}")
    print(f"  BM25 top-5:")
    for rank, (idx, score) in enumerate(bm25_results):
        preview = all_chunks[idx][:70].replace("\n", " ")
        print(f"    #{rank+1} [{chunk_sources[idx]}] score={score:.4f} | {preview}...")

    print("\n" + "=" * 65)
    print("STEP 2: Vector Search vs BM25")
    print("=" * 65)

    vocab = build_vocabulary(all_chunks)
    idf = compute_idf(all_chunks, vocab)
    embeddings = [tfidf_embed(c, vocab, idf) for c in all_chunks]

    queries = [
        "What is the refund policy for enterprise customers?",
        "What was revenue last quarter?",
        "How is data encrypted?",
        "What are the API rate limits for enterprise?",
        "What happens if uptime falls below SLA?"
    ]

    for query in queries:
        query_emb = tfidf_embed(query, vocab, idf)
        vec_top1 = vector_search(query_emb, embeddings, top_k=1)[0]
        bm25_top1 = bm25.search(query, top_k=1)[0]

        print(f"\n  Query: {query}")
        print(f"    Vector #1: [{chunk_sources[vec_top1[0]]}] score={vec_top1[1]:.4f}")
        print(f"    BM25   #1: [{chunk_sources[bm25_top1[0]]}] score={bm25_top1[1]:.4f}")
        agree = "AGREE" if chunk_sources[vec_top1[0]] == chunk_sources[bm25_top1[0]] else "DISAGREE"
        print(f"    {agree}")

    print("\n" + "=" * 65)
    print("STEP 3: Reciprocal Rank Fusion (Hybrid Search)")
    print("=" * 65)

    query = "What was revenue last quarter?"
    print(f"  Query: {query}")

    query_emb = tfidf_embed(query, vocab, idf)
    vec_results = vector_search(query_emb, embeddings, top_k=10)
    bm25_results = bm25.search(query, top_k=10)

    print(f"\n  Vector top-3:")
    for rank, (idx, score) in enumerate(vec_results[:3]):
        print(f"    #{rank+1} [{chunk_sources[idx]}] {score:.4f}")

    print(f"\n  BM25 top-3:")
    for rank, (idx, score) in enumerate(bm25_results[:3]):
        print(f"    #{rank+1} [{chunk_sources[idx]}] {score:.4f}")

    fused = reciprocal_rank_fusion([vec_results, bm25_results])
    print(f"\n  RRF fused top-5:")
    for rank, (idx, score) in enumerate(fused[:5]):
        preview = all_chunks[idx][:60].replace("\n", " ")
        print(f"    #{rank+1} [{chunk_sources[idx]}] rrf={score:.4f} | {preview}...")

    print("\n" + "=" * 65)
    print("STEP 4: Reranking")
    print("=" * 65)

    query = "enterprise refund policy"
    print(f"  Query: {query}")

    hybrid_results = hybrid_search(query, all_chunks, embeddings, vocab, idf, bm25, top_k=10)
    reranked = rerank_with_cross_encoder(query, hybrid_results, all_chunks)

    print(f"\n  Before reranking (top-5):")
    for rank, (idx, score) in enumerate(hybrid_results[:5]):
        preview = all_chunks[idx][:60].replace("\n", " ")
        print(f"    #{rank+1} [{chunk_sources[idx]}] score={score:.4f} | {preview}...")

    print(f"\n  After reranking (top-5):")
    for rank, (idx, score) in enumerate(reranked[:5]):
        preview = all_chunks[idx][:60].replace("\n", " ")
        print(f"    #{rank+1} [{chunk_sources[idx]}] score={score:.4f} | {preview}...")

    print("\n" + "=" * 65)
    print("STEP 5: HyDE (Hypothetical Document Embeddings)")
    print("=" * 65)

    query = "How much money did the company make?"
    print(f"  Query: {query}")
    print(f"  (Note: query uses 'money', docs use 'revenue' and 'earnings')")

    query_emb = tfidf_embed(query, vocab, idf)
    direct_results = vector_search(query_emb, embeddings, top_k=3)
    hyde_results, hypothesis = hyde_search(query, all_chunks, embeddings, vocab, idf, top_k=3)

    print(f"\n  Hypothesis: {hypothesis[:100]}...")

    print(f"\n  Direct search top-3:")
    for rank, (idx, score) in enumerate(direct_results):
        print(f"    #{rank+1} [{chunk_sources[idx]}] {score:.4f}")

    print(f"\n  HyDE search top-3:")
    for rank, (idx, score) in enumerate(hyde_results):
        print(f"    #{rank+1} [{chunk_sources[idx]}] {score:.4f}")

    print("\n" + "=" * 65)
    print("STEP 6: Parent-Child Chunking")
    print("=" * 65)

    full_text = " ".join(SAMPLE_DOCUMENTS)
    parents, children, child_to_parent = create_parent_child_chunks(
        full_text, parent_size=100, child_size=25
    )

    print(f"  Total words: {len(full_text.split())}")
    print(f"  Parent chunks: {len(parents)} (100 words each)")
    print(f"  Child chunks: {len(children)} (25 words each)")
    print(f"  Ratio: {len(children)/len(parents):.1f} children per parent")

    child_vocab = build_vocabulary(children)
    child_idf = compute_idf(children, child_vocab)
    child_embeddings = [tfidf_embed(c, child_vocab, child_idf) for c in children]

    query = "enterprise refund 60 days"
    query_emb = tfidf_embed(query, child_vocab, child_idf)
    child_results = vector_search(query_emb, child_embeddings, top_k=3)

    print(f"\n  Query: {query}")
    print(f"\n  Matched children:")
    for rank, (idx, score) in enumerate(child_results):
        parent_idx = child_to_parent[idx]
        print(f"    Child #{idx} (score={score:.4f}):")
        print(f"      Child text: {children[idx][:80]}...")
        print(f"      Parent #{parent_idx}: {parents[parent_idx][:80]}...")

    print("\n" + "=" * 65)
    print("STEP 7: Faithfulness Evaluation")
    print("=" * 65)

    good_answer = (
        "Enterprise customers receive a 60-day refund window. "
        "Refunds are pro-rated from the date of cancellation. "
        "Processing takes 5-7 business days."
    )
    bad_answer = (
        "Enterprise customers receive a 90-day refund window. "
        "Refunds are processed instantly. "
        "There is a $50 processing fee."
    )
    context_chunks = [all_chunks[i] for i, _ in hybrid_search(
        "enterprise refund", all_chunks, embeddings, vocab, idf, bm25, top_k=3
    )]

    good_score, good_ungrounded = evaluate_faithfulness(good_answer, context_chunks)
    bad_score, bad_ungrounded = evaluate_faithfulness(bad_answer, context_chunks)

    print(f"  Context: {len(context_chunks)} chunks about refund policy")
    print(f"\n  Good answer: \"{good_answer[:80]}...\"")
    print(f"  Faithfulness: {good_score:.2f}")
    if good_ungrounded:
        print(f"  Ungrounded claims: {good_ungrounded}")
    else:
        print(f"  All claims grounded in context.")

    print(f"\n  Bad answer: \"{bad_answer[:80]}...\"")
    print(f"  Faithfulness: {bad_score:.2f}")
    if bad_ungrounded:
        print(f"  Ungrounded claims:")
        for claim in bad_ungrounded:
            print(f"    - \"{claim}\"")

    print("\n" + "=" * 65)
    print("STEP 8: Full Advanced RAG Pipeline Comparison")
    print("=" * 65)

    comparison_queries = [
        ("What is the refund policy for enterprise?", "refund"),
        ("What was Q3 revenue?", "earnings"),
        ("How is customer data encrypted?", "security"),
        ("What are the API rate limits?", "api"),
        ("What is the uptime guarantee?", "uptime"),
    ]

    print(f"  {'Query':<45s} {'Vector':>8s} {'BM25':>8s} {'Hybrid':>8s} {'Rerank':>8s}")
    print("  " + "-" * 77)

    for query, expected_source in comparison_queries:
        query_emb = tfidf_embed(query, vocab, idf)

        vec_top = vector_search(query_emb, embeddings, top_k=1)[0]
        vec_hit = "HIT" if chunk_sources[vec_top[0]] == expected_source else "miss"

        bm25_top = bm25.search(query, top_k=1)[0]
        bm25_hit = "HIT" if chunk_sources[bm25_top[0]] == expected_source else "miss"

        hybrid_top = hybrid_search(query, all_chunks, embeddings, vocab, idf, bm25, top_k=1)[0]
        hybrid_hit = "HIT" if chunk_sources[hybrid_top[0]] == expected_source else "miss"

        hybrid_pool = hybrid_search(query, all_chunks, embeddings, vocab, idf, bm25, top_k=10)
        reranked_top = rerank_with_cross_encoder(query, hybrid_pool, all_chunks)[0]
        rerank_hit = "HIT" if chunk_sources[reranked_top[0]] == expected_source else "miss"

        print(f"  {query:<45s} {vec_hit:>8s} {bm25_hit:>8s} {hybrid_hit:>8s} {rerank_hit:>8s}")

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print("  Advanced RAG techniques:")
    print("    1. BM25 keyword search catches exact term matches")
    print("    2. Hybrid search (vector + BM25 + RRF) combines both signals")
    print("    3. Reranking scores candidates more carefully with cross-attention")
    print("    4. HyDE bridges the query-document vocabulary gap")
    print("    5. Parent-child chunking: precise search, rich context")
    print("    6. Faithfulness evaluation catches hallucinated claims")
    print("\n  In production:")
    print("    - Replace TF-IDF with neural embeddings")
    print("    - Replace the simple reranker with a cross-encoder model")
    print("    - Replace HyDE templates with actual LLM hypothesis generation")
    print("    - Add metadata filtering before search")
    print("    - Evaluate with Recall@k and faithfulness on a test set")
