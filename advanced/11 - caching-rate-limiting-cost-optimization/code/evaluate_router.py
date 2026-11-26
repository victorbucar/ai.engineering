"""Measure routing-classifier accuracy on the held-out test set. No LLM calls.

    python evaluate_router.py
"""

import warnings

warnings.filterwarnings("ignore")

from model_router import LABELS, EmbeddingRouter, classify_complexity_keywords
from routing_data import TEST_SET, TRAIN_SET


def confusion_matrix(pairs):
    """pairs: list of (expected, predicted). Rows = expected, columns = predicted."""
    m = {e: {p: 0 for p in LABELS} for e in LABELS}
    for expected, predicted in pairs:
        m[expected][predicted] += 1
    return m


def print_report(name, pairs):
    correct = sum(e == p for e, p in pairs)
    print(f"\n=== {name}: {correct}/{len(pairs)} correct = {correct / len(pairs):.0%} accuracy ===")
    m = confusion_matrix(pairs)
    print(f"  {'expected / predicted':<22}" +"".join(f"{p:>9}" for p in LABELS) + "   recall")
    for e in LABELS:
        total = sum(m[e].values())
        recall = m[e][e] / total if total else 0
        print(f"  {e:<22}" + "".join(f"{m[e][p]:>9}" for p in LABELS) + f"   {recall:>5.0%}")
    return correct / len(pairs)


def main():
    print(f"Train: {len(TRAIN_SET)} labeled examples   Test: {len(TEST_SET)} held-out queries")

    # 1-nearest-neighbor: the classifier the router uses.
    router = EmbeddingRouter(k=1)
    results = [(q, expected, router.classify(q)) for q, expected in TEST_SET]
    print_report("Embedding router, k=1 (nearest labeled example)", [(e, r["label"]) for _, e, r in results])

    print("\n  Misclassified queries (with the training example they matched):")
    errors = [(q, e, r) for q, e, r in results if r["label"] != e]
    for q, expected, r in errors:
        n = r["neighbors"][0]
        print(f"    - expected {expected}, got {r['label']}: \"{q[:70]}\"")
        print(f"        nearest (sim {n['similarity']:.2f}, {n['label']}): \"{n['query'][:70]}\"")
    if not errors:
        print("    (none)")

    print("\n  Similarity to the nearest example (low = the training set has nothing like this query):")
    for q, expected, r in sorted(results, key=lambda x: x[2]["neighbors"][0]["similarity"])[:5]:
        print(f"    {r['neighbors'][0]['similarity']:.2f}  {q[:70]}")

    # Comparisons: majority vote over 3 neighbors, and the old keyword classifier.
    router3 = EmbeddingRouter(k=3)
    print_report("Embedding router, k=3 (majority vote)", [(e, router3.classify(q)["label"]) for q, e in TEST_SET])
    print_report("Baseline: old keyword classifier", [(e, classify_complexity_keywords(q)) for q, e in TEST_SET])


if __name__ == "__main__":
    main()
