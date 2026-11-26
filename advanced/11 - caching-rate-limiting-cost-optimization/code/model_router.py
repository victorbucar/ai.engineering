"""Model routing: classify a query's complexity, then pick the cheapest model that can handle it.

The classifier embeds the query and finds the nearest labeled example in routing_data.TRAIN_SET
(k-nearest-neighbor, k=1 by default). No LLM call is involved: the embeddings are computed
locally by embeddings.embed(), the same model the semantic cache uses.
"""

import re
from collections import Counter

import numpy as np

from routing_data import TRAIN_SET
from embeddings import embed

LABELS = ("simple", "medium", "complex")

# Cheapest model that can handle each complexity level, per subscription tier.
# Free users never reach the most expensive model. Claude-only, because these are the
# models the pipeline actually calls through the Anthropic SDK.
TIER_MODELS = {
    "free": {"simple": "claude-haiku-5-5", "medium": "claude-haiku-5-5", "complex": "claude-sonnet-5-5"},
    "pro": {"simple": "claude-haiku-5-5", "medium": "claude-sonnet-5-5", "complex": "claude-opus-5-5"},
    "enterprise": {"simple": "claude-sonnet-5-5", "medium": "claude-sonnet-5-5", "complex": "claude-opus-5-5"},
}


class EmbeddingRouter:
    """k-nearest-neighbor classifier over embedded, labeled example queries."""

    def __init__(self, examples=TRAIN_SET, k=1):
        self.examples = list(examples)
        self.k = k
        self._matrix = None  # (n_examples, dim), rows L2-normalized; built on first use

    def _example_matrix(self):
        if self._matrix is None:
            # One batched encode for all examples; done once and reused for every query.
            self._matrix = embed([q for q, _ in self.examples])
        return self._matrix

    def classify(self, query):
        sims = self._example_matrix() @ embed(query)  # cosine similarity (vectors are normalized)
        top = np.argsort(sims)[::-1][: self.k]
        neighbors = [
            {"query": self.examples[i][0], "label": self.examples[i][1], "similarity": round(float(sims[i]), 4)}
            for i in top
        ]
        if self.k == 1:
            label = neighbors[0]["label"]
        else:
            # Majority vote; a tie goes to the label of the single closest neighbor among the tied.
            votes = Counter(n["label"] for n in neighbors)
            best = max(votes.values())
            label = next(n["label"] for n in neighbors if votes[n["label"]] == best)
        return {"label": label, "neighbors": neighbors}


_default_router = None


def classify_complexity(query):
    global _default_router
    if _default_router is None:
        _default_router = EmbeddingRouter()
    return _default_router.classify(query)


def route_model(query, tier="free"):
    result = classify_complexity(query)
    complexity = result["label"]
    models = TIER_MODELS.get(tier, TIER_MODELS["free"])
    return {
        "model": models[complexity],
        "complexity": complexity,
        "tier": tier,
        "nearest_example": result["neighbors"][0],
    }


# --- Previous keyword-based classifier, kept only as a baseline for evaluate_router.py ---

COMPLEX_KEYWORDS = [
    "analyze", "analyse", "trade-offs", "tradeoffs", "pros and cons", "architecture",
    "design", "write code", "implement", "debug", "algorithm", "prove", "optimize",
    "function", "compare",
]
MEDIUM_KEYWORDS = [
    "summarize", "summarise", "explain", "difference", "describe", "translate",
    "rewrite", "list", "report", "vs",
]


def _has_keyword(text, keywords):
    return any(re.search(rf"\b{re.escape(k)}\b", text) for k in keywords)


def classify_complexity_keywords(query):
    text = query.lower()
    words = len(text.split())
    if _has_keyword(text, COMPLEX_KEYWORDS) or words > 60:
        return "complex"
    if _has_keyword(text, MEDIUM_KEYWORDS) or words > 20:
        return "medium"
    return "simple"
