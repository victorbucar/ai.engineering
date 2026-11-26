"""Sentence embeddings shared by the semantic cache and the model router.

Runs locally with sentence-transformers (no API call). The model is downloaded on first use
(~420 MB, then cached) and loaded once per process, the first time embed() is called.
"""

import numpy as np

# Chosen over all-MiniLM-L6-v2, bge-small-en-v1.5 and paraphrase-multilingual-MiniLM-L12-v2:
# on this demo's queries it separated paraphrases from different-topic questions best.
EMBEDDING_MODEL = "all-mpnet-base-v2"

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed(text):
    """Embed one string (returns a 768-dim vector) or a list of strings (returns an n x 768 matrix).

    Vectors are L2-normalized, so the cosine similarity of two vectors is their dot product.
    """
    return _get_model().encode(text, normalize_embeddings=True)


def cosine_similarity(a, b):
    """Cosine similarity of two vectors returned by embed()."""
    return float(np.dot(a, b))
