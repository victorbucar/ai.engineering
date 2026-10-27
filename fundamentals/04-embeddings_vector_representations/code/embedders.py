import numpy as np
from simple_embedder import SimpleEmbedder


class BgeM3Embedder:
    name = "bge-m3"
    _model = None  # shared across instances so the 2.2 GB model loads once

    def __init__(self, model_name="BAAI/bge-m3"):
        if BgeM3Embedder._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"  Loading {model_name} (first run downloads ~2.2 GB)...")
            BgeM3Embedder._model = SentenceTransformer(model_name)
        self.model = BgeM3Embedder._model
        self.dimensions = self.model.get_embedding_dimension()

    def fit(self, documents):
        pass  # pretrained, nothing to learn from the corpus

    def embed(self, text):
        return self.embed_batch([text])[0]

    def embed_batch(self, texts):
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return [np.asarray(v, dtype=np.float64) for v in vecs]


EMBEDDERS = {
    "1": ("tfidf", SimpleEmbedder),
    "2": ("bge-m3", BgeM3Embedder),
}


def choose_embedder(choice=None):
    if choice is None:
        print("Choose an embedder:")
        print("  [1] tfidf   - SimpleEmbedder, bag-of-words (default)")
        print("  [2] bge-m3  - BAAI/bge-m3, local and free")
        choice = input("Embedder [1]: ").strip() or "1"
    for key, (name, cls) in EMBEDDERS.items():
        if choice.lower() in (key, name):
            return cls
    raise SystemExit(f"Unknown embedder: {choice!r}. Use 1/2 or tfidf/bge-m3.")
