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
        return self  # pretrained, nothing to learn from the corpus

    def embed(self, text):
        return self.embed_batch([text])[0]

    def embed_batch(self, texts):
        # plain lists, same as TfidfEmbedder, so similarity.py works unchanged
        return self.model.encode(texts, normalize_embeddings=True).tolist()