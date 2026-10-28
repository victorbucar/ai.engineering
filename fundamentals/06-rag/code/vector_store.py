from similarity import cosine_similarity


class VectorStore:
    def __init__(self, similarity=cosine_similarity):
        self.similarity = similarity
        self.clear()

    def clear(self):
        self.embeddings = []
        self.texts = []
        self.metadata = []

    def add(self, embedding, text, meta=None):
        self.embeddings.append(embedding)
        self.texts.append(text)
        self.metadata.append(meta or {})

    def search(self, query_embedding, top_k=5):
        scores = [
            (i, self.similarity(query_embedding, emb))
            for i, emb in enumerate(self.embeddings)
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [
            {
                "chunk": self.texts[idx],
                "metadata": self.metadata[idx],
                "score": score,
                "index": idx,
            }
            for idx, score in scores[:top_k]
        ]

    def __len__(self):
        return len(self.embeddings)
