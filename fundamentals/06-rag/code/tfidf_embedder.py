import math
from collections import Counter


class TfidfEmbedder:
    name = "tfidf"

    def __init__(self):
        self.vocab = []
        self.idf = []

    @property
    def dimensions(self):
        return len(self.vocab)

    def fit(self, documents):
        self.vocab = self.build_vocabulary(documents)
        self.idf = self.compute_idf(documents, self.vocab)
        return self

    def embed(self, text):
        tf = self.compute_tf(text, self.vocab)
        return [t * i for t, i in zip(tf, self.idf)]

    def embed_batch(self, texts):
        return [self.embed(text) for text in texts]

    @staticmethod
    def build_vocabulary(documents):
        vocab = set()
        for doc in documents:
            vocab.update(doc.lower().split())
        return sorted(vocab)

    @staticmethod
    def compute_tf(text, vocab):
        words = text.lower().split()
        count = Counter(words)
        total = len(words)
        if total == 0:
            return [0.0] * len(vocab)
        return [count.get(word, 0) / total for word in vocab]

    @staticmethod
    def compute_idf(documents, vocab):
        n = len(documents)
        idf = []
        for word in vocab:
            doc_count = sum(1 for doc in documents if word in doc.lower().split())
            idf.append(math.log((n + 1) / (doc_count + 1)) + 1)
        return idf
