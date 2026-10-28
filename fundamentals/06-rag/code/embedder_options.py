from tfidf_embedder import TfidfEmbedder
from bgem3_embedder import BgeM3Embedder


EMBEDDERS = {
    "1": ("tfidf", TfidfEmbedder),
    "2": ("bge-m3", BgeM3Embedder),
}


def choose_embedder(choice=None):
    if choice is None:
        print("Choose an embedder:")
        print("  [1] tfidf   - TfidfEmbedder, bag-of-words (default)")
        print("  [2] bge-m3  - BAAI/bge-m3, local and free")
        choice = input("Embedder [1]: ").strip() or "1"
    for key, (name, cls) in EMBEDDERS.items():
        if choice.lower() in (key, name):
            return cls
    raise SystemExit(f"Unknown embedder: {choice!r}. Use 1/2 or tfidf/bge-m3.")
