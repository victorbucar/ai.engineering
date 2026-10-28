from text_chunker import TextChunker
from tfidf_embedder import TfidfEmbedder
from vector_store import VectorStore
from prompt_builder import PromptBuilder
from extractive_generator import ExtractiveGenerator


# Any component can be swapped as long as it keeps the same methods:
# chunker.chunk, embedder.fit/embed/embed_batch, store.add/search/clear,
# prompt_builder.build, generator.generate.
class RAGPipeline:
    def __init__(self, chunk_size=200, overlap=50, top_k=5,
                 chunker=None, embedder=None, store=None,
                 prompt_builder=None, generator=None):
        self.top_k = top_k
        self.chunker = chunker or TextChunker(chunk_size, overlap)
        self.embedder = embedder or TfidfEmbedder()
        self.store = store or VectorStore()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.generator = generator or ExtractiveGenerator()

    def index(self, documents, source_names=None):
        all_chunks = []
        all_sources = []
        for i, doc in enumerate(documents):
            doc_chunks = self.chunker.chunk(doc)
            all_chunks.extend(doc_chunks)
            name = source_names[i] if source_names else f"doc_{i}"
            all_sources.extend([name] * len(doc_chunks))

        self.embedder.fit(all_chunks)
        self.store.clear()
        embeddings = self.embedder.embed_batch(all_chunks)
        for embedding, chunk, source in zip(embeddings, all_chunks, all_sources):
            self.store.add(embedding, chunk, {"source": source})
        return len(all_chunks)

    def retrieve(self, question, top_k=None):
        k = top_k or self.top_k
        query_emb = self.embedder.embed(question)
        return [
            {
                "chunk": hit["chunk"],
                "source": hit["metadata"].get("source", "unknown"),
                "score": hit["score"],
                "index": hit["index"],
            }
            for hit in self.store.search(query_emb, k)
        ]

    def query(self, question, top_k=None):
        retrieved = self.retrieve(question, top_k)
        chunk_texts = [r["chunk"] for r in retrieved]
        prompt = self.prompt_builder.build(question, chunk_texts)
        answer = self.generator.generate(prompt, chunk_texts)

        return {
            "question": question,
            "answer": answer,
            "prompt": prompt,
            "retrieved": retrieved
        }
