class PromptBuilder:
    def __init__(self, separator="\n\n---\n\n"):
        self.separator = separator

    def build(self, query, retrieved_chunks):
        context = self.separator.join(
            f"[Source {i+1}]\n{chunk}"
            for i, chunk in enumerate(retrieved_chunks)
        )
        return (
            "Answer the question based ONLY on the following context.\n"
            "If the context doesn't contain enough information, "
            "say \"I don't have enough information to answer that.\"\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
