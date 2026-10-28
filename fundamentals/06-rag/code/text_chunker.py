class TextChunker:
    def __init__(self, chunk_size=200, overlap=50):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text):
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunks.append(" ".join(words[start:end]))
            start += self.chunk_size - self.overlap
        return chunks
