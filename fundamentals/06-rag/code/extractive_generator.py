# Stand-in for an LLM: returns the retrieved sentence sharing the most words with the question.
class ExtractiveGenerator:
    STOP_WORDS = {"the", "a", "an", "is", "are", "was", "were", "what", "how",
                  "why", "when", "where", "do", "does", "for", "of", "in", "to",
                  "and", "or", "on", "at", "by", "it", "its", "this", "that"}

    def __init__(self, min_sentence_length=10, fallback="I don't have enough information."):
        self.min_sentence_length = min_sentence_length
        self.fallback = fallback

    def generate(self, prompt, retrieved_chunks):
        query_section = prompt.lower().split("question:")[-1]
        query_words = set(query_section.split()) - self.STOP_WORDS

        best_sentence = ""
        best_score = 0

        for chunk in retrieved_chunks:
            for sentence in chunk.split("."):
                sentence = sentence.strip()
                if len(sentence) < self.min_sentence_length:
                    continue
                words = set(sentence.lower().split())
                overlap = len(query_words & words)
                if overlap > best_score:
                    best_score = overlap
                    best_sentence = sentence

        return best_sentence if best_sentence else self.fallback
