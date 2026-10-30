import anthropic


class ClaudeGenerator:
    name = "claude"
    SYSTEM_PROMPT = (
        "You are a retrieval-augmented assistant. Follow the instructions in the "
        "user's message exactly and keep answers short: one to three sentences."
    )

    def __init__(self, model="claude-haiku-5-5", effort="medium", max_tokens=16000,
                 fallback="I don't have enough information."):
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.fallback = fallback

    # retrieved_chunks is unused: the prompt from PromptBuilder already contains them
    def generate(self, prompt, retrieved_chunks):
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": self.effort},
            # if a safety classifier declines, the API retries on a fallback model
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            return self.fallback
        text = "".join(block.text for block in response.content if block.type == "text")
        return text.strip() or self.fallback
