from extractive_generator import ExtractiveGenerator


def _claude_generator():
    from claude_generator import ClaudeGenerator  # imported lazily so the demo runs without the anthropic package
    return ClaudeGenerator()


GENERATORS = {
    "1": ("extractive", ExtractiveGenerator),
    "2": ("claude", _claude_generator),
}


def choose_generator(choice=None):
    if choice is None:
        print("Choose a generator:")
        print("  [1] extractive - picks the best-matching sentence (default)")
        print("  [2] claude     - Claude Haiku 5.5, needs ANTHROPIC_API_KEY")
        choice = input("Generator [1]: ").strip() or "1"
    for key, (name, factory) in GENERATORS.items():
        if choice.lower() in (key, name):
            return name, factory
    raise SystemExit(f"Unknown generator: {choice!r}. Use 1/2 or extractive/claude.")
