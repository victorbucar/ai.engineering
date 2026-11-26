import re

# Cheapest model that can handle each complexity level, per subscription tier.
# Free users never reach the most expensive model. Claude-only, because these are the
# models the pipeline actually calls through the Anthropic SDK.
TIER_MODELS = {
    "free": {"simple": "claude-haiku-5-5", "moderate": "claude-haiku-5-5", "complex": "claude-sonnet-5-5"},
    "pro": {"simple": "claude-haiku-5-5", "moderate": "claude-sonnet-5-5", "complex": "claude-opus-5-5"},
    "enterprise": {"simple": "claude-sonnet-5-5", "moderate": "claude-sonnet-5-5", "complex": "claude-opus-5-5"},
}

COMPLEX_KEYWORDS = [
    "analyze", "analyse", "trade-offs", "tradeoffs", "pros and cons", "architecture",
    "design", "write code", "implement", "debug", "algorithm", "prove", "optimize",
    "function", "compare",
]
MODERATE_KEYWORDS = [
    "summarize", "summarise", "explain", "difference", "describe", "translate",
    "rewrite", "list", "report", "vs",
]


def _has_keyword(text, keywords):
    return any(re.search(rf"\b{re.escape(k)}\b", text) for k in keywords)


def classify_complexity(query):
    text = query.lower()
    words = len(text.split())
    if _has_keyword(text, COMPLEX_KEYWORDS) or words > 60:
        return "complex"
    if _has_keyword(text, MODERATE_KEYWORDS) or words > 20:
        return "moderate"
    return "simple"


def route_model(query, tier="free"):
    complexity = classify_complexity(query)
    models = TIER_MODELS.get(tier, TIER_MODELS["free"])
    return {"model": models[complexity], "complexity": complexity, "tier": tier}
