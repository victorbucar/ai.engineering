from dataclasses import dataclass, field


MODEL_PRICING = {
    "gpt-6-astra": {"input": 10.00, "output": 50.00, "cached_input": 1.00},
    "gpt-6-luna": {"input": 0.10, "output": 0.50, "cached_input": 0.01},
    "gpt-6-sol": {"input": 2.00, "output": 10.00, "cached_input": 0.20},
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cached_input": 0.20},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cached_input": 0.10},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cached_input": 0.01},
    # Possible server-side refusal-fallback targets: if a fallback answers, response.model
    # names it and the call is billed at its rates. Cache reads are 0.1x input.
    "claude-opus-5": {"input": 5.00, "output": 25.00, "cached_input": 0.50},
    "claude-opus-4-8": {"input": 5.00, "output": 25.00, "cached_input": 0.50},
    "claude-sonnet-5": {"input": 2.00, "output": 10.00, "cached_input": 0.20},
}

# Writing a prefix into the provider's prompt cache (5-minute TTL) costs 1.25x the input price.
CACHE_WRITE_MULTIPLIER = 1.25


def calculate_cost(model, input_tokens, output_tokens, cached_input_tokens=0, cache_write_tokens=0):
    """input_tokens is the total prompt size; cached_input_tokens (read from cache) and
    cache_write_tokens (written to cache) are the parts of it billed at different rates."""
    if model not in MODEL_PRICING:
        raise ValueError(f"Unknown model: {model!r}. Known models: {', '.join(MODEL_PRICING)}")
    pricing = MODEL_PRICING[model]
    non_cached = input_tokens - cached_input_tokens - cache_write_tokens
    input_cost = (non_cached / 1_000_000) * pricing["input"]
    cached_cost = (cached_input_tokens / 1_000_000) * pricing["cached_input"]
    cache_write_cost = (cache_write_tokens / 1_000_000) * pricing["input"] * CACHE_WRITE_MULTIPLIER
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    total = input_cost + cached_cost + cache_write_cost + output_cost
    return {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": cached_input_tokens,
        "input_cost": round(input_cost, 6),
        "cache_write_tokens": cache_write_tokens,
        "cached_input_cost": round(cached_cost, 6),
        "cache_write_cost": round(cache_write_cost, 6),
        "output_cost": round(output_cost, 6),
        "total_cost": round(total, 6),
    }