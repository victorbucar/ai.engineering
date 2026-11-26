from cost_calculator import *
from cost_tracker import *
from exact_cache import *
from semantic_cache import *
from rate_limiter import *
from model_router import *
import sys
import time

# Real Claude API calls by default (needs ANTHROPIC_API_KEY, costs a few cents per run).
# Pass --simulate to use made-up token counts instead.
SIMULATE = "--simulate" in sys.argv
BASELINE_MODEL = "claude-sonnet-5-5"


def simulate_llm_call(model, query, cache_prompt=True):
    input_tokens = len(query.split()) * 4 + 500
    output_tokens = 150 + (len(query.split()) * 2)
    latency = 200 + (output_tokens * 2)
    return {
        "model": model,
        "response": f"[Simulated {model} response to: {query[:50]}...]",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": latency,
    }


def run_demo():
    print("=" * 60)
    print("  Caching, Rate Limiting & Cost Optimization Demo")
    print("=" * 60)

    print("\n--- Model Pricing ---")
    for model, pricing in list(MODEL_PRICING.items())[:6]:
        cost_1k = calculate_cost(model, 1000, 500)
        print(f"  {model}: ${cost_1k['total_cost']:.6f} per 1K in + 500 out")

    print("\n--- Cost Comparison: 100K Requests ---")
    for model in ["gpt-6-sol", "gpt-6-luna", "claude-sonnet-5-5", "claude-haiku-5-5"]:
        cost = calculate_cost(model, 1000 * 100_000, 500 * 100_000)
        print(f"  {model}: ${cost['total_cost']:.2f}")

    print("\n--- Anthropic Cache Savings ---")
    no_cache = calculate_cost("claude-sonnet-5-5", 2000, 500, 0)
    with_cache = calculate_cost("claude-sonnet-5-5", 2000, 500, 1500)
    saving = no_cache["total_cost"] - with_cache["total_cost"]
    print(f"  Without cache: ${no_cache['total_cost']:.6f}")
    print(f"  With 1500 cached tokens: ${with_cache['total_cost']:.6f}")
    print(f"  Savings per call: ${saving:.6f} ({saving/no_cache['total_cost']*100:.1f}%)")

    exact_cache = ExactCache(max_size=100, ttl_seconds=300)
    semantic_cache = SemanticCache(max_size=100)
    rate_limiter = TokenBucketRateLimiter()
    tracker = CostTracker(monthly_budget=100.0)

    print("\n--- Exact Cache ---")
    messages_1 = [{"role": "user", "content": "What is the return policy?"}]
    result = exact_cache.get("gpt-6-luna", messages_1, 0.0)
    print(f"  First lookup: {'HIT' if result else 'MISS'}")
    exact_cache.put("gpt-6-luna", messages_1, 0.0, "You can return items within 30 days.")
    result = exact_cache.get("gpt-6-luna", messages_1, 0.0)
    print(f"  Second lookup: {'HIT' if result else 'MISS'} -> {result}")
    result = exact_cache.get("gpt-6-luna", messages_1, 0.7)
    print(f"  With temp=0.7: {'HIT' if result else 'MISS (non-deterministic, skip cache)'}")
    print(f"  Stats: {exact_cache.stats()}")

    print("\n--- Semantic Cache ---")
    test_queries = [
        ("What is the return policy?", "Items can be returned within 30 days with receipt."),
        ("How do I return an item?", None),
        ("What are your store hours?", "We are open 9am-9pm Monday through Saturday."),
        ("When does the store open?", None),
        ("Tell me about quantum computing", "Quantum computers use qubits..."),
        ("Explain quantum mechanics", None),
    ]
    for query, response in test_queries:
        cached = semantic_cache.get(query)
        if cached:
            print(f"  '{query[:40]}' -> CACHE HIT (sim={cached['similarity']}, original='{cached['original_query'][:40]}')")
        elif response:
            semantic_cache.put(query, response)
            print(f"  '{query[:40]}' -> MISS (stored)")
        else:
            print(f"  '{query[:40]}' -> MISS (no match)")
    print(f"  Stats: {semantic_cache.stats()}")

    print("\n--- Rate Limiting ---")
    for i in range(12):
        check = rate_limiter.check("user_1", 1000, "free")
        if check["allowed"]:
            rate_limiter.consume("user_1", 1000, "free")
        status = "OK" if check["allowed"] else f"BLOCKED ({check['reason']})"
        if i < 5 or not check["allowed"]:
            print(f"  Request {i+1}: {status}")
    print(f"  Usage: {rate_limiter.get_usage('user_1')}")

    print("\n--- Model Routing ---")
    routing_queries = [
        "What time do you close?",
        "Summarize this quarterly earnings report",
        "Analyze the trade-offs between microservices and monoliths",
        "Hello",
        "Write code for a binary search tree with deletion",
    ]
    for q in routing_queries:
        route = route_model(q, "pro")
        print(f"  '{q[:50]}' -> {route['model']} ({route['complexity']})")

    print("\n--- Full Pipeline: Before vs After Optimization ---")
    queries = [
        "What is the return policy?",
        "How do I return something?",
        "What are your hours?",
        "When do you open?",
        "Explain the difference between TCP and UDP",
        "Compare TCP vs UDP protocols",
        "Hello",
        "What is your phone number?",
        "Write a Python function to sort a list",
        "Analyze the pros and cons of serverless architecture",
    ]

    if SIMULATE:
        llm = simulate_llm_call
        print("\n  (SIMULATED calls: token counts and latency are made up)")
    else:
        from llm_client import call_llm, MAX_TOKENS
        llm = call_llm
        print(f"\n  (REAL Claude API calls, max_tokens={MAX_TOKENS})")

    def log_result(tracker, result, cache_status="miss"):
        tracker.log_call(
            result["model"], result["input_tokens"], result["output_tokens"],
            cached_input_tokens=result.get("cached_input_tokens", 0),
            cache_write_tokens=result.get("cache_write_tokens", 0),
            latency_ms=result["latency_ms"], cache_status=cache_status,
        )

    def show(q, result=None, label=None, latency_ms=None):
        if result is None:
            print(f"    {label:<17} {latency_ms:>7.1f}ms  {q[:45]}")
            return
        cache_info = ""
        if result.get("cached_input_tokens") or result.get("cache_write_tokens"):
            cache_info = f" (cache read {result['cached_input_tokens']}, write {result['cache_write_tokens']})"
        print(f"    {result['model']:<17} {result['latency_ms']:>7.1f}ms  in {result['input_tokens']:>4}{cache_info} / out {result['output_tokens']:>4}  {q[:45]}")

    print(f"\n  [Before: no caching, single model ({BASELINE_MODEL})]")
    tracker_before = CostTracker(monthly_budget=1000.0)
    for q in queries:
        result = llm(BASELINE_MODEL, q, cache_prompt=False)
        log_result(tracker_before, result)
        show(q, result)
    before = tracker_before.summary()
    print(f"  Total cost: ${before['total_cost']:.6f}")
    print(f"  Avg cost/call: ${before['avg_cost_per_call']:.6f}")
    print(f"  Avg latency: {before['avg_latency_ms']}ms")

    print("\n  [After: exact cache + semantic cache + routing + prompt caching + rate limiting]")
    exact_c = ExactCache()
    semantic_c = SemanticCache()
    limiter = TokenBucketRateLimiter()
    tracker_after = CostTracker(monthly_budget=1000.0)
    user, tier = "demo_user", "pro"

    for q in queries:
        messages = [{"role": "user", "content": q}]
        # Route first: routing is local and cheap, and the exact-cache key includes the model.
        route = route_model(q, tier)

        t0 = time.perf_counter()
        cached = exact_c.get(route["model"], messages, 0.0)
        if cached:
            ms = (time.perf_counter() - t0) * 1000
            tracker_after.log_call(route["model"], 0, 0, latency_ms=ms, cache_status="hit")
            show(q, label="EXACT HIT", latency_ms=ms)
            continue
        sem_cached = semantic_c.get(q)
        if sem_cached:
            ms = (time.perf_counter() - t0) * 1000
            tracker_after.log_call(route["model"], 0, 0, latency_ms=ms, cache_status="hit")
            show(q, label=f"SEMANTIC HIT {sem_cached['similarity']:.2f}", latency_ms=ms)
            continue

        # Pre-call estimate (~4 chars/token for the question, ~800 for the system prompt,
        # worst-case output). The real usage is consumed after the call.
        estimate = len(q) // 4 + 800 + 1024
        check = limiter.check(user, estimate, tier)
        if not check["allowed"]:
            print(f"    RATE LIMITED ({check['reason']}), retry in {check['retry_after_seconds']:.0f}s: {q[:45]}")
            continue

        result = llm(route["model"], q, cache_prompt=True)
        limiter.consume(user, result["input_tokens"] + result["output_tokens"], tier)
        log_result(tracker_after, result)
        show(q, result)
        exact_c.put(route["model"], messages, 0.0, result["response"])
        semantic_c.put(q, result["response"])

    after = tracker_after.summary()
    print(f"  Total cost: ${after['total_cost']:.6f}")
    print(f"  Avg cost/call: ${after['avg_cost_per_call']:.6f}")
    print(f"  Avg latency: {after['avg_latency_ms']}ms")
    print(f"  Cache hit rate: {after['cache_hit_rate']:.0%}")

    if before["total_cost"] > 0:
        savings_pct = (1 - after["total_cost"] / before["total_cost"]) * 100
        print(f"\n  SAVINGS: {savings_pct:.1f}% cost reduction")
        print(f"  Latency improvement: {(1 - after['avg_latency_ms'] / before['avg_latency_ms']) * 100:.1f}% faster")

    print("\n--- Budget Alerts Demo ---")
    alert_tracker = CostTracker(monthly_budget=0.01)
    for i in range(5):
        alert_tracker.log_call("gpt-6-sol", 5000, 2000, latency_ms=500)
    print(f"  Total spent: ${alert_tracker.total_cost():.6f} / ${alert_tracker.monthly_budget}")
    for alert in alert_tracker.alerts:
        print(f"  ALERT [{alert['level'].upper()}]: {alert['message']}")

    print("\n--- Cost Breakdown by Model ---")
    multi_tracker = CostTracker(monthly_budget=500.0)
    for _ in range(50):
        multi_tracker.log_call("gpt-6-luna", 800, 200, latency_ms=150)
    for _ in range(30):
        multi_tracker.log_call("claude-sonnet-5-5", 1500, 500, latency_ms=400)
    for _ in range(10):
        multi_tracker.log_call("gpt-6-sol", 2000, 800, latency_ms=600)
    for _ in range(10):
        multi_tracker.log_call("claude-opus-5-5", 3000, 1000, latency_ms=1200)
    breakdown = multi_tracker.cost_by_model()
    for model, data in sorted(breakdown.items(), key=lambda x: x[1]["cost"], reverse=True):
        print(f"  {model}: {data['calls']} calls, ${data['cost']:.6f}, {data['input_tokens']:,} in / {data['output_tokens']:,} out")
    print(f"  Total: ${multi_tracker.total_cost():.6f}")

    print("\n" + "=" * 60)
    print("  Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()