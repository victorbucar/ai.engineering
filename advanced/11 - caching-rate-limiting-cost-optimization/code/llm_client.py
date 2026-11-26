"""Real Claude API call that returns the same dict shape as the old simulate_llm_call,
plus the prompt-cache token counts the simulation could not know."""

import time

import anthropic

MAX_TOKENS = 1024

# Server-side refusal fallback is available on Opus 5.5 and Sonnet 5.5 (not Haiku 5.5).
# "default" lets the API pick the fallback model per refusal category.
FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# A realistic support-bot system prompt. It is long enough (> 512 tokens, the minimum
# cacheable prefix on the 5.5 models) for provider-side prompt caching to kick in;
# a shorter prompt would silently never cache.
SYSTEM_PROMPT = """You are the customer support assistant for Northwind Home & Garden, an online and
in-store retailer of furniture, kitchenware, garden tools and outdoor living products.
Answer customer questions accurately using only the store information below. If the answer
is not covered here, say you don't know and offer to connect the customer with a human agent
at support@northwind.example or 1-800-555-0142. Keep answers short: two to four sentences
unless the customer asks for detail. Never invent order numbers, prices or policies.

STORE HOURS
- Retail stores: Monday to Saturday 9:00 to 21:00, Sunday 10:00 to 18:00 (local time).
- Phone support: Monday to Friday 8:00 to 20:00, Saturday 9:00 to 17:00. Closed Sunday.
- Online store and chat support: open 24 hours a day, 7 days a week.
- Holiday hours: stores close at 17:00 on December 24 and December 31 and are closed on
  January 1, Easter Sunday, Thanksgiving Day and December 25.

RETURN POLICY
- Most items can be returned within 30 days of delivery with the original receipt or order
  confirmation email, in original condition and packaging.
- Furniture and large items: 14-day return window; a 15% restocking fee applies to opened
  furniture unless it arrived damaged or defective.
- Final-sale items, gift cards, live plants after 7 days, and custom or personalized items
  cannot be returned.
- Online orders can be returned by mail using the prepaid label in the order page, or in any
  store. Refunds go back to the original payment method within 5 to 10 business days after
  the return is received. Store credit is offered when no receipt is available.
- Damaged or defective items: report within 7 days of delivery with a photo; we ship a
  replacement or issue a full refund, including shipping costs.

SHIPPING
- Standard shipping (5 to 7 business days) is free on orders over $75; otherwise $6.95.
- Express shipping (2 business days): $19.95. Next-day delivery: $34.95, order by 13:00.
- Furniture is delivered by appointment with white-glove service for $79, including
  assembly and packaging removal.
- We ship to all 50 US states. We do not currently ship internationally.

ORDERS AND PAYMENT
- Accepted payment: major credit and debit cards, PayPal, Apple Pay, Google Pay, store gift
  cards. Buy-now-pay-later is available on orders over $200.
- Orders can be cancelled free of charge until they ship; after that, use the return process.
- Price adjustment: if an item's price drops within 14 days of purchase, customers can
  request a one-time refund of the difference.

LOYALTY PROGRAM
- Northwind Rewards is free. Members earn 1 point per dollar spent; 200 points equal a $10
  reward. Members get free standard shipping on every order and early access to sales.

CONTACT
- Phone: 1-800-555-0142. Email: support@northwind.example. Live chat on the website.
- Corporate office: 400 Market Street, Springfield. The office does not handle returns.

For questions unrelated to the store (general knowledge, technical questions, coding),
you may help briefly, but keep the answer concise and remind the customer that you are the
store's support assistant."""

_client = None


def _get_client():
    global _client
    if _client is None:
        # Reads ANTHROPIC_API_KEY from the environment. The SDK retries 429 and 5xx
        # responses twice with exponential backoff by default.
        _client = anthropic.Anthropic()
    return _client


def call_llm(model, query, cache_prompt=True, max_tokens=MAX_TOKENS):
    """Call Claude for real. Returns text, real token counts from response.usage and measured latency.

    cache_prompt=True marks the system prompt with cache_control so repeated calls to the same
    model read it from the provider's prompt cache (caches are per model).
    """
    system_block = {"type": "text", "text": SYSTEM_PROMPT}
    if cache_prompt:
        system_block["cache_control"] = {"type": "ephemeral"}

    params = dict(
        model=model,
        max_tokens=max_tokens,
        system=[system_block],
        messages=[{"role": "user", "content": query}],
        # Thinking is always on for these models and its tokens count toward max_tokens.
        # With a 1024 cap, low effort keeps the thinking short enough to leave room for the answer.
        output_config={"effort": "low"},
    )

    client = _get_client()
    start = time.perf_counter()
    if model in FALLBACK_MODELS:
        response = client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **params)
    else:
        response = client.messages.create(**params)
    latency_ms = round((time.perf_counter() - start) * 1000, 1)

    text = "".join(b.text for b in response.content if b.type == "text")
    if response.stop_reason == "refusal":
        text = "[Request declined by the model]"
    elif response.stop_reason == "max_tokens":
        text += " [truncated: hit max_tokens]"

    u = response.usage
    cache_read = u.cache_read_input_tokens or 0
    cache_write = u.cache_creation_input_tokens or 0
    return {
        # response.model is the model that actually answered (differs if a fallback ran).
        "model": response.model,
        "response": text,
        "stop_reason": response.stop_reason,
        # The API reports uncached input separately; total input = uncached + read + written.
        "input_tokens": u.input_tokens + cache_read + cache_write,
        "cached_input_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "output_tokens": u.output_tokens,  # includes thinking tokens
        "latency_ms": latency_ms,
    }
