"""Per-field confidence via self-consistency, with low-confidence fields flagged for human review.

The Claude API does not return token log-probabilities, so confidence is estimated by
running the extraction N times (default sampling, so runs can differ) and measuring
agreement per field:

    confidence(field) = (# runs agreeing with the majority value) / N

The returned value for each field is the majority value (ties go to the earliest run).
A run that fails to parse counts as a disagreement, so API/parse trouble lowers confidence.

Limitation: this catches *unstable* answers, not *consistently wrong* ones. If the model
reliably makes the same mistake, all N runs agree and the field gets confidence 1.0.
"""
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import anthropic

from extractor import DEFAULT_MODEL, FIELDS, extract

DEFAULT_SAMPLES = 3
DEFAULT_MIN_CONFIDENCE = 1.0  # flag any field the runs did not unanimously agree on


def _s(v):
    return re.sub(r"\s+", " ", v).strip().strip(".,;:").casefold() if isinstance(v, str) else v


def canon(field: str, value):
    """Hashable canonical form: values that would be scored as equal compare equal."""
    if value is None or value == []:
        return None
    if field == "colors":
        return frozenset(_s(c) for c in value)
    if field in ("name", "brand"):
        return _s(value)
    if field in ("price", "rating") and isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(float(value), 2)
    return value


def aggregate_samples(parsed: list[dict | None], n: int, min_confidence: float = DEFAULT_MIN_CONFIDENCE) -> dict:
    """Pure function: N parsed runs (None = failed) -> consensus, per-field confidence, flags."""
    valid = [p for p in parsed if p is not None]
    if not valid:
        return {"consensus": None, "confidence": {f: 0.0 for f in FIELDS}, "flagged": list(FIELDS)}
    consensus, confidence = {}, {}
    for f in FIELDS:
        vals = [p.get(f) for p in valid]
        keys = [canon(f, v) for v in vals]
        top_key, top = Counter(keys).most_common(1)[0]  # ties -> first-seen, i.e. earliest run
        consensus[f] = vals[keys.index(top_key)]
        confidence[f] = top / n
    flagged = [f for f in FIELDS if confidence[f] < min_confidence]
    return {"consensus": consensus, "confidence": confidence, "flagged": flagged}


def extract_with_confidence(text: str, n: int = DEFAULT_SAMPLES, model: str = DEFAULT_MODEL,
                            client: anthropic.Anthropic | None = None,
                            min_confidence: float = DEFAULT_MIN_CONFIDENCE) -> dict:
    client = client or anthropic.Anthropic()

    def one(_):
        try:
            return extract(text, model=model, client=client)
        except Exception as e:  # a failed run is a disagreement, not a crash
            return {"raw": "", "error": f"{type(e).__name__}: {e}"}

    with ThreadPoolExecutor(max_workers=n) as pool:
        runs = list(pool.map(one, range(n)))

    parsed = []
    for r in runs:
        try:
            d = json.loads(r["raw"])
            parsed.append(d if isinstance(d, dict) else None)
        except (json.JSONDecodeError, TypeError):
            parsed.append(None)

    agg = aggregate_samples(parsed, n, min_confidence)
    return {
        "raw": json.dumps(agg["consensus"]) if agg["consensus"] is not None else "",
        "confidence": agg["confidence"],
        "flagged": agg["flagged"],
        "samples": [r["raw"] for r in runs],
        "errors": [r["error"] for r in runs if r.get("error")],
        "input_tokens": sum(r.get("input_tokens") or 0 for r in runs),
        "output_tokens": sum(r.get("output_tokens") or 0 for r in runs),
    }


def format_result(res: dict) -> str:
    data = json.loads(res["raw"]) if res["raw"] else {}
    lines = []
    for f in FIELDS:
        flag = "  <-- REVIEW" if f in res["flagged"] else ""
        lines.append(f"  {f:16} {res['confidence'][f]:4.2f}  {json.dumps(data.get(f), ensure_ascii=False)}{flag}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    text = " ".join(sys.argv[1:]) or "Gaming mouse, 12000 DPI, black or white. Ships soon. Rated 8/10."
    print(format_result(extract_with_confidence(text)))
