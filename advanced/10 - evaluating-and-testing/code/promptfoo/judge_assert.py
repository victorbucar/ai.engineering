"""Deterministic promptfoo metric: ROUGE-L.

The four quality criteria are graded by Claude Haiku 5.5 through llm-rubric
(see promptfooconfig.yaml); this file only adds ROUGE-L against
vars.reference_output as an extra, non-gating metric.
"""


def _rouge_l(reference, hypothesis):
    if not reference or not hypothesis:
        return 0.0
    ref, hyp = reference.lower().split(), hypothesis.lower().split()
    dp = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if ref[i - 1] == hyp[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(hyp), lcs / len(ref)
    return round(2 * p * r / (p + r), 4)


def rouge_l(output, context):
    score = _rouge_l(context.get("vars", {}).get("reference_output", ""), output)
    return {"pass": True, "score": score, "reason": f"ROUGE-L={score:.4f}",
            "namedScores": {"rouge_l": score}}
