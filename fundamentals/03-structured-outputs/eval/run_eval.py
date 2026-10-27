"""Run the product-extraction eval and report exact match, field accuracy and type compliance.

    python eval/run_eval.py                    # live run against the API
    python eval/run_eval.py --model claude-sonnet-5-5
    python eval/run_eval.py --selftest         # offline: check the scorer itself, no API calls
"""
import argparse
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

sys.path.insert(0, str(Path(__file__).parent))
from extractor import DEFAULT_MODEL, FIELDS, Product, extract  # noqa: E402

HERE = Path(__file__).parent
DATASET = HERE / "dataset.jsonl"
RESULTS_DIR = HERE / "results"

ADAPTERS = {f: TypeAdapter(Product.model_fields[f].annotation) for f in FIELDS}


# ---------------------------------------------------------------- dataset

def load_dataset(path: Path = DATASET) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate ids"
    for r in rows:
        # Gold labels must themselves satisfy the schema and the format rules.
        Product.model_validate(r["expected"], strict=True)
        bad = format_errors(r["expected"])
        assert not bad, f"{r['id']}: gold label breaks format rules: {bad}"
    return rows


# ---------------------------------------------------------------- compliance

def _is_real_date(s: str) -> bool:
    try:
        date.fromisoformat(s)
        return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", s))
    except ValueError:
        return False


def format_errors(d: dict) -> list[str]:
    """Semantic format rules beyond JSON types (the schema cannot express these)."""
    errs = []
    if isinstance(d.get("release_date"), str) and not _is_real_date(d["release_date"]):
        errs.append("release_date")
    if isinstance(d.get("currency"), str) and not re.fullmatch(r"[A-Z]{3}", d["currency"]):
        errs.append("currency")
    if isinstance(d.get("rating"), (int, float)) and not 0 <= d["rating"] <= 5:
        errs.append("rating")
    for f in ("price", "weight_grams", "warranty_months"):
        if isinstance(d.get(f), (int, float)) and d[f] < 0:
            errs.append(f)
    if isinstance(d.get("colors"), list) and any(isinstance(c, str) and c != c.lower() for c in d["colors"]):
        errs.append("colors")
    return errs


def check_compliance(raw: str) -> tuple[dict | None, dict]:
    """Returns (parsed dict or None, compliance info)."""
    info = {"parsed": False, "type_ok": {f: False for f in FIELDS}, "extra_keys": [],
            "all_types_ok": False, "format_errors": []}
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None, info
    if not isinstance(data, dict):
        return None, info
    info["parsed"] = True
    info["extra_keys"] = sorted(set(data) - set(FIELDS))
    for f in FIELDS:
        if f in data:
            try:
                ADAPTERS[f].validate_python(data[f], strict=True)
                info["type_ok"][f] = True
            except ValidationError:
                pass
    info["all_types_ok"] = all(info["type_ok"].values()) and not info["extra_keys"]
    info["format_errors"] = format_errors(data)
    return data, info


# ---------------------------------------------------------------- scoring

def _norm_str(s):
    return re.sub(r"\s+", " ", s).strip().strip(".,;:").casefold() if isinstance(s, str) else s


def _empty(v):
    return v is None or v == []


def field_equal(field: str, exp, got) -> bool:
    if _empty(exp) or _empty(got):
        return _empty(exp) and _empty(got)
    if field in ("name", "brand"):
        return _norm_str(exp) == _norm_str(got)
    if field == "colors":
        if not isinstance(got, list):
            return False
        return {_norm_str(c) for c in exp} == {_norm_str(c) for c in got}
    if field == "price":
        return isinstance(got, (int, float)) and not isinstance(got, bool) and abs(exp - got) < 0.005
    if field == "rating":
        return isinstance(got, (int, float)) and not isinstance(got, bool) and abs(exp - got) < 0.05
    return exp == got and type(exp) is type(got)


def classify(field: str, exp, got) -> str:
    """correct | hallucinated (expected empty, got value) | missed (expected value, got empty) | wrong."""
    if field_equal(field, exp, got):
        return "correct"
    if _empty(exp):
        return "hallucinated"
    if _empty(got):
        return "missed"
    return "wrong"


def score_example(expected: dict, pred: dict | None) -> dict:
    pred = pred or {}
    outcome = {f: classify(f, expected[f], pred.get(f)) for f in FIELDS}
    return {"outcome": outcome, "exact": all(v == "correct" for v in outcome.values())}


# ---------------------------------------------------------------- aggregation / report

def aggregate(records: list[dict]) -> dict:
    n = len(records)
    per_field = {}
    for f in FIELDS:
        c = Counter(r["score"]["outcome"][f] for r in records)
        per_field[f] = {
            "accuracy": c["correct"] / n,
            "wrong": c["wrong"], "hallucinated": c["hallucinated"], "missed": c["missed"],
            "type_compliance": sum(r["compliance"]["type_ok"][f] for r in records) / n,
        }
    total_fields = n * len(FIELDS)
    return {
        "n": n,
        "exact_match": sum(r["score"]["exact"] for r in records) / n,
        "field_accuracy": sum(v["accuracy"] for v in per_field.values()) / len(FIELDS),
        "parse_rate": sum(r["compliance"]["parsed"] for r in records) / n,
        "type_compliance_example": sum(r["compliance"]["all_types_ok"] for r in records) / n,
        "type_compliance_field": sum(sum(r["compliance"]["type_ok"].values()) for r in records) / total_fields,
        "format_compliance_example": sum(not r["compliance"]["format_errors"] for r in records if r["compliance"]["parsed"])
        / max(1, sum(r["compliance"]["parsed"] for r in records)),
        "per_field": per_field,
        "errors_by_count": Counter(sum(v != "correct" for v in r["score"]["outcome"].values()) for r in records),
    }


def render_report(agg: dict, records: list[dict], model: str) -> str:
    pct = lambda x: f"{x * 100:5.1f}%"  # noqa: E731
    L = [f"# Extraction eval - {model}", "",
         f"Examples: {agg['n']}  |  Fields per example: {len(FIELDS)}", "",
         "## Headline", "",
         f"- Exact match (all {len(FIELDS)} fields correct): **{pct(agg['exact_match'])}**",
         f"- Field-level accuracy (mean over fields): **{pct(agg['field_accuracy'])}**",
         f"- JSON parse rate: {pct(agg['parse_rate'])}",
         f"- Type compliance, per example (all fields strictly typed, no extra keys): **{pct(agg['type_compliance_example'])}**",
         f"- Type compliance, per field: {pct(agg['type_compliance_field'])}",
         f"- Format compliance (date/currency/range/lowercase rules): {pct(agg['format_compliance_example'])}",
         "", "## Per-field accuracy (hardest first)", "",
         "| field | accuracy | wrong | hallucinated | missed | type compliance |",
         "|---|---|---|---|---|---|"]
    for f, v in sorted(agg["per_field"].items(), key=lambda kv: kv[1]["accuracy"]):
        L.append(f"| {f} | {pct(v['accuracy'])} | {v['wrong']} | {v['hallucinated']} | {v['missed']} | {pct(v['type_compliance'])} |")
    L += ["", "wrong = value present but incorrect; hallucinated = expected null/[] but model gave a value; "
          "missed = expected a value but model gave null/[].", "",
          "## Wrong fields per example", ""]
    for k in sorted(agg["errors_by_count"]):
        L.append(f"- {k} wrong field(s): {agg['errors_by_count'][k]} examples")

    L += ["", "## Failures by field", ""]
    ranked = sorted(FIELDS, key=lambda f: agg["per_field"][f]["accuracy"])
    for f in ranked:
        fails = [r for r in records if r["score"]["outcome"][f] != "correct"]
        if not fails:
            continue
        L += [f"### {f} ({len(fails)} failures)", ""]
        for r in fails:
            L.append(f"- `{r['id']}` [{r['score']['outcome'][f]}] expected `{json.dumps(r['expected'][f], ensure_ascii=False)}` "
                     f"got `{json.dumps((r['pred'] or {}).get(f, '<missing>'), ensure_ascii=False)}`  _(tags: {', '.join(r['tags'])})_")
        L.append("")

    tag_stats: dict[str, list[bool]] = {}
    for r in records:
        for t in r["tags"]:
            tag_stats.setdefault(t, []).append(r["score"]["exact"])
    L += ["## Exact match by trap tag (tags with >=2 examples, worst first)", "",
          "| tag | n | exact match |", "|---|---|---|"]
    for t, v in sorted(tag_stats.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        if len(v) >= 2:
            L.append(f"| {t} | {len(v)} | {pct(sum(v) / len(v))} |")
    cal = calibration(records)
    if cal:
        L += render_calibration(cal)
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- runners

def evaluate(rows: list[dict], predict) -> list[dict]:
    """predict(row) -> dict with 'raw' (+ optional metadata)."""
    records = []
    for row, out in zip(rows, predict(rows)):
        pred, comp = check_compliance(out.get("raw", ""))
        rec = {
            "id": row["id"], "tags": row["tags"], "text": row["text"], "expected": row["expected"],
            "pred": pred, "raw": out.get("raw", ""), "error": out.get("error") or next(iter(out.get("errors") or []), None),
            "stop_reason": out.get("stop_reason"),
            "input_tokens": out.get("input_tokens"), "output_tokens": out.get("output_tokens"),
            "compliance": comp, "score": score_example(row["expected"], pred),
        }
        if "confidence" in out:
            first = _load_dict(out["samples"][0]) if out.get("samples") else None
            rec.update(confidence=out["confidence"], flagged=out["flagged"], samples=out.get("samples"),
                       single_run_score=score_example(row["expected"], first))
        records.append(rec)
    return records


def _load_dict(raw: str) -> dict | None:
    try:
        d = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return d if isinstance(d, dict) else None


def calibration(records: list[dict]) -> dict | None:
    """How well does confidence predict correctness? Only for records produced with --samples > 1."""
    recs = [r for r in records if "confidence" in r]
    if not recs:
        return None
    n = len(recs)
    by_level: dict[float, list[bool]] = {}
    per_field = {f: {"flagged": 0, "errors": 0, "caught": 0} for f in FIELDS}
    for r in recs:
        for f in FIELDS:
            ok = r["score"]["outcome"][f] == "correct"
            flagged = f in r["flagged"]
            by_level.setdefault(round(r["confidence"][f], 2), []).append(ok)
            pf = per_field[f]
            pf["flagged"] += flagged
            pf["errors"] += not ok
            pf["caught"] += flagged and not ok
    flagged_n = sum(v["flagged"] for v in per_field.values())
    errors_n = sum(v["errors"] for v in per_field.values())
    caught_n = sum(v["caught"] for v in per_field.values())
    clean = [r for r in recs if not r["flagged"]]
    return {
        "n": n,
        "by_level": {k: (len(v), sum(v) / len(v)) for k, v in sorted(by_level.items())},
        "per_field": per_field,
        "flagged_fields": flagged_n, "error_fields": errors_n, "caught_errors": caught_n,
        "flag_precision": caught_n / flagged_n if flagged_n else None,
        "error_recall": caught_n / errors_n if errors_n else None,
        "examples_needing_review": n - len(clean),
        "clean_examples": len(clean),
        "clean_exact_match": sum(r["score"]["exact"] for r in clean) / len(clean) if clean else None,
        "consensus_exact": sum(r["score"]["exact"] for r in recs) / n,
        "single_exact": sum(r["single_run_score"]["exact"] for r in recs) / n,
        "consensus_field_acc": sum(sum(v == "correct" for v in r["score"]["outcome"].values()) for r in recs) / (n * len(FIELDS)),
        "single_field_acc": sum(sum(v == "correct" for v in r["single_run_score"]["outcome"].values()) for r in recs) / (n * len(FIELDS)),
    }


def render_calibration(cal: dict) -> list[str]:
    pct = lambda x: "n/a" if x is None else f"{x * 100:5.1f}%"  # noqa: E731
    L = ["", "## Confidence and human review", "",
         "Confidence = share of the runs that agree with the consensus value. Fields below the review threshold are flagged.", "",
         "| confidence | fields | accuracy |", "|---|---|---|"]
    for lvl, (cnt, acc) in cal["by_level"].items():
        L.append(f"| {lvl:.2f} | {cnt} | {pct(acc)} |")
    L += ["",
          f"- Flagged {cal['flagged_fields']} of {cal['n'] * len(FIELDS)} fields; they contain {cal['caught_errors']} of the {cal['error_fields']} errors.",
          f"- Flag precision (flagged fields that were actually wrong): {pct(cal['flag_precision'])}",
          f"- Error recall (wrong fields that got flagged): {pct(cal['error_recall'])}",
          f"- Examples needing review: {cal['examples_needing_review']} of {cal['n']}; "
          f"the {cal['clean_examples']} unflagged examples are exact-match {pct(cal['clean_exact_match'])}",
          f"- Consensus vs a single run: exact {pct(cal['consensus_exact'])} vs {pct(cal['single_exact'])}, "
          f"field accuracy {pct(cal['consensus_field_acc'])} vs {pct(cal['single_field_acc'])}",
          "", "| field | flagged | errors | errors flagged |", "|---|---|---|---|"]
    for f, v in sorted(cal["per_field"].items(), key=lambda kv: -kv[1]["errors"]):
        if v["flagged"] or v["errors"]:
            L.append(f"| {f} | {v['flagged']} | {v['errors']} | {v['caught']} |")
    return L


def live_predictor(model: str, workers: int, samples: int = 1, min_confidence: float = 1.0):
    import anthropic
    from confidence import extract_with_confidence
    client = anthropic.Anthropic()

    def one(row):
        try:
            if samples > 1:
                return extract_with_confidence(row["text"], n=samples, model=model, client=client,
                                               min_confidence=min_confidence)
            return extract(row["text"], model=model, client=client)
        except Exception as e:  # record and keep going; counted as a parse failure
            return {"raw": "", "error": f"{type(e).__name__}: {e}"}

    def predict(rows):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(one, rows))
    return predict


def selftest() -> None:
    rows = load_dataset()
    assert len(rows) == 50, len(rows)

    # 1. Gold as prediction -> perfect scores.
    recs = evaluate(rows, lambda rs: [{"raw": json.dumps(r["expected"])} for r in rs])
    agg = aggregate(recs)
    assert agg["exact_match"] == 1 and agg["field_accuracy"] == 1 and agg["type_compliance_example"] == 1, agg

    # 2. Known corruptions are caught and classified.
    def corrupt(r):
        d = json.loads(json.dumps(r["expected"]))
        if r["id"] == "p01":
            d["price"] = 349.0                       # wrong
        if r["id"] == "p07":
            d["brand"] = "Acme"                      # hallucinated
        if r["id"] == "p08":
            d["weight_grams"] = None                 # missed
        if r["id"] == "p09":
            d["weight_grams"] = "272"                # type violation (and wrong)
        if r["id"] == "p10":
            d["release_date"] = "June 5, 2025"       # format violation (and wrong)
        if r["id"] == "p11":
            d["extra"] = 1                           # extra key
        if r["id"] == "p12":
            d["colors"] = ["AQUA", "Black"]          # case/order-insensitive match, but format-lowercase violation
        if r["id"] == "p13":
            return {"raw": "not json"}
        return {"raw": json.dumps(d)}
    recs = evaluate(rows, lambda rs: [corrupt(r) for r in rs])
    by = {r["id"]: r for r in recs}
    assert by["p01"]["score"]["outcome"]["price"] == "wrong"
    assert by["p07"]["score"]["outcome"]["brand"] == "hallucinated"
    assert by["p08"]["score"]["outcome"]["weight_grams"] == "missed"
    assert not by["p09"]["compliance"]["type_ok"]["weight_grams"]
    assert by["p10"]["compliance"]["format_errors"] == ["release_date"]
    assert by["p11"]["compliance"]["extra_keys"] == ["extra"] and not by["p11"]["compliance"]["all_types_ok"]
    assert by["p12"]["score"]["outcome"]["colors"] == "correct" and by["p12"]["compliance"]["format_errors"] == ["colors"]
    assert not by["p13"]["compliance"]["parsed"] and not by["p13"]["score"]["exact"]
    agg = aggregate(recs)
    assert agg["exact_match"] == 44 / 50, agg["exact_match"]  # p01,p07,p08,p09,p10,p13 wrong; p11,p12 still exact
    assert abs(agg["parse_rate"] - 49 / 50) < 1e-9
    assert abs(agg["type_compliance_example"] - 47 / 50) < 1e-9  # p09, p11, p13
    render_report(agg, recs, "selftest")

    # 3. Confidence aggregation.
    from confidence import aggregate_samples
    g = rows[0]["expected"]
    with_price = lambda p: {**g, "price": p}  # noqa: E731
    a = aggregate_samples([g, g, g], 3)
    assert a["flagged"] == [] and all(c == 1.0 for c in a["confidence"].values())
    a = aggregate_samples([with_price(1.0), with_price(2.0), with_price(2.0)], 3)
    assert a["consensus"]["price"] == 2.0 and abs(a["confidence"]["price"] - 2 / 3) < 1e-9 and a["flagged"] == ["price"]
    a = aggregate_samples([with_price(1.0), with_price(2.0), with_price(3.0)], 3)
    assert a["consensus"]["price"] == 1.0 and abs(a["confidence"]["price"] - 1 / 3) < 1e-9  # tie -> earliest run
    a = aggregate_samples([g, None, g], 3)  # failed run counts as disagreement on every field
    assert abs(a["confidence"]["name"] - 2 / 3) < 1e-9 and len(a["flagged"]) == len(FIELDS)
    assert aggregate_samples([None, None, None], 3)["consensus"] is None
    a = aggregate_samples([{**g, "colors": ["Silver", "black"]}, {**g, "colors": ["black", "silver"]}, g], 3)
    assert a["confidence"]["colors"] == 1.0  # case/order differences are not disagreement
    assert aggregate_samples([g, g, g], 3, min_confidence=0.5)["flagged"] == []

    # 4. Calibration metrics on a synthetic 3-run predictor.
    def fake(r):
        g, id_ = r["expected"], r["id"]
        runs = [g, g, g]
        if id_ == "p01":
            runs = [{**g, "price": 1.0}, g, g]                       # unstable, consensus right
        if id_ == "p02":
            runs = [{**g, "price": 1.0}] * 3                         # confidently wrong: not flagged
        if id_ == "p03":
            runs = [g, {**g, "price": 1.0}, {**g, "price": 2.0}]     # unstable, consensus = first run (right)
        if id_ == "p04":
            runs = [{**g, "price": 1.0}, {**g, "price": 1.0}, g]     # unstable, consensus wrong
        a = aggregate_samples(runs, 3)
        return {"raw": json.dumps(a["consensus"]), "confidence": a["confidence"], "flagged": a["flagged"],
                "samples": [json.dumps(x) for x in runs]}
    recs = evaluate(rows, lambda rs: [fake(r) for r in rs])
    cal = calibration(recs)
    assert cal["flagged_fields"] == 3 and cal["error_fields"] == 2 and cal["caught_errors"] == 1, cal
    assert cal["per_field"]["price"] == {"flagged": 3, "errors": 2, "caught": 1}
    assert cal["examples_needing_review"] == 3 and cal["clean_examples"] == 47
    assert cal["clean_exact_match"] == 46 / 47  # p02 is a confident error
    assert cal["consensus_exact"] == 48 / 50 and cal["single_exact"] == 47 / 50  # first runs: p01,p02,p04 wrong... p03 right
    render_report(aggregate(recs), recs, "selftest")
    print("selftest OK: 50 gold labels valid, scorer and confidence logic behave as specified")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--limit", type=int, help="only the first N examples (smoke test)")
    ap.add_argument("--samples", type=int, default=1,
                    help="runs per example; >1 enables consistency-based confidence and review flags")
    ap.add_argument("--min-confidence", type=float, default=1.0,
                    help="flag fields whose agreement is below this (1.0 = flag any disagreement)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    rows = load_dataset()[: args.limit]
    records = evaluate(rows, live_predictor(args.model, args.workers, args.samples, args.min_confidence))
    agg = aggregate(records)
    report = render_report(agg, records, args.model)

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = RESULTS_DIR / f"{args.model}-{stamp}"
    stem.with_suffix(".json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    stem.with_suffix(".md").write_text(report, encoding="utf-8")
    api_errors = [r for r in records if r["error"]]
    print(report)
    print(f"Saved {stem.with_suffix('.md')} and .json" + (f"  ({len(api_errors)} API errors)" if api_errors else ""))


if __name__ == "__main__":
    main()
