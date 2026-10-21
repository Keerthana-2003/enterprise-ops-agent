"""Foundation-model selection on AWS Bedrock: run the same evaluation set against several models,
judge every model's answers with one fixed judge, and rank them on quality, latency and cost.

  python -m evals.compare_models \
      --models us.anthropic.claude-3-5-haiku-20241022-v1:0 us.meta.llama3-3-70b-instruct-v1:0 us.amazon.nova-pro-v1:0 \
      --judge-model us.anthropic.claude-3-5-sonnet-20241022-v2:0 --prices prices.json

prices.json maps model id -> {"input": $ per 1M input tokens, "output": $ per 1M output tokens}; use
current Bedrock pricing for your region (prices are inputs, never hard-coded). Writes
evals/results/model_comparison.{json,md}.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

HERE = Path(__file__).parent
DEFAULT_WEIGHTS = {"quality": 0.6, "latency": 0.2, "cost": 0.2}


def cost_per_run(summary: dict, price: dict | None) -> float | None:
    if not price:
        return None
    p_in, p_out = summary.get("avg_prompt_tokens"), summary.get("avg_completion_tokens")
    if p_in is None or p_out is None:
        return None
    return (p_in * price["input"] + p_out * price["output"]) / 1e6


def rank_models(results: dict[str, dict], prices: dict, weights: dict = DEFAULT_WEIGHTS,
                min_success: float = 0.0) -> list[dict]:
    """Score = weighted mix of quality (task success and judge score), speed and cost, each min-max
    normalised across the candidates. Models below `min_success` are listed but not recommended."""
    rows = []
    for model, s in results.items():
        judge = (s.get("judge_mean_score") or 0) / 5
        rows.append({"model": model, "task_success": s["task_success_rate"], "judge_mean": s.get("judge_mean_score"),
                     "p95_latency_s": s["p95_latency_s"], "cost_per_run_usd": cost_per_run(s, prices.get(model)),
                     "quality": 0.5 * s["task_success_rate"] + 0.5 * judge})

    def norm(values, invert=False):
        vals = [v for v in values if v is not None]
        lo, hi = (min(vals), max(vals)) if vals else (0, 0)
        out = []
        for v in values:
            if v is None:
                out.append(0.0)
            elif hi == lo:
                out.append(1.0)
            else:
                x = (v - lo) / (hi - lo)
                out.append(1 - x if invert else x)
        return out

    q = norm([r["quality"] for r in rows])
    lat = norm([r["p95_latency_s"] for r in rows], invert=True)
    cost = norm([r["cost_per_run_usd"] for r in rows], invert=True)
    for r, a, b, c in zip(rows, q, lat, cost):
        r["score"] = round(weights["quality"] * a + weights["latency"] * b + weights["cost"] * c, 4)
        r["eligible"] = r["task_success"] >= min_success
    rows.sort(key=lambda r: (not r["eligible"], -r["score"]))
    return rows


def to_markdown(rows: list[dict]) -> str:
    lines = ["| Rank | Model | Task success | Judge (1-5) | p95 latency (s) | Cost per run ($) | Score |",
             "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        cost = f"{r['cost_per_run_usd']:.5f}" if r["cost_per_run_usd"] is not None else "n/a"
        lines.append(f"| {i}{'' if r['eligible'] else ' (below bar)'} | {r['model']} | {r['task_success']:.0%} | "
                     f"{r['judge_mean'] if r['judge_mean'] is not None else 'n/a'} | {r['p95_latency_s']} | {cost} | {r['score']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    from .run_evals import evaluate

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--prices", default=None)
    ap.add_argument("--min-success", type=float, default=0.75)
    ap.add_argument("--tasks", default=str(HERE / "tasks.jsonl"))
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=2)
    args = ap.parse_args()
    prices = json.loads(Path(args.prices).read_text()) if args.prices else {}
    os.environ["LLM_PROVIDER"] = "bedrock"
    results = {}
    for model in args.models:
        os.environ["BEDROCK_MODEL_ID"] = model
        ns = argparse.Namespace(tasks=args.tasks, max_steps=args.max_steps, concurrency=args.concurrency,
                                judge_model=args.judge_model, no_judge=False, runtime="loop")
        report = asyncio.run(evaluate(ns))
        runs = report["runs"]
        summary = report["summary"]
        summary["avg_prompt_tokens"] = sum(r["prompt_tokens"] for r in runs) / max(len(runs), 1)
        summary["avg_completion_tokens"] = sum(r["tokens"] - r["prompt_tokens"] for r in runs) / max(len(runs), 1)
        results[model] = summary
        print(model, {k: summary[k] for k in ("task_success_rate", "judge_mean_score", "p95_latency_s")})
    rows = rank_models(results, prices, min_success=args.min_success)
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "model_comparison.json").write_text(json.dumps({"ranking": rows, "summaries": results}, indent=2))
    (out / "model_comparison.md").write_text(to_markdown(rows))
    print(to_markdown(rows))
    print(f"Recommended: {rows[0]['model'] if rows and rows[0]['eligible'] else 'none met the quality bar'}")


if __name__ == "__main__":
    main()
