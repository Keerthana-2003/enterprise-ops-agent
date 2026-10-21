"""Run-cost model for proposals: monthly LLM and infrastructure cost from expected volume.

Prices change often and differ by provider, region and contract, so every price is an input.
The defaults are illustrative placeholders; replace them with the client's negotiated rates.

  python deploy/cost_estimate.py --runs-per-day 2000 --prompt-tokens 6000 --completion-tokens 400 \
      --cache-hit-rate 0.6 --price-in 0.15 --price-cached 0.075 --price-out 0.60 --pods 4
Token counts per run come from the eval harness (evals/results/*.json: avg_tokens, prompt_cache_hit_rate).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass


@dataclass
class Assumptions:
    runs_per_day: float = 2000
    prompt_tokens: float = 6000          # per run, all LLM calls in the agent loop
    completion_tokens: float = 400
    cache_hit_rate: float = 0.5          # share of prompt tokens served from the prompt cache
    price_in: float = 0.15               # $ per 1M uncached input tokens (placeholder)
    price_cached: float = 0.075          # $ per 1M cached input tokens (placeholder)
    price_out: float = 0.60              # $ per 1M output tokens (placeholder)
    pods: int = 4
    vcpu_per_pod: float = 1.0
    price_vcpu_hour: float = 0.05        # placeholder blended compute price
    observability_monthly: float = 200.0  # tracing / metrics (placeholder)
    days_per_month: float = 30.0


def estimate(a: Assumptions) -> dict:
    runs = a.runs_per_day * a.days_per_month
    cached = a.prompt_tokens * a.cache_hit_rate
    uncached = a.prompt_tokens - cached
    per_run = (uncached * a.price_in + cached * a.price_cached + a.completion_tokens * a.price_out) / 1e6
    llm = per_run * runs
    no_cache = (a.prompt_tokens * a.price_in + a.completion_tokens * a.price_out) / 1e6 * runs
    infra = a.pods * a.vcpu_per_pod * a.price_vcpu_hour * 24 * a.days_per_month
    total = llm + infra + a.observability_monthly
    return {
        "runs_per_month": round(runs),
        "llm_cost_per_run_usd": round(per_run, 5),
        "monthly_usd": {"llm": round(llm, 2), "infrastructure": round(infra, 2),
                        "observability": round(a.observability_monthly, 2), "total": round(total, 2)},
        "prompt_caching_saves_usd_per_month": round(no_cache - llm, 2),
        "cost_per_run_all_in_usd": round(total / runs, 4) if runs else None,
        "assumptions": asdict(a),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    for name, default in asdict(Assumptions()).items():
        ap.add_argument("--" + name.replace("_", "-"), type=type(default), default=default)
    print(json.dumps(estimate(Assumptions(**vars(ap.parse_args()))), indent=2))


if __name__ == "__main__":
    main()
