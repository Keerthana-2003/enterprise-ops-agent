"""Progressive-delivery decision logic: metric parsing, quantiles, canary evaluation, dial-up plan."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))
from rollout import Health, Thresholds, dial_up_plan, evaluate, histogram_quantile, parse_prometheus, summarize  # noqa: E402

METRICS = """# HELP http_requests_total HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="POST",route="/runs",status="200"} 95.0
http_requests_total{method="POST",route="/runs",status="500"} 5.0
http_requests_total{method="GET",route="/health",status="200"} 400.0
http_request_duration_seconds_bucket{le="0.5",route="/runs"} 10.0
http_request_duration_seconds_bucket{le="1.0",route="/runs"} 50.0
http_request_duration_seconds_bucket{le="2.5",route="/runs"} 90.0
http_request_duration_seconds_bucket{le="5.0",route="/runs"} 100.0
http_request_duration_seconds_bucket{le="+Inf",route="/runs"} 100.0
http_request_duration_seconds_bucket{le="+Inf",route="/health"} 400.0
"""


def test_parse_and_summarize_ignores_health_checks():
    h = summarize(parse_prometheus(METRICS))
    assert h.requests == 100 and h.errors == 5 and h.error_rate == 0.05
    assert h.p95_s == pytest.approx(2.5 + 2.5 * (95 - 90) / 10)  # interpolated inside the 2.5-5 s bucket


def test_histogram_quantile_edge_cases():
    assert histogram_quantile(0.5, {}) is None
    assert histogram_quantile(0.5, {1.0: 0.0, float("inf"): 0.0}) is None
    assert histogram_quantile(0.99, {1.0: 10.0, float("inf"): 100.0}) == 1.0  # falls in +Inf: report last bound


def test_evaluate_canary_against_baseline():
    t = Thresholds()
    base = Health(1000, 5, 1.0)
    assert evaluate(Health(200, 1, 1.1), base, t).healthy
    bad_errors = evaluate(Health(200, 10, 1.0), base, t)
    assert not bad_errors.healthy and any("error rate" in r for r in bad_errors.reasons)
    slow = evaluate(Health(200, 0, 2.0), base, t)
    assert not slow.healthy and "p95" in slow.reasons[0]
    assert evaluate(Health(5, 0, 9.0), base, t).healthy  # too little traffic to judge latency


def test_dial_up_plan_starts_one_box_and_reaches_full():
    assert dial_up_plan(4) == [(1, 4), (1, 3), (2, 2), (4, 0)]
    assert dial_up_plan(1) == [(1, 1), (1, 0)]


def test_cost_model_arithmetic_and_cache_savings():
    from cost_estimate import Assumptions, estimate

    a = Assumptions(runs_per_day=1000, prompt_tokens=1_000_000, completion_tokens=0, cache_hit_rate=0.5,
                    price_in=1.0, price_cached=0.5, price_out=0.0, pods=0, observability_monthly=0, days_per_month=1)
    r = estimate(a)
    assert r["llm_cost_per_run_usd"] == 0.75  # 0.5M uncached at $1 + 0.5M cached at $0.5
    assert r["monthly_usd"]["total"] == 750.0 and r["prompt_caching_saves_usd_per_month"] == 250.0
